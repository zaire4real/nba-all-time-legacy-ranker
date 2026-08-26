#!/usr/bin/env python3
"""Build the browser-ready player dataset from name-keyed source tables."""

from __future__ import annotations

import argparse
import csv
import json
import re
import unicodedata
import zipfile
from collections import defaultdict
from pathlib import Path


LEAGUES = {"NBA", "ABA", "BAA"}
MULTI_TEAM = re.compile(r"^\d+TM$")


def norm(value: str) -> str:
    value = unicodedata.normalize("NFD", value or "")
    value = "".join(char for char in value if unicodedata.category(char) != "Mn")
    return re.sub(r"[^a-z0-9]", "", value.lower())


def number(value: str | None) -> float:
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0


def integer(value: str | None) -> int:
    return int(number(value))


def rows_from_zip(archive: zipfile.ZipFile, name: str) -> list[dict[str, str]]:
    with archive.open(name) as raw:
        text = (line.decode("utf-8-sig") for line in raw)
        return list(csv.DictReader(text))


def rows_from_file(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def select_total_row(rows: list[dict[str, str]]) -> dict[str, str] | None:
    if not rows:
        return None
    total = next((row for row in rows if MULTI_TEAM.match((row.get("team") or "").upper())), None)
    if total:
        return total
    if len(rows) == 1:
        return rows[0]
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("legacy_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    with zipfile.ZipFile(args.archive) as archive:
        career_rows = rows_from_zip(archive, "Player Career Info.csv")
        total_rows = rows_from_zip(archive, "Player Totals.csv")
        team_rows = rows_from_zip(archive, "End of Season Teams.csv")
        award_rows = rows_from_zip(archive, "Player Award Shares.csv")
        star_rows = rows_from_zip(archive, "All-Star Selections.csv")
        advanced_rows = rows_from_zip(archive, "Advanced.csv")

    players: dict[str, dict[str, object]] = {}
    for row in career_rows:
        player_id = row.get("player_id", "").strip()
        if not player_id:
            continue
        players[player_id] = {
            "id": player_id,
            "name": row.get("player", "").strip(),
            "seasons": 0,
            "first_seas": integer(row.get("from")),
            "last_seas": integer(row.get("to")),
            "ppg": 0,
            "apg": 0,
            "rpg": 0,
            "spg": 0,
            "bpg": 0,
            "mvp": 0,
            "fmvp": 0,
            "champs": 0,
            "nba1": 0,
            "nba2": 0,
            "nba3": 0,
            "def1": 0,
            "def2": 0,
            "dpoy": 0,
            "allstar": 0,
            "per": None,
            "ws": None,
            "ws48": None,
            "bpm": None,
            "vorp": None,
        }

    season_groups: dict[tuple[str, int, str], list[dict[str, str]]] = defaultdict(list)
    for row in total_rows:
        player_id = row.get("player_id", "").strip()
        league = (row.get("lg") or "").upper()
        if player_id in players and league in LEAGUES:
            season_groups[(player_id, integer(row.get("season")), league)].append(row)

    accumulated: dict[str, dict[str, object]] = defaultdict(
        lambda: {"g": 0.0, "pts": 0.0, "trb": 0.0, "ast": 0.0, "stl": 0.0, "blk": 0.0, "seasons": set()}
    )
    for (player_id, season, _league), rows in season_groups.items():
        total = select_total_row(rows)
        selected = [total] if total else rows
        bucket = accumulated[player_id]
        for row in selected:
            for field in ("g", "pts", "trb", "ast", "stl", "blk"):
                bucket[field] = float(bucket[field]) + number(row.get(field))
        cast_seasons = bucket["seasons"]
        assert isinstance(cast_seasons, set)
        cast_seasons.add(season)

    for player_id, bucket in accumulated.items():
        player = players[player_id]
        games = float(bucket["g"])
        if games:
            player["ppg"] = round(float(bucket["pts"]) / games, 3)
            player["rpg"] = round(float(bucket["trb"]) / games, 3)
            player["apg"] = round(float(bucket["ast"]) / games, 3)
            player["spg"] = round(float(bucket["stl"]) / games, 3)
            player["bpg"] = round(float(bucket["blk"]) / games, 3)
        player["seasons"] = len(bucket["seasons"])

    for player in players.values():
        if not player["seasons"] and player["first_seas"] and player["last_seas"]:
            player["seasons"] = int(player["last_seas"]) - int(player["first_seas"]) + 1

    for row in team_rows:
        player = players.get((row.get("player_id") or "").strip())
        if not player:
            continue
        team_type = row.get("type")
        number_tm = row.get("number_tm")
        if team_type in {"All-NBA", "All-ABA", "All-BAA"} and number_tm in {"1st", "2nd", "3rd"}:
            key = {"1st": "nba1", "2nd": "nba2", "3rd": "nba3"}[number_tm]
            player[key] = int(player[key]) + 1
        elif team_type == "All-Defense" and number_tm in {"1st", "2nd"}:
            key = {"1st": "def1", "2nd": "def2"}[number_tm]
            player[key] = int(player[key]) + 1

    for row in award_rows:
        if (row.get("winner") or "").upper() != "TRUE":
            continue
        player = players.get((row.get("player_id") or "").strip())
        if not player:
            continue
        award = (row.get("award") or "").lower()
        if award in {"nba mvp", "aba mvp"}:
            player["mvp"] = int(player["mvp"]) + 1
        elif award == "nba dpoy":
            player["dpoy"] = int(player["dpoy"]) + 1

    star_seasons: dict[str, set[tuple[str, int]]] = defaultdict(set)
    for row in star_rows:
        player_id = (row.get("player_id") or "").strip()
        league = (row.get("lg") or "").upper()
        if player_id in players and league in LEAGUES:
            star_seasons[player_id].add((league, integer(row.get("season"))))
    for player_id, selections in star_seasons.items():
        players[player_id]["allstar"] = len(selections)

    by_name_start: dict[tuple[str, int], dict[str, object]] = {}
    by_name: dict[str, list[dict[str, object]]] = defaultdict(list)
    for player in players.values():
        name_key = norm(str(player["name"]))
        by_name_start[(name_key, int(player["first_seas"]))] = player
        by_name[name_key].append(player)

    legacy_champs = rows_from_file(args.legacy_dir / "MVPs and Championships.csv")
    for row in legacy_champs:
        name_key = norm(row.get("player", ""))
        player = by_name_start.get((name_key, integer(row.get("first_seas"))))
        if not player and len(by_name[name_key]) == 1:
            player = by_name[name_key][0]
        if player:
            player["fmvp"] = integer(row.get("Finals MVP"))
            player["champs"] = integer(row.get("Championships"))

    def patch(names: list[str], key: str) -> None:
        for name in names:
            matches = by_name.get(norm(name), [])
            if len(matches) != 1:
                raise RuntimeError(f"Cannot uniquely patch {name!r}: {len(matches)} matches")
            matches[0][key] = int(matches[0][key]) + 1

    patch(["Shai Gilgeous-Alexander"], "mvp")
    patch(["Victor Wembanyama"], "dpoy")
    patch(["Shai Gilgeous-Alexander", "Nikola Jokić", "Victor Wembanyama", "Luka Dončić", "Cade Cunningham"], "nba1")
    patch(["Jaylen Brown", "Kawhi Leonard", "Donovan Mitchell", "Kevin Durant", "Jalen Brunson"], "nba2")
    patch(["Tyrese Maxey", "Jamal Murray", "Jalen Johnson", "Jalen Duren", "Chet Holmgren"], "nba3")
    patch(["Victor Wembanyama", "Chet Holmgren", "Ausar Thompson", "Rudy Gobert", "Derrick White"], "def1")
    patch(["Scottie Barnes", "Cason Wallace", "Bam Adebayo", "OG Anunoby", "Dyson Daniels"], "def2")

    champions = {
        2023: ["Bruce Brown", "Thomas Bryant", "Kentavious Caldwell-Pope", "Vlatko Čančar", "Aaron Gordon", "Jeff Green", "Reggie Jackson", "Nikola Jokić", "DeAndre Jordan", "Jamal Murray", "Zeke Nnaji", "Michael Porter Jr.", "Ish Smith", "Peyton Watson", "Jack White"],
        2024: ["Oshae Brissett", "Jaylen Brown", "JD Davison", "Sam Hauser", "Jrue Holiday", "Al Horford", "Luke Kornet", "Svi Mykhailiuk", "Drew Peterson", "Kristaps Porziņģis", "Payton Pritchard", "Neemias Queta", "Jayson Tatum", "Xavier Tillman Sr.", "Jordan Walsh", "Derrick White"],
        2025: ["Shai Gilgeous-Alexander", "Jalen Williams", "Chet Holmgren", "Luguentz Dort", "Alex Caruso", "Isaiah Hartenstein", "Isaiah Joe", "Cason Wallace", "Aaron Wiggins", "Jaylin Williams", "Kenrich Williams", "Ousmane Dieng", "Branden Carlson", "Alex Ducas", "Adam Flagler"],
        2026: ["Jose Alvarado", "OG Anunoby", "Mikal Bridges", "Jalen Brunson", "Jordan Clarkson", "Pacome Dadiet", "Mohamed Diawara", "Tosan Evbuomwan", "Josh Hart", "Miles McBride", "Mitchell Robinson", "Karl-Anthony Towns", "Tyler Kolek", "Landry Shamet", "Ariel Hukporti", "Guerschon Yabusele", "Jeremy Sochan", "Kevin McCullar Jr."],
    }
    for roster in champions.values():
        patch(roster, "champs")
    patch(["Nikola Jokić", "Jaylen Brown", "Shai Gilgeous-Alexander", "Jalen Brunson"], "fmvp")

    advanced_2026: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in advanced_rows:
        if integer(row.get("season")) == 2026 and (row.get("lg") or "").upper() == "NBA":
            advanced_2026[(row.get("player_id") or "").strip()].append(row)
    for player_id, rows in advanced_2026.items():
        player = players.get(player_id)
        if not player:
            continue
        row = select_total_row(rows) or max(rows, key=lambda item: number(item.get("g")))
        player["per"] = number(row.get("per")) if row.get("per") not in {"", "NA", None} else None
        player["ws"] = number(row.get("ws")) if row.get("ws") not in {"", "NA", None} else None
        player["ws48"] = number(row.get("ws_48")) if row.get("ws_48") not in {"", "NA", None} else None
        player["bpm"] = number(row.get("bpm")) if row.get("bpm") not in {"", "NA", None} else None
        player["vorp"] = number(row.get("vorp")) if row.get("vorp") not in {"", "NA", None} else None

    output = sorted(players.values(), key=lambda player: (str(player["name"]), str(player["id"])))
    args.output.write_text(json.dumps(output, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")

    lookup = {norm(str(player["name"])): player for player in output}
    barbosa = lookup["leandrobarbosa"]
    lebron = lookup["lebronjames"]
    assert (barbosa["mvp"], barbosa["fmvp"], barbosa["champs"], barbosa["nba1"]) == (0, 0, 1, 0)
    assert 10.5 < float(barbosa["ppg"]) < 10.7
    assert (lebron["mvp"], lebron["fmvp"], lebron["champs"], lebron["nba1"]) == (4, 4, 4, 13)
    print(json.dumps({"players": len(output), "barbosa": barbosa, "lebron": lebron}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
