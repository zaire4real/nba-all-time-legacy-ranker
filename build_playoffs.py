#!/usr/bin/env python3
"""Extract a compact, reproducible all-time playoff season table.

The historical source is the embedded seed table from nba-stats-lab.  A final
Basketball-Reference per-game page can be supplied to append the latest season.
Only fields used by the scoring model are retained.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from html.parser import HTMLParser
import json
from pathlib import Path
import re


SEED_RE = re.compile(r'<script id="seed-data" type="application/json">(.*?)</script>', re.S)
METRIC_MAP = {
    "points": "pts",
    "rebounds": "trb",
    "assists": "ast",
    "blocks": "blk",
    "fg_pct": "fg_pct",
    "turnovers": "tov",
}


def optional_float(value: object) -> float | None:
    if value in {None, ""}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class LatestTableParser(HTMLParser):
    """Read the per_game_stats table without an external HTML dependency."""

    def __init__(self) -> None:
        super().__init__()
        self.in_table = False
        self.in_row = False
        self.cell_stat: str | None = None
        self.cell_text: list[str] = []
        self.row: dict[str, str] = {}
        self.rows: list[dict[str, str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if tag == "table" and attributes.get("id") == "per_game_stats":
            self.in_table = True
        elif self.in_table and tag == "tr":
            self.in_row = True
            self.row = {}
        elif self.in_row and tag in {"th", "td"}:
            self.cell_stat = attributes.get("data-stat")
            self.cell_text = []

    def handle_data(self, data: str) -> None:
        if self.cell_stat:
            self.cell_text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if self.in_row and tag in {"th", "td"} and self.cell_stat:
            self.row[self.cell_stat] = "".join(self.cell_text).strip()
            self.cell_stat = None
        elif self.in_row and tag == "tr":
            if self.row.get("player") and self.row.get("g"):
                self.rows.append(self.row)
            self.in_row = False
        elif self.in_table and tag == "table":
            self.in_table = False


class ChampionsParser(HTMLParser):
    """Extract season, champion team code and Finals MVP player id."""

    def __init__(self) -> None:
        super().__init__()
        self.in_table = False
        self.in_row = False
        self.cell_stat: str | None = None
        self.cell_text: list[str] = []
        self.cell_href: str | None = None
        self.row: dict[str, str] = {}
        self.rows: list[dict[str, str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if tag == "table" and attributes.get("id") == "champions_index":
            self.in_table = True
        elif self.in_table and tag == "tr":
            self.in_row = True
            self.row = {}
        elif self.in_row and tag in {"th", "td"}:
            self.cell_stat = attributes.get("data-stat")
            self.cell_text = []
            self.cell_href = None
        elif self.in_row and tag == "a" and self.cell_stat:
            self.cell_href = attributes.get("href")

    def handle_data(self, data: str) -> None:
        if self.cell_stat:
            self.cell_text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if self.in_row and tag in {"th", "td"} and self.cell_stat:
            self.row[self.cell_stat] = "".join(self.cell_text).strip()
            if self.cell_href:
                self.row[f"{self.cell_stat}_href"] = self.cell_href
            self.cell_stat = None
        elif self.in_row and tag == "tr":
            if self.row.get("year_id") and self.row.get("champion_href"):
                self.rows.append(self.row)
            self.in_row = False
        elif self.in_table and tag == "table":
            self.in_table = False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("historical_html", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--latest-html", type=Path)
    parser.add_argument("--latest-season", type=int, default=2026)
    parser.add_argument("--champions-html", type=Path, required=True)
    args = parser.parse_args()

    match = SEED_RE.search(args.historical_html.read_text(encoding="utf-8", errors="ignore"))
    if not match:
        raise RuntimeError("Embedded seed-data table not found")
    seed = json.loads(match.group(1))

    grouped: dict[tuple[str, int], dict[str, object]] = defaultdict(dict)
    for row in seed:
        if row.get("season_type") != "Playoffs" or row.get("metric_key") not in METRIC_MAP:
            continue
        key = (str(row["player_name"]).strip(), int(row["season_start_year"]) + 1)
        target = grouped[key]
        target.update({"name": key[0], "season": key[1]})
        if row.get("team_abbreviation"):
            target["team"] = str(row["team_abbreviation"])
        if optional_float(row.get("gp")) is not None:
            target["gp"] = optional_float(row.get("gp"))
        if optional_float(row.get("min")) is not None:
            target["mpg"] = optional_float(row.get("min"))
        metric = METRIC_MAP[str(row["metric_key"])]
        target[metric] = optional_float(row.get("player_value"))
        target[f"{metric}_adj"] = optional_float(row.get("adjusted_score"))

    if args.latest_html:
        latest = LatestTableParser()
        latest.feed(args.latest_html.read_text(encoding="utf-8", errors="ignore"))
        stat_map = {
            "mp_per_g": "mpg",
            "pts_per_g": "pts",
            "trb_per_g": "trb",
            "ast_per_g": "ast",
            "blk_per_g": "blk",
            "fg_pct": "fg_pct",
            "tov_per_g": "tov",
        }
        for source in latest.rows:
            key = (source["player"].strip(), args.latest_season)
            target: dict[str, object] = {
                "name": key[0],
                "season": key[1],
                "gp": optional_float(source.get("g")),
                "team": source.get("team_id"),
            }
            for old, new in stat_map.items():
                target[new] = optional_float(source.get(old))
            grouped[key] = target

    champions_parser = ChampionsParser()
    champions_parser.feed(args.champions_html.read_text(encoding="utf-8", errors="ignore"))
    champions: dict[int, tuple[str, str | None]] = {}
    for row in champions_parser.rows:
        if row.get("lg_id") not in {"NBA", "BAA"}:
            continue
        team_match = re.search(r"/teams/([^/]+)/", row.get("champion_href", ""))
        fmvp_match = re.search(r"/players/[^/]+/([^/.]+)\.html", row.get("mvp_finals_href", ""))
        if team_match:
            champions[int(row["year_id"])] = (team_match.group(1), fmvp_match.group(1) if fmvp_match else None)

    for row in grouped.values():
        season = int(row["season"])
        champion = champions.get(season)
        if champion and row.get("team") == champion[0]:
            row["champion"] = True
            if champion[1]:
                row["fmvp_player_id"] = champion[1]

    output = sorted(
        (row for row in grouped.values() if int(row.get("gp") or 0) > 0),
        key=lambda row: (int(row["season"]), str(row["name"])),
    )
    args.output.write_text(json.dumps(output, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")

    seasons = {int(row["season"]) for row in output}
    assert min(seasons) == 1950
    assert max(seasons) == args.latest_season
    assert all(int(row["gp"] or 0) > 0 for row in output)
    champion_seasons = {int(row["season"]) for row in output if row.get("champion")}
    assert champion_seasons == set(range(1950, args.latest_season + 1))
    print(json.dumps({"rows": len(output), "players": len({row["name"] for row in output}), "from": min(seasons), "to": max(seasons), "champion_seasons": len(champion_seasons)}))


if __name__ == "__main__":
    main()
