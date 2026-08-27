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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("historical_html", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--latest-html", type=Path)
    parser.add_argument("--latest-season", type=int, default=2026)
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
            }
            for old, new in stat_map.items():
                target[new] = optional_float(source.get(old))
            grouped[key] = target

    output = sorted(
        (row for row in grouped.values() if int(row.get("gp") or 0) > 0),
        key=lambda row: (int(row["season"]), str(row["name"])),
    )
    args.output.write_text(json.dumps(output, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")

    seasons = {int(row["season"]) for row in output}
    assert min(seasons) == 1950
    assert max(seasons) == args.latest_season
    assert all(int(row["gp"] or 0) > 0 for row in output)
    print(json.dumps({"rows": len(output), "players": len({row["name"] for row in output}), "from": min(seasons), "to": max(seasons)}))


if __name__ == "__main__":
    main()
