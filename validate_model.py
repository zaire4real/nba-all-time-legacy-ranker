#!/usr/bin/env python3
"""Deterministic integrity, role matching and sensitivity checks for Model v4."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from math import exp, isfinite
from pathlib import Path


WEIGHT_SETS = {
    "legacy_default": {"peak_v4": 27, "career_v4": 28, "playoff_v4": 25, "recognition_v4": 8, "championship_v4": 5, "durability_v4": 7},
    "individual_impact": {"peak_v4": 40, "career_v4": 35, "playoff_v4": 25, "recognition_v4": 0, "championship_v4": 0, "durability_v4": 0},
    "awards_and_titles": {"peak_v4": 20, "career_v4": 15, "playoff_v4": 20, "recognition_v4": 20, "championship_v4": 20, "durability_v4": 5},
    "longevity": {"peak_v4": 20, "career_v4": 30, "playoff_v4": 15, "recognition_v4": 10, "championship_v4": 10, "durability_v4": 15},
}


def score(player: dict[str, object], weights: dict[str, int]) -> float:
    return sum(float(player[key]) * weight for key, weight in weights.items()) / sum(weights.values())


def soft_score(credits: float, scale: float) -> float:
    return 100 * (1 - exp(-max(0.0, credits) / scale))


def role_years(player: dict[str, object]) -> list[dict[str, object]]:
    return list(player["championship_role_years"])  # type: ignore[arg-type]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("players", type=Path)
    parser.add_argument("playoffs", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    players = json.loads(args.players.read_text(encoding="utf-8"))
    playoffs = json.loads(args.playoffs.read_text(encoding="utf-8"))
    ids = [player["id"] for player in players]
    assert len(ids) == len(set(ids))
    assert len(players) >= 5_400
    playoff_seasons = {int(row["season"]) for row in playoffs}
    champion_seasons = {int(row["season"]) for row in playoffs if row.get("champion")}
    assert playoff_seasons >= set(range(1950, 2027))
    assert champion_seasons == set(range(1950, 2027))

    fields = tuple(next(iter(WEIGHT_SETS.values())))
    grades = Counter()
    title_players = 0
    complete_title_roles = 0
    for player in players:
        grades[str(player["confidence_grade"])] += 1
        for field in fields:
            value = float(player[field])
            assert isfinite(value) and 0 <= value <= 100
        assert 0 <= float(player["data_coverage"]) <= 1
        assert 0 <= float(player["title_role_coverage"]) <= 1
        assert float(player["model_uncertainty"]) >= 0
        assert 1 <= int(player["weight_rank_low"]) <= int(player["weight_rank_median"]) <= int(player["weight_rank_high"]) <= len(players)
        assert 0 <= float(player["top100_probability"]) <= 1
        assert int(player["weight_sample_count"]) == 800
        assert int(player["fmvp"]) <= int(player["champs"]) <= int(player["seasons"])
        roles = role_years(player)
        assert len({int(row["season"]) for row in roles}) == len(roles)
        assert all(0 < float(row["credit"]) <= 1 for row in roles)
        if int(player["champs"]):
            title_players += 1
            complete_title_roles += float(player["title_role_coverage"]) == 1

    # Soft curves are monotone and approach, but never prematurely hit, 100.
    # Published scales put the former full-score threshold near 80.
    scales = {"career": (8.70, 14), "playoff_accumulation": (4.66, 7.5), "recognition": (6.21, 10), "championship": (2.49, 4), "durability": (11.18, 18)}
    for scale, former_threshold in scales.values():
        curve = [soft_score(value, scale) for value in range(0, 51)]
        assert all(left <= right < 100 for left, right in zip(curve, curve[1:]))
        assert 62.5 <= soft_score(scale, scale) <= 63.5
        assert 79 <= soft_score(former_threshold, scale) <= 81

    rankings: dict[str, list[dict[str, object]]] = {}
    for name, weights in WEIGHT_SETS.items():
        rankings[name] = sorted(players, key=lambda player: (-score(player, weights), str(player["name"])))
    default_top = {player["id"] for player in rankings["legacy_default"][:100]}
    top100_overlap = {
        name: len(default_top & {player["id"] for player in ranking[:100]})
        for name, ranking in rankings.items()
    }
    assert min(top100_overlap.values()) >= 65

    lookup = {str(player["name"]): player for player in players}
    nash, kawhi = lookup["Steve Nash"], lookup["Kawhi Leonard"]
    nash_total = score(nash, WEIGHT_SETS["legacy_default"])
    kawhi_total = score(kawhi, WEIGHT_SETS["legacy_default"])
    assert abs(float(nash["peak_v4"]) - float(kawhi["peak_v4"])) < 1
    assert float(kawhi["peak_v4"]) > float(nash["peak_v4"])
    assert float(kawhi["playoff_v4"]) > float(nash["playoff_v4"])
    assert kawhi_total > nash_total

    # Known records catch the original data error and verify actual title-year matching.
    barbosa = lookup["Leandro Barbosa"]
    assert (barbosa["mvp"], barbosa["fmvp"], barbosa["champs"], barbosa["nba1"], barbosa["allstar"]) == (0, 0, 1, 0, 0)
    expected_roles = {
        "Kawhi Leonard": ({2014, 2019}, 2),
        "LeBron James": ({2012, 2013, 2016, 2020}, 4),
        "Michael Jordan": ({1991, 1992, 1993, 1996, 1997, 1998}, 6),
        "Robert Horry": ({1994, 1995, 2000, 2001, 2002, 2005, 2007}, 0),
    }
    for name, (seasons, fmvp_count) in expected_roles.items():
        rows = role_years(lookup[name])
        assert {int(row["season"]) for row in rows} == seasons
        assert sum(bool(row["fmvp"]) for row in rows) == fmvp_count
        assert float(lookup[name]["title_role_coverage"]) == 1
    assert score(lookup["Robert Horry"], WEIGHT_SETS["legacy_default"]) < 50
    assert float(lookup["Robert Horry"]["championship_credits_v4"]) < float(lookup["Kawhi Leonard"]["championship_credits_v4"]) + 0.5

    report = {
        "model": "v4",
        "players": len(players),
        "playoff_rows": len(playoffs),
        "playoff_seasons": [min(playoff_seasons), max(playoff_seasons)],
        "champion_seasons_matched": len(champion_seasons),
        "title_role_coverage": {
            "players_with_titles": title_players,
            "complete_players": complete_title_roles,
            "complete_rate": round(complete_title_roles / title_players, 4),
        },
        "confidence_grades": dict(sorted(grades.items())),
        "top100_overlap_vs_default": top100_overlap,
        "nash_kawhi": {
            "peak_gap_kawhi_minus_nash": round(float(kawhi["peak_v4"]) - float(nash["peak_v4"]), 2),
            "total_gap_kawhi_minus_nash": round(kawhi_total - nash_total, 2),
            "weight_rank_intervals": {
                "Steve Nash": [nash["weight_rank_low"], nash["weight_rank_high"]],
                "Kawhi Leonard": [kawhi["weight_rank_low"], kawhi["weight_rank_high"]],
            },
        },
        "weight_robustness_samples": int(players[0]["weight_sample_count"]),
        "checks": ["ranges", "unique_ids", "complete_playoff_years", "complete_champion_years", "diminishing_return_monotonicity", "title_year_role_matching", "bounded_weight_sampling", "weight_sensitivity", "known-record_regression"],
        "status": "passed",
    }
    if args.report:
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
