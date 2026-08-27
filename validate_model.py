#!/usr/bin/env python3
"""Deterministic integrity, monotonicity and sensitivity checks for Model v3."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from math import isfinite
from pathlib import Path


WEIGHT_SETS = {
    "legacy_default": {"peak_v3": 25, "career_v3": 25, "playoff_v3": 20, "recognition_v3": 10, "championship_v3": 10, "durability_v3": 10},
    "individual_impact": {"peak_v3": 40, "career_v3": 35, "playoff_v3": 25, "recognition_v3": 0, "championship_v3": 0, "durability_v3": 0},
    "resume": {"peak_v3": 20, "career_v3": 15, "playoff_v3": 20, "recognition_v3": 20, "championship_v3": 20, "durability_v3": 5},
    "longevity": {"peak_v3": 20, "career_v3": 30, "playoff_v3": 15, "recognition_v3": 10, "championship_v3": 10, "durability_v3": 15},
}


def score(player: dict[str, object], weights: dict[str, int]) -> float:
    return sum(float(player[key]) * weight for key, weight in weights.items()) / sum(weights.values())


def championship_credits(fmvp: int, titles: int, playoff_score: float) -> float:
    return fmvp + max(0, titles - fmvp) * (0.10 + 0.55 * playoff_score / 100)


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
    assert {int(row["season"]) for row in playoffs} >= set(range(1950, 2027))

    fields = tuple(next(iter(WEIGHT_SETS.values())))
    grades = Counter()
    for player in players:
        grades[str(player["confidence_grade"])] += 1
        for field in fields:
            value = float(player[field])
            assert isfinite(value) and 0 <= value <= 100
        assert 0 <= float(player["data_coverage"]) <= 1
        assert float(player["model_uncertainty"]) >= 0
        assert int(player["fmvp"]) <= int(player["champs"]) <= int(player["seasons"])

    # Monotonicity: improving the input while holding everything else fixed
    # must never lower championship contribution.
    for playoff_score in range(0, 101, 5):
        assert championship_credits(1, 1, playoff_score) >= championship_credits(0, 1, playoff_score)
        assert championship_credits(0, 2, playoff_score) >= championship_credits(0, 1, playoff_score)
    assert all(
        championship_credits(0, 1, right) >= championship_credits(0, 1, left)
        for left, right in zip(range(0, 100, 5), range(5, 105, 5))
    )

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
    bands_overlap = (
        nash_total + float(nash["model_uncertainty"]) >= kawhi_total - float(kawhi["model_uncertainty"])
        and kawhi_total + float(kawhi["model_uncertainty"]) >= nash_total - float(nash["model_uncertainty"])
    )
    assert abs(float(nash["peak_v3"]) - float(kawhi["peak_v3"])) < 1
    assert float(kawhi["playoff_v3"]) > float(nash["playoff_v3"])
    assert bands_overlap

    barbosa = lookup["Leandro Barbosa"]
    assert (barbosa["mvp"], barbosa["fmvp"], barbosa["champs"], barbosa["nba1"]) == (0, 0, 1, 0)
    assert score(lookup["Robert Horry"], WEIGHT_SETS["legacy_default"]) < 50

    report = {
        "model": "v3",
        "players": len(players),
        "playoff_rows": len(playoffs),
        "playoff_seasons": [min(int(row["season"]) for row in playoffs), max(int(row["season"]) for row in playoffs)],
        "confidence_grades": dict(sorted(grades.items())),
        "top100_overlap_vs_default": top100_overlap,
        "nash_kawhi": {
            "peak_gap": round(float(kawhi["peak_v3"]) - float(nash["peak_v3"]), 2),
            "total_gap": round(kawhi_total - nash_total, 2),
            "uncertainty_bands_overlap": bands_overlap,
        },
        "checks": ["ranges", "unique_ids", "complete_playoff_years", "championship_monotonicity", "weight_sensitivity", "known-record regression"],
        "status": "passed",
    }
    if args.report:
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
