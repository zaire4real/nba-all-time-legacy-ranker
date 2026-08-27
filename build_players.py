#!/usr/bin/env python3
"""Build the browser-ready player dataset from name-keyed source tables."""

from __future__ import annotations

import argparse
from bisect import bisect_left, bisect_right
import csv
import json
from math import sqrt
import random
import re
from statistics import fmean
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


def optional_number(value: str | None) -> float | None:
    if value in {"", "NA", None}:
        return None
    return number(value)


def percentile_against(values: list[float], value: float | None) -> float | None:
    """Midrank empirical percentile, robust to ties and season-to-season scale changes."""
    if value is None or not values:
        return None
    left = bisect_left(values, value)
    right = bisect_right(values, value)
    return 100 * (left + (right - left) / 2) / len(values)


def build_model_v3(
    players: dict[str, dict[str, object]],
    total_rows: list[dict[str, str]],
    advanced_rows: list[dict[str, str]],
    team_rows: list[dict[str, str]],
    award_rows: list[dict[str, str]],
    star_rows: list[dict[str, str]],
    team_summary_rows: list[dict[str, str]],
    playoff_rows: list[dict[str, object]],
) -> None:
    """Add era-normalized season-level Model v2 and v3 component scores.

    Peak uses the best three season scores and no awards. Season scores combine
    within-league percentiles for impact rate, value volume, box production and
    availability. Other categories use non-overlapping season credits where
    possible, which avoids counting MVP + All-NBA + All-Star three times in the
    same year. Model v3 separates regular-season peak, career impact, actual
    playoff performance, recognition, championship contribution and durability.
    This prevents an MVP, All-NBA berth and championship from being silently
    counted as three measurements of the same underlying thing.
    """

    total_groups: dict[tuple[str, int, str], list[dict[str, str]]] = defaultdict(list)
    advanced_groups: dict[tuple[str, int, str], list[dict[str, str]]] = defaultdict(list)
    for row in total_rows:
        player_id = (row.get("player_id") or "").strip()
        league = (row.get("lg") or "").upper()
        if player_id in players and league in LEAGUES:
            total_groups[(player_id, integer(row.get("season")), league)].append(row)
    for row in advanced_rows:
        player_id = (row.get("player_id") or "").strip()
        league = (row.get("lg") or "").upper()
        if player_id in players and league in LEAGUES:
            advanced_groups[(player_id, integer(row.get("season")), league)].append(row)

    schedule_games: dict[tuple[int, str], int] = {}
    for row in team_summary_rows:
        league = (row.get("lg") or "").upper()
        if league not in LEAGUES:
            continue
        key = (integer(row.get("season")), league)
        schedule_games[key] = max(schedule_games.get(key, 0), integer(row.get("w")) + integer(row.get("l")))

    records: list[dict[str, object]] = []
    for (player_id, season, league), rows in total_groups.items():
        total = select_total_row(rows)
        selected = [total] if total else rows
        games = sum(number(row.get("g")) for row in selected)
        minutes = sum(number(row.get("mp")) for row in selected)
        points = sum(number(row.get("pts")) for row in selected)
        rebounds = sum(number(row.get("trb")) for row in selected)
        assists = sum(number(row.get("ast")) for row in selected)
        advanced = advanced_groups.get((player_id, season, league), [])
        advanced_row = select_total_row(advanced) or (max(advanced, key=lambda row: number(row.get("mp"))) if advanced else None)
        schedule = schedule_games.get((season, league), 0) or max(integer(row.get("g")) for row in rows)
        availability = min(1.0, games / schedule) if schedule else 0.0
        box_rate = (points + 1.2 * rebounds + 1.5 * assists) / games if games else 0.0
        records.append(
            {
                "player_id": player_id,
                "season": season,
                "league": league,
                "g": games,
                "mp": minutes,
                "schedule": schedule,
                "availability": availability,
                "box_rate": box_rate,
                "box_total": points + 1.2 * rebounds + 1.5 * assists,
                "per": optional_number(advanced_row.get("per")) if advanced_row else None,
                "ws48": optional_number(advanced_row.get("ws_48")) if advanced_row else None,
                "bpm": optional_number(advanced_row.get("bpm")) if advanced_row else None,
                "ws": optional_number(advanced_row.get("ws")) if advanced_row else None,
                "vorp": optional_number(advanced_row.get("vorp")) if advanced_row else None,
                "dws": optional_number(advanced_row.get("dws")) if advanced_row else None,
                "dbpm": optional_number(advanced_row.get("dbpm")) if advanced_row else None,
            }
        )

    cohorts: dict[tuple[int, str], list[dict[str, object]]] = defaultdict(list)
    for record in records:
        cohorts[(int(record["season"]), str(record["league"]))].append(record)

    metric_names = ("per", "ws48", "bpm", "ws", "vorp", "dws", "dbpm", "box_rate", "box_total")
    distributions: dict[tuple[int, str], dict[str, list[float]]] = {}
    for key, cohort in cohorts.items():
        schedule = max(int(record["schedule"]) for record in cohort)
        qualified = [
            record
            for record in cohort
            if float(record["mp"]) >= max(300, schedule * 6) and float(record["g"]) >= max(12, schedule * 0.2)
        ]
        if len(qualified) < 10:
            qualified = [record for record in cohort if float(record["mp"]) > 0]
        distributions[key] = {
            metric: sorted(float(record[metric]) for record in qualified if record[metric] is not None)
            for metric in metric_names
        }

    for record in records:
        key = (int(record["season"]), str(record["league"]))
        pct = {metric: percentile_against(distributions[key][metric], record[metric]) for metric in metric_names}
        advanced_rate = [pct[metric] for metric in ("per", "ws48", "bpm") if pct[metric] is not None]
        advanced_volume = [pct[metric] for metric in ("ws", "vorp") if pct[metric] is not None]
        raw_rate = fmean(advanced_rate) if advanced_rate else float(pct["box_rate"] or 0)
        reliability = min(1.0, float(record["mp"]) / max(1, int(record["schedule"]) * 24))
        rate = 50 + (raw_rate - 50) * reliability
        volume = fmean(advanced_volume) if advanced_volume else float(pct["box_total"] or 0)
        box_rate_pct = float(pct["box_rate"] or 0)
        availability_pct = float(record["availability"]) * 100
        record["season_score"] = 0.50 * rate + 0.25 * volume + 0.15 * box_rate_pct + 0.10 * availability_pct
        record["production_pct"] = box_rate_pct
        defense_metrics = [pct[metric] for metric in ("dws", "dbpm") if pct[metric] is not None]
        record["defense_metric_pct"] = fmean(defense_metrics) if defense_metrics else None
        record["advanced_coverage"] = len(advanced_rate + advanced_volume) / 5

    all_nba: dict[tuple[str, int], float] = {}
    all_defense: dict[tuple[str, int], float] = {}
    for row in team_rows:
        player_id = (row.get("player_id") or "").strip()
        if player_id not in players:
            continue
        key = (player_id, integer(row.get("season")))
        team_type = row.get("type")
        tier = row.get("number_tm")
        if team_type in {"All-NBA", "All-ABA", "All-BAA"}:
            all_nba[key] = max(all_nba.get(key, 0), {"1st": 0.85, "2nd": 0.65, "3rd": 0.45}.get(tier, 0))
        elif team_type == "All-Defense":
            all_defense[key] = max(all_defense.get(key, 0), {"1st": 0.75, "2nd": 0.50}.get(tier, 0))

    mvp_share: dict[tuple[str, int], float] = {}
    mvp_winner: set[tuple[str, int]] = set()
    dpoy_share: dict[tuple[str, int], float] = {}
    dpoy_winner: set[tuple[str, int]] = set()
    for row in award_rows:
        player_id = (row.get("player_id") or "").strip()
        if player_id not in players:
            continue
        key = (player_id, integer(row.get("season")))
        award = (row.get("award") or "").lower()
        share = number(row.get("share"))
        if award in {"nba mvp", "aba mvp"}:
            mvp_share[key] = max(mvp_share.get(key, 0), share)
            if (row.get("winner") or "").upper() == "TRUE":
                mvp_winner.add(key)
        elif award == "nba dpoy":
            dpoy_share[key] = max(dpoy_share.get(key, 0), share)
            if (row.get("winner") or "").upper() == "TRUE":
                dpoy_winner.add(key)

    all_stars = {
        ((row.get("player_id") or "").strip(), integer(row.get("season")))
        for row in star_rows
        if (row.get("player_id") or "").strip() in players and (row.get("lg") or "").upper() in LEAGUES
    }
    by_name = {norm(str(player["name"])): player_id for player_id, player in players.items()}

    def recent_keys(names: list[str]) -> list[tuple[str, int]]:
        return [(by_name[norm(name)], 2026) for name in names]

    for key in recent_keys(["Shai Gilgeous-Alexander"]):
        mvp_share[key] = 1.0
        mvp_winner.add(key)
    for key in recent_keys(["Victor Wembanyama"]):
        dpoy_share[key] = 1.0
        dpoy_winner.add(key)
    for names, value in (
        (["Shai Gilgeous-Alexander", "Nikola Jokić", "Victor Wembanyama", "Luka Dončić", "Cade Cunningham"], 0.85),
        (["Jaylen Brown", "Kawhi Leonard", "Donovan Mitchell", "Kevin Durant", "Jalen Brunson"], 0.65),
        (["Tyrese Maxey", "Jamal Murray", "Jalen Johnson", "Jalen Duren", "Chet Holmgren"], 0.45),
    ):
        for key in recent_keys(names):
            all_nba[key] = value
    for names, value in (
        (["Victor Wembanyama", "Chet Holmgren", "Ausar Thompson", "Rudy Gobert", "Derrick White"], 0.75),
        (["Scottie Barnes", "Cason Wallace", "Bam Adebayo", "OG Anunoby", "Dyson Daniels"], 0.50),
    ):
        for key in recent_keys(names):
            all_defense[key] = value

    records_by_player: dict[str, list[dict[str, object]]] = defaultdict(list)
    for record in records:
        records_by_player[str(record["player_id"])].append(record)

    # Actual postseason performance, normalized within each postseason so that
    # raw pace and scoring environments do not decide cross-era comparisons.
    playoff_by_season: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in playoff_rows:
        if integer(str(row.get("gp") or 0)) > 0:
            playoff_by_season[integer(str(row.get("season") or 0))].append(row)

    playoff_scored: list[dict[str, object]] = []
    playoff_metric_weights = {"pts": 0.38, "trb": 0.18, "ast": 0.24, "blk": 0.08, "fg_pct": 0.12}
    for season, cohort in playoff_by_season.items():
        max_games = max(number(str(row.get("gp") or 0)) for row in cohort)
        qualified = [
            row
            for row in cohort
            if number(str(row.get("gp") or 0)) >= max(3, max_games * 0.20)
            and (row.get("mpg") is None or number(str(row.get("mpg"))) >= 10)
        ]
        if len(qualified) < 10:
            qualified = cohort
        distributions: dict[str, list[float]] = {}
        for metric in playoff_metric_weights:
            adjusted = f"{metric}_adj"
            values = [
                number(str(row.get(adjusted) if row.get(adjusted) is not None else row.get(metric)))
                for row in qualified
                if row.get(adjusted) is not None or row.get(metric) is not None
            ]
            distributions[metric] = sorted(values)

        for row in cohort:
            component_values: list[tuple[float, float]] = []
            for metric, weight in playoff_metric_weights.items():
                adjusted = f"{metric}_adj"
                raw = row.get(adjusted) if row.get(adjusted) is not None else row.get(metric)
                percentile = percentile_against(distributions[metric], optional_number(str(raw)) if raw is not None else None)
                if percentile is not None:
                    component_values.append((percentile, weight))
            if not component_values:
                continue
            weight_total = sum(weight for _, weight in component_values)
            raw_rate = sum(value * weight for value, weight in component_values) / weight_total
            games = number(str(row.get("gp") or 0))
            minutes = optional_number(str(row.get("mpg"))) if row.get("mpg") is not None else None
            minutes_factor = min(1.0, minutes / 32) if minutes is not None else 1.0
            reliability = sqrt(min(1.0, games / max_games) * max(0.10, minutes_factor))
            shrunk_rate = 50 + (raw_rate - 50) * reliability
            run_share = min(1.0, games / max_games) * minutes_factor
            # Run depth affects reliability and career accumulation, but does
            # not directly add points to the per-run performance rate. This
            # avoids rewarding team advancement here and again in titles.
            season_score = shrunk_rate
            playoff_scored.append(
                {
                    "name": str(row.get("name") or ""),
                    "season": season,
                    "gp": games,
                    "score": max(0.0, min(100.0, season_score)),
                    "run_share": run_share,
                    "metric_coverage": weight_total,
                }
            )

    players_by_name: dict[str, list[str]] = defaultdict(list)
    for player_id, player in players.items():
        players_by_name[norm(str(player["name"]))].append(player_id)
    playoff_by_player: dict[str, list[dict[str, object]]] = defaultdict(list)
    unmatched_playoff_rows = 0
    for record in playoff_scored:
        candidates = [
            player_id
            for player_id in players_by_name.get(norm(str(record["name"])), [])
            if int(players[player_id]["first_seas"]) <= int(record["season"]) <= int(players[player_id]["last_seas"])
        ]
        if len(candidates) == 1:
            playoff_by_player[candidates[0]].append(record)
        else:
            unmatched_playoff_rows += 1

    for player_id, player in players.items():
        best_by_year: dict[int, dict[str, object]] = {}
        for record in records_by_player.get(player_id, []):
            season = int(record["season"])
            if season not in best_by_year or float(record["season_score"]) > float(best_by_year[season]["season_score"]):
                best_by_year[season] = record
        career_records = list(best_by_year.values())
        top_peak = sorted(career_records, key=lambda record: float(record["season_score"]), reverse=True)[:3]
        peak_percentile = fmean(float(record["season_score"]) for record in top_peak) if top_peak else 0
        peak_score = max(0.0, min(100.0, (peak_percentile - 50) * 2))
        peak_years = sorted(int(record["season"]) for record in top_peak)
        peak_coverage = fmean(float(record["advanced_coverage"]) for record in top_peak) if top_peak else 0

        full_season_equiv = sum(float(record["availability"]) for record in career_records)
        career_impact_credits = sum(
            max(0.0, (float(record["season_score"]) - 50) / 50) * float(record["availability"])
            for record in career_records
        )
        career_advanced_coverage = (
            sum(float(record["advanced_coverage"]) * float(record["availability"]) for record in career_records)
            / full_season_equiv
            if full_season_equiv
            else 0
        )
        production_credits = sum(
            max(0.0, (float(record["production_pct"]) - 50) / 50) * float(record["availability"])
            for record in career_records
        )

        accolade_years = {season for pid, season in set(all_nba) | mvp_winner | set(mvp_share) | all_stars if pid == player_id}
        elite_credits = 0.0
        for season in accolade_years:
            key = (player_id, season)
            vote_credit = 1.0 if key in mvp_winner else sqrt(max(0.0, mvp_share.get(key, 0)))
            elite_credits += max(vote_credit, all_nba.get(key, 0), 0.25 if key in all_stars else 0)

        award_defense_credits = 0.0
        hybrid_defense_credits = 0.0
        award_eligible_equiv = 0.0
        for record in career_records:
            season = int(record["season"])
            league = str(record["league"])
            key = (player_id, season)
            award_eligible = (league == "NBA" and season >= 1969) or (league == "ABA" and season >= 1973)
            if award_eligible:
                award_eligible_equiv += float(record["availability"])
            vote_credit = 1.0 if key in dpoy_winner else 0.85 * sqrt(max(0.0, dpoy_share.get(key, 0)))
            award_credit = max(vote_credit, all_defense.get(key, 0))
            defense_pct = record["defense_metric_pct"]
            metric_strength = max(0.0, min(1.0, (float(defense_pct) - 70) / 30)) if defense_pct is not None else 0.0
            metric_credit = metric_strength * (0.35 if award_eligible else 0.70)
            award_defense_credits += award_credit
            hybrid_defense_credits += max(award_credit, metric_credit)

        sustained_score = min(100.0, elite_credits / 10 * 100)
        role_strength = 0.60 * peak_score / 100 + 0.40 * sustained_score / 100
        non_fmvp_titles = max(0, int(player["champs"]) - int(player["fmvp"]))
        playoff_credits = int(player["fmvp"]) + non_fmvp_titles * (0.15 + 0.50 * role_strength)
        defense_award_coverage = award_eligible_equiv / full_season_equiv if full_season_equiv else 0
        defense_credits = award_defense_credits if defense_award_coverage >= 0.5 else hybrid_defense_credits

        player_playoffs = playoff_by_player.get(player_id, [])
        top_playoffs = sorted(player_playoffs, key=lambda record: float(record["score"]), reverse=True)[:3]
        playoff_peak_percentile = fmean(float(record["score"]) for record in top_playoffs) if top_playoffs else 50
        playoff_peak = max(0.0, min(100.0, (playoff_peak_percentile - 50) * 2)) if top_playoffs else 0
        playoff_performance_credits = sum(
            max(0.0, (float(record["score"]) - 50) / 50) * float(record["run_share"])
            for record in player_playoffs
        )
        playoff_accumulation = min(100.0, playoff_performance_credits / 7.5 * 100)
        playoff_score_v3 = 0.65 * playoff_peak + 0.35 * playoff_accumulation
        playoff_games = sum(float(record["gp"]) for record in player_playoffs)

        non_fmvp_titles = max(0, int(player["champs"]) - int(player["fmvp"]))
        championship_credits_v3 = int(player["fmvp"]) + non_fmvp_titles * (0.10 + 0.55 * playoff_score_v3 / 100)
        championship_score_v3 = min(100.0, championship_credits_v3 / 4 * 100)

        observable_playoff_years = max(0, int(player["last_seas"]) - max(1950, int(player["first_seas"])) + 1)
        playoff_year_coverage = min(1.0, observable_playoff_years / max(1, int(player["seasons"])))
        data_coverage = min(1.0, 0.55 * career_advanced_coverage + 0.20 * peak_coverage + 0.25 * playoff_year_coverage)
        confidence_grade = "A" if data_coverage >= 0.85 else "B" if data_coverage >= 0.65 else "C" if data_coverage >= 0.40 else "D"
        model_uncertainty = 0.75 + 5 * (1 - data_coverage)

        player.update(
            {
                "peak_v2": round(min(100.0, peak_score), 2),
                "peak_v3": round(min(100.0, peak_score), 2),
                "peak_years": peak_years,
                "peak_coverage": round(peak_coverage, 3),
                "playoff_v2": round(min(100.0, playoff_credits / 4 * 100), 2),
                "playoff_credits": round(playoff_credits, 3),
                "sustained_v2": round(sustained_score, 2),
                "elite_credits": round(elite_credits, 3),
                "defense_v2": round(min(100.0, defense_credits / 6 * 100), 2),
                "defense_credits": round(defense_credits, 3),
                "defense_award_coverage": round(defense_award_coverage, 3),
                "longevity_v2": round(min(100.0, full_season_equiv / 18 * 100), 2),
                "full_season_equiv": round(full_season_equiv, 2),
                "production_v2": round(min(100.0, production_credits / 12 * 100), 2),
                "production_credits": round(production_credits, 3),
                "career_v3": round(min(100.0, career_impact_credits / 14 * 100), 2),
                "career_impact_credits": round(career_impact_credits, 3),
                "career_advanced_coverage": round(career_advanced_coverage, 3),
                "playoff_v3": round(playoff_score_v3, 2),
                "playoff_peak_years": sorted(int(record["season"]) for record in top_playoffs),
                "playoff_games": int(round(playoff_games)),
                "playoff_performance_credits": round(playoff_performance_credits, 3),
                "recognition_v3": round(sustained_score, 2),
                "championship_v3": round(championship_score_v3, 2),
                "championship_credits_v3": round(championship_credits_v3, 3),
                "durability_v3": round(min(100.0, full_season_equiv / 18 * 100), 2),
                "data_coverage": round(data_coverage, 3),
                "confidence_grade": confidence_grade,
                "model_uncertainty": round(model_uncertainty, 2),
            }
        )

    if unmatched_playoff_rows > max(100, len(playoff_scored) * 0.05):
        raise RuntimeError(f"Too many unmatched playoff rows: {unmatched_playoff_rows}/{len(playoff_scored)}")


def add_weight_robustness(players: dict[str, dict[str, object]], sample_count: int = 800) -> None:
    """Estimate rank stability across a disclosed family of plausible weights.

    Sampling is deterministic. A bounded Dirichlet distribution varies the
    value judgments around the default without allowing implausible extremes.
    The resulting 10th–90th percentile is a weight-sensitivity interval, not a
    statistical confidence interval for player ability.
    """

    fields = ("peak_v3", "career_v3", "playoff_v3", "recognition_v3", "championship_v3", "durability_v3")
    centers = (27.0, 28.0, 25.0, 8.0, 5.0, 7.0)
    bounds = ((20, 33), (20, 35), (17, 32), (3, 16), (2, 13), (3, 15))
    rng = random.Random(20260827)
    samples: list[tuple[float, ...]] = [centers]
    attempts = 0
    concentration = 65
    while len(samples) < sample_count and attempts < sample_count * 500:
        attempts += 1
        raw = [rng.gammavariate(center / 100 * concentration, 1.0) for center in centers]
        total = sum(raw)
        weights = tuple(value / total * 100 for value in raw)
        if all(low <= value <= high for value, (low, high) in zip(weights, bounds)):
            samples.append(weights)
    if len(samples) != sample_count:
        raise RuntimeError(f"Could only produce {len(samples)} bounded weight samples")

    rank_samples: dict[str, list[int]] = {player_id: [] for player_id in players}
    for weights in samples:
        ranked = sorted(
            players.items(),
            key=lambda item: (
                -sum(float(item[1][field]) * weight for field, weight in zip(fields, weights)),
                str(item[1]["name"]),
            ),
        )
        for rank, (player_id, _) in enumerate(ranked, 1):
            rank_samples[player_id].append(rank)

    def quantile(values: list[int], proportion: float) -> int:
        ordered = sorted(values)
        return ordered[round((len(ordered) - 1) * proportion)]

    for player_id, player in players.items():
        ranks = rank_samples[player_id]
        player.update(
            {
                "weight_rank_median": quantile(ranks, 0.50),
                "weight_rank_low": quantile(ranks, 0.10),
                "weight_rank_high": quantile(ranks, 0.90),
                "top100_probability": round(sum(rank <= 100 for rank in ranks) / len(ranks), 3),
                "weight_sample_count": len(ranks),
            }
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("legacy_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--playoffs", type=Path, required=True)
    args = parser.parse_args()

    with zipfile.ZipFile(args.archive) as archive:
        career_rows = rows_from_zip(archive, "Player Career Info.csv")
        total_rows = rows_from_zip(archive, "Player Totals.csv")
        team_rows = rows_from_zip(archive, "End of Season Teams.csv")
        award_rows = rows_from_zip(archive, "Player Award Shares.csv")
        star_rows = rows_from_zip(archive, "All-Star Selections.csv")
        advanced_rows = rows_from_zip(archive, "Advanced.csv")
        team_summary_rows = rows_from_zip(archive, "Team Summaries.csv")

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
            "fmvp_eligible_seasons": 0,
            "defense_eligible_seasons": 0,
            "stocks_eligible_seasons": 0,
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
        lambda: {
            "g": 0.0,
            "pts": 0.0,
            "trb": 0.0,
            "ast": 0.0,
            "tracked_g": 0.0,
            "tracked_stl": 0.0,
            "tracked_blk": 0.0,
            "seasons": set(),
            "fmvp_eligible_seasons": set(),
            "defense_eligible_seasons": set(),
            "stocks_eligible_seasons": set(),
        }
    )
    for (player_id, season, league), rows in season_groups.items():
        total = select_total_row(rows)
        selected = [total] if total else rows
        bucket = accumulated[player_id]
        for row in selected:
            for field in ("g", "pts", "trb", "ast"):
                bucket[field] = float(bucket[field]) + number(row.get(field))
            stocks_are_tracked = all(row.get(field) not in {"", "NA", None} for field in ("stl", "blk"))
            if stocks_are_tracked:
                bucket["tracked_g"] = float(bucket["tracked_g"]) + number(row.get("g"))
                bucket["tracked_stl"] = float(bucket["tracked_stl"]) + number(row.get("stl"))
                bucket["tracked_blk"] = float(bucket["tracked_blk"]) + number(row.get("blk"))
        cast_seasons = bucket["seasons"]
        assert isinstance(cast_seasons, set)
        cast_seasons.add(season)
        if league == "NBA" and season >= 1969:
            cast_fmvp = bucket["fmvp_eligible_seasons"]
            assert isinstance(cast_fmvp, set)
            cast_fmvp.add(season)
        if (league == "NBA" and season >= 1969) or (league == "ABA" and season >= 1973):
            cast_defense = bucket["defense_eligible_seasons"]
            assert isinstance(cast_defense, set)
            cast_defense.add(season)
        if any(all(row.get(field) not in {"", "NA", None} for field in ("stl", "blk")) for row in selected):
            cast_stocks = bucket["stocks_eligible_seasons"]
            assert isinstance(cast_stocks, set)
            cast_stocks.add(season)

    for player_id, bucket in accumulated.items():
        player = players[player_id]
        games = float(bucket["g"])
        if games:
            player["ppg"] = round(float(bucket["pts"]) / games, 3)
            player["rpg"] = round(float(bucket["trb"]) / games, 3)
            player["apg"] = round(float(bucket["ast"]) / games, 3)
        tracked_games = float(bucket["tracked_g"])
        if tracked_games:
            player["spg"] = round(float(bucket["tracked_stl"]) / tracked_games, 3)
            player["bpg"] = round(float(bucket["tracked_blk"]) / tracked_games, 3)
        seasons = bucket["seasons"]
        assert isinstance(seasons, set)
        player["seasons"] = len(seasons)
        if seasons:
            player["first_seas"] = min(seasons)
            player["last_seas"] = max(seasons)
        for field in ("fmvp_eligible_seasons", "defense_eligible_seasons", "stocks_eligible_seasons"):
            eligible = bucket[field]
            assert isinstance(eligible, set)
            player[field] = len(eligible)

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
        2023: ["Bruce Brown", "Thomas Bryant", "Kentavious Caldwell-Pope", "Vlatko Čančar", "Aaron Gordon", "Christian Braun", "Jeff Green", "Reggie Jackson", "Nikola Jokić", "DeAndre Jordan", "Jamal Murray", "Zeke Nnaji", "Michael Porter Jr.", "Ish Smith", "Peyton Watson", "Jack White"],
        2024: ["Oshae Brissett", "Jaylen Brown", "JD Davison", "Sam Hauser", "Jrue Holiday", "Al Horford", "Luke Kornet", "Svi Mykhailiuk", "Drew Peterson", "Kristaps Porziņģis", "Payton Pritchard", "Neemias Queta", "Jaden Springer", "Jayson Tatum", "Xavier Tillman Sr.", "Jordan Walsh", "Derrick White"],
        2025: ["Shai Gilgeous-Alexander", "Jalen Williams", "Chet Holmgren", "Luguentz Dort", "Alex Caruso", "Isaiah Hartenstein", "Isaiah Joe", "Cason Wallace", "Aaron Wiggins", "Jaylin Williams", "Kenrich Williams", "Ousmane Dieng", "Branden Carlson", "Alex Ducas", "Adam Flagler", "Ajay Mitchell", "Dillon Jones", "Nikola Topić"],
        2026: ["Jose Alvarado", "OG Anunoby", "Mikal Bridges", "Jalen Brunson", "Jordan Clarkson", "Pacome Dadiet", "Mohamed Diawara", "Tosan Evbuomwan", "Josh Hart", "Miles McBride", "Mitchell Robinson", "Karl-Anthony Towns", "Tyler Kolek", "Landry Shamet", "Ariel Hukporti", "Guerschon Yabusele", "Jeremy Sochan", "Kevin McCullar Jr.", "Dillon Jones", "Trey Jemison"],
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

    playoff_rows = json.loads(args.playoffs.read_text(encoding="utf-8"))
    build_model_v3(players, total_rows, advanced_rows, team_rows, award_rows, star_rows, team_summary_rows, playoff_rows)
    add_weight_robustness(players)

    output = sorted(players.values(), key=lambda player: (str(player["name"]), str(player["id"])))
    args.output.write_text(json.dumps(output, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")

    lookup = {norm(str(player["name"])): player for player in output}
    barbosa = lookup["leandrobarbosa"]
    lebron = lookup["lebronjames"]
    assert (barbosa["mvp"], barbosa["fmvp"], barbosa["champs"], barbosa["nba1"]) == (0, 0, 1, 0)
    assert 10.5 < float(barbosa["ppg"]) < 10.7
    assert (lebron["mvp"], lebron["fmvp"], lebron["champs"], lebron["nba1"]) == (4, 4, 4, 13)
    expected_recent_champions = {
        "christianbraun": 1,
        "jadenspringer": 1,
        "ajaymitchell": 1,
        "dillonjones": 2,
        "nikolatopic": 1,
        "treyjemison": 1,
    }
    for name_key, expected in expected_recent_champions.items():
        assert int(lookup[name_key]["champs"]) == expected
    assert (lookup["mouhamadougueye"]["first_seas"], lookup["mouhamadougueye"]["last_seas"]) == (2024, 2026)
    assert (lookup["tristennewton"]["first_seas"], lookup["tristennewton"]["last_seas"]) == (2025, 2026)
    for player in output:
        assert int(player["first_seas"]) <= int(player["last_seas"])
        assert int(player["fmvp"]) <= int(player["champs"])
        assert int(player["champs"]) <= int(player["seasons"])
        for field in (
            "peak_v2", "playoff_v2", "sustained_v2", "defense_v2", "longevity_v2", "production_v2",
            "career_v3", "playoff_v3", "recognition_v3", "championship_v3", "durability_v3",
        ):
            assert 0 <= float(player[field]) <= 100
        assert len(player["peak_years"]) <= 3
        assert 0 <= float(player["peak_coverage"]) <= 1
        assert 0 <= float(player["defense_award_coverage"]) <= 1
        assert 0 <= float(player["data_coverage"]) <= 1
        assert player["confidence_grade"] in {"A", "B", "C", "D"}
        assert 1 <= int(player["weight_rank_low"]) <= int(player["weight_rank_high"]) <= len(output)
        assert int(player["weight_rank_low"]) <= int(player["weight_rank_median"]) <= int(player["weight_rank_high"])
        assert 0 <= float(player["top100_probability"]) <= 1
        assert int(player["weight_sample_count"]) == 800
        assert float(player["full_season_equiv"]) <= int(player["seasons"]) + 0.01
    nash = lookup["stevenash"]
    kawhi = lookup["kawhileonard"]
    russell = lookup["billrussell"]
    assert abs(float(nash["peak_v2"]) - float(kawhi["peak_v2"])) < 5
    assert float(kawhi["playoff_v2"]) > float(nash["playoff_v2"])
    assert float(kawhi["playoff_v3"]) > float(nash["playoff_v3"])
    assert int(nash["playoff_games"]) > 100 and int(kawhi["playoff_games"]) > 100
    assert float(russell["defense_v2"]) > 0 and float(russell["defense_award_coverage"]) < 0.5
    print(json.dumps({"players": len(output), "barbosa": barbosa, "lebron": lebron}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
