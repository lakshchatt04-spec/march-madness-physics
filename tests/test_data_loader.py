"""Validation suite for the March Madness data pipeline.

Every test runs against Kaggle-shaped CSV fixtures written to ``tmp_path``, so
the suite is hermetic and needs no downloaded dataset.  Expected values are
derived two ways: from a reference implementation written inline (Elo), and
from the stdlib ``statistics`` module (variance), so the assertions do not
merely restate the pandas code under test.
"""

from __future__ import annotations

import csv
import dataclasses
import math
import statistics
from collections.abc import Sequence
from pathlib import Path

import pandas as pd
import pytest

from src import data_loader
from src.data_loader import (
    DataSchemaError,
    EloConfig,
    build_particle_teams,
    compute_efficiency_variance,
    compute_elo_ratings,
    compute_p3ar,
    load_games,
    load_particle_teams,
    load_teams,
)
from src.models import DEFAULT_VOLATILITY, ParticleTeam, VolatilityParams

# The Kaggle file carries ~50 columns; the pipeline only consumes a subset, but
# writing the full header keeps the fixtures faithful to the real input.
GAMES_HEADER: tuple[str, ...] = (
    "Season", "DayNum", "WTeamID", "WTeamScore", "LTeamID", "LTeamScore",
    "WLoc", "WConf", "WRank", "LConf", "LRank",
    "WFGM", "WFGA", "WFGA3", "WFGA3A", "WFT", "WFTA",
    "WOREB", "WOREDA", "WOREBT", "WDREB", "WDA", "WDAP", "WTO", "WBTO", "WPF", "WFTG", "WPFP",
    "LFGM", "LFGA", "LFGA3", "LFGA3A", "LFT", "LFTA",
    "LOREB", "LOREDA", "LOREBT", "LDREB", "LDA", "LDAP", "LTO", "LBTO", "LPF", "LFTG", "LPFP",
)
TEAMS_HEADER: tuple[str, ...] = ("TeamID", "TeamName", "TeamAbbrev",
                                 "FirstSeasonLast", "LastSeasonLast")

ALPHA, BETA, GAMMA, DELTA = 1101, 1102, 1103, 1104

BoxStats = dict[str, float]


def _stats(fga: float, fga3: float, fta: float, oreb: float, to: float) -> BoxStats:
    """Build a full box-score dict from the five fields the pipeline reads."""
    return {
        "FGM": round(fga * 0.47),
        "FGA": fga,
        "FGA3": fga3,
        "FGA3A": round(fga3 * 0.36),
        "FT": round(fta * 0.75),
        "FTA": fta,
        "OREB": oreb,
        "OREDA": round(oreb * 0.3),
        "OREBT": round(oreb * 0.15),
        "DREB": 30,
        "DA": 5,
        "DAP": 1,
        "TO": to,
        "BTO": 13,
        "PF": 15,
        "FTG": 2,
        "PFP": 12,
    }


def _game(
    season: int,
    day: int,
    site: str,
    winner: int,
    wscore: int,
    wstats: BoxStats,
    loser: int,
    lscore: int,
    lstats: BoxStats,
) -> dict[str, object]:
    """Assemble one CSV row in the Kaggle winner/loser layout."""
    row: dict[str, object] = {
        "Season": season,
        "DayNum": day,
        "WTeamID": winner,
        "WTeamScore": wscore,
        "LTeamID": loser,
        "LTeamScore": lscore,
        "WLoc": site,
        "WConf": "MAAC",
        "WRank": "",
        "LConf": "MAAC",
        "LRank": "",
    }
    for key, value in wstats.items():
        row[f"W{key}"] = value
    for key, value in lstats.items():
        row[f"L{key}"] = value
    return row


# A four-game season, chosen so every possession estimate lands on a whole number
# (FTA values are multiples of 25, which keeps 0.44 * FTA exact).
SEASON_2023: tuple[dict[str, object], ...] = (
    _game(2023, 0, "H", ALPHA, 80, _stats(fga=60, fga3=20, fta=25, oreb=8, to=12),
          BETA, 70, _stats(fga=58, fga3=18, fta=25, oreb=6, to=14)),
    _game(2023, 10, "N", BETA, 75, _stats(fga=62, fga3=25, fta=25, oreb=7, to=10),
          ALPHA, 65, _stats(fga=57, fga3=15, fta=25, oreb=5, to=15)),
    _game(2023, 20, "A", BETA, 72, _stats(fga=60, fga3=20, fta=25, oreb=9, to=13),
          ALPHA, 68, _stats(fga=63, fga3=22, fta=25, oreb=6, to=11)),
    _game(2023, 30, "H", GAMMA, 70, _stats(fga=65, fga3=30, fta=25, oreb=10, to=9),
          ALPHA, 60, _stats(fga=55, fga3=12, fta=25, oreb=4, to=16)),
)

# Two more games in 2024. DELTA appears exactly once, which is what exercises
# the zero-variance branch of the efficiency feature.
SEASON_2024: tuple[dict[str, object], ...] = (
    _game(2024, 0, "H", ALPHA, 77, _stats(fga=59, fga3=24, fta=25, oreb=9, to=10),
          GAMMA, 63, _stats(fga=61, fga3=26, fta=25, oreb=8, to=12)),
    _game(2024, 5, "H", DELTA, 66, _stats(fga=50, fga3=10, fta=25, oreb=5, to=13),
          BETA, 60, _stats(fga=62, fga3=22, fta=25, oreb=7, to=12)),
)

TEAM_NAMES: tuple[dict[str, object], ...] = (
    {"TeamID": ALPHA, "TeamName": "Alpha State", "TeamAbbrev": "Alpha",
     "FirstSeasonLast": "1987", "LastSeasonLast": "2024"},
    {"TeamID": BETA, "TeamName": "Beta University", "TeamAbbrev": "Beta",
     "FirstSeasonLast": "1990", "LastSeasonLast": "2024"},
    {"TeamID": GAMMA, "TeamName": "Gamma College", "TeamAbbrev": "Gamma",
     "FirstSeasonLast": "2001", "LastSeasonLast": "2024"},
    {"TeamID": DELTA, "TeamName": "Delta Institute", "TeamAbbrev": "Delta",
     "FirstSeasonLast": "2010", "LastSeasonLast": "2024"},
)

ALL_GAMES: tuple[dict[str, object], ...] = SEASON_2023 + SEASON_2024


def _write_csv(path: Path, header: Sequence[str], rows: Sequence[dict[str, object]]) -> Path:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(header), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path


@pytest.fixture
def games_csv(tmp_path: Path) -> Path:
    return _write_csv(tmp_path / data_loader.GAMES_FILENAME, GAMES_HEADER, ALL_GAMES)


@pytest.fixture
def teams_csv(tmp_path: Path) -> Path:
    return _write_csv(tmp_path / data_loader.TEAMS_FILENAME, TEAMS_HEADER, TEAM_NAMES)


@pytest.fixture
def games(games_csv: Path) -> pd.DataFrame:
    return load_games(games_csv)


@pytest.fixture
def games_2023(games_csv: Path) -> pd.DataFrame:
    return load_games(games_csv, seasons=[2023])


def _elo_spec() -> tuple[tuple[int, str, int, int], ...]:
    """Chronological (season, site, winner, loser) tuples for every fixture game."""
    ordered = sorted(ALL_GAMES, key=lambda r: (int(r["Season"]), int(r["DayNum"])))
    return tuple(
        (int(r["Season"]), str(r["WLoc"]), int(r["WTeamID"]), int(r["LTeamID"]))
        for r in ordered
    )


def _reference_elo(
    spec: Sequence[tuple[int, str, int, int]],
    *,
    initial: float = 1500.0,
    k: float = 20.0,
    scale: float = 400.0,
    home_advantage: float = 65.0,
    reset_each_season: bool = True,
) -> dict[int, float]:
    """Reference Elo loop, written independently of the loader implementation."""
    ratings: dict[int, float] = {}
    previous_season: int | None = None
    for season, site, winner, loser in spec:
        if reset_each_season and season != previous_season:
            ratings.clear()
        previous_season = season
        winner_rating = ratings.get(winner, initial)
        loser_rating = ratings.get(loser, initial)
        if site == "H":
            advantage = home_advantage
        elif site == "A":
            advantage = -home_advantage
        else:
            advantage = 0.0
        expected = 1.0 / (1.0 + 10.0 ** ((loser_rating - (winner_rating + advantage)) / scale))
        delta = k * (1.0 - expected)
        ratings[winner] = winner_rating + delta
        ratings[loser] = loser_rating - delta
    return ratings


# Ground-truth efficiencies derived from the fixtures via the possession model
# POSS = FGA + 0.44 * FTA + TO - OREB, which is exact for every fixture because
# each FTA value is a multiple of 25.
ALPHA_EFFICIENCIES = (80 / 75, 65 / 78, 68 / 79, 60 / 78, 77 / 71)
BETA_EFFICIENCIES = (70 / 77, 75 / 76, 72 / 75, 60 / 78)
GAMMA_EFFICIENCIES = (70 / 75, 63 / 76)
BETA_EFFICIENCIES_2023 = (70 / 77, 75 / 76, 72 / 75)

# Season totals of FGA3 / FGA, summed over every game each team played.
P3AR_EXPECTED = {
    ALPHA: 93 / 294,
    BETA: 85 / 242,
    GAMMA: 56 / 126,
    DELTA: 10 / 50,
}


class TestSchemaValidation:
    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(DataSchemaError, match="input file not found"):
            load_games(tmp_path / "nope.csv")

    def test_directory_is_resolved_to_default_filename(self, tmp_path: Path) -> None:
        _write_csv(tmp_path / data_loader.GAMES_FILENAME, GAMES_HEADER, ALL_GAMES)
        frame = load_games(tmp_path)
        assert len(frame) == len(ALL_GAMES)

    def test_missing_column_names_the_column(self, tmp_path: Path) -> None:
        header = tuple(c for c in GAMES_HEADER if c != "WFGA3")
        rows = [{k: v for k, v in row.items() if k in header} for row in ALL_GAMES]
        path = _write_csv(tmp_path / data_loader.GAMES_FILENAME, header, rows)
        with pytest.raises(DataSchemaError, match=r"WFGA3"):
            load_games(path)

    def test_non_numeric_stat_raises(self, tmp_path: Path) -> None:
        broken = [dict(row) for row in ALL_GAMES]
        broken[0]["WFGA3"] = "n/a"
        path = _write_csv(tmp_path / data_loader.GAMES_FILENAME, GAMES_HEADER, broken)
        with pytest.raises(DataSchemaError, match="WFGA3"):
            load_games(path)

    def test_missing_stat_raises(self, tmp_path: Path) -> None:
        broken = [dict(row) for row in ALL_GAMES]
        broken[0]["LOREB"] = ""
        path = _write_csv(tmp_path / data_loader.GAMES_FILENAME, GAMES_HEADER, broken)
        with pytest.raises(DataSchemaError, match="LOREB"):
            load_games(path)

    def test_unrecognised_location_raises(self, tmp_path: Path) -> None:
        broken = [dict(row) for row in ALL_GAMES]
        broken[0]["WLoc"] = "X"
        path = _write_csv(tmp_path / data_loader.GAMES_FILENAME, GAMES_HEADER, broken)
        with pytest.raises(DataSchemaError, match="WLoc"):
            load_games(path)

    def test_teams_file_requires_columns(self, tmp_path: Path) -> None:
        path = _write_csv(tmp_path / data_loader.TEAMS_FILENAME, ("TeamID",), TEAM_NAMES)
        with pytest.raises(DataSchemaError, match="TeamName"):
            load_teams(path)

    def test_duplicate_team_id_raises(self, tmp_path: Path) -> None:
        rows = list(TEAM_NAMES) + [dict(TEAM_NAMES[0], TeamName="Impostor")]
        path = _write_csv(tmp_path / data_loader.TEAMS_FILENAME, TEAMS_HEADER, rows)
        with pytest.raises(DataSchemaError, match="duplicate TeamID"):
            load_teams(path)

    def test_games_sorted_chronologically(self, games: pd.DataFrame) -> None:
        keys = list(zip(games["Season"], games["DayNum"], strict=True))
        assert keys == sorted(keys)

    def test_only_consumed_columns_are_loaded(self, games: pd.DataFrame) -> None:
        assert set(games.columns) == set(data_loader._REQUIRED_GAME_COLUMNS)

    def test_season_filter(self, games_csv: Path) -> None:
        frame = load_games(games_csv, seasons=[2023])
        assert set(frame["Season"]) == {2023}
        assert len(frame) == len(SEASON_2023)


class TestP3ar:
    def test_matches_hand_computed_ratios(self, games: pd.DataFrame) -> None:
        rates = compute_p3ar(games)
        for team_id, expected in P3AR_EXPECTED.items():
            assert rates[team_id] == pytest.approx(expected, rel=1e-12)

    def test_alpha_numerator_and_denominator(self, games: pd.DataFrame) -> None:
        assert compute_p3ar(games)[ALPHA] == pytest.approx(93 / 294, rel=1e-12)

    def test_within_unit_interval(self, games: pd.DataFrame) -> None:
        assert all(0.0 <= v <= 1.0 for v in compute_p3ar(games).values())

    def test_zero_field_goal_attempts_yields_zero(self, tmp_path: Path) -> None:
        rows = [_game(2023, 0, "H", ALPHA, 10, _stats(0, 0, 25, 0, 5),
                      BETA, 8, _stats(40, 12, 25, 5, 9))]
        path = _write_csv(tmp_path / data_loader.GAMES_FILENAME, GAMES_HEADER, rows)
        rates = compute_p3ar(load_games(path))
        assert rates[ALPHA] == 0.0

    def test_counts_both_wins_and_losses(self, games: pd.DataFrame) -> None:
        rates = compute_p3ar(games)
        assert set(rates) == set(P3AR_EXPECTED)
        assert rates[BETA] > 0.0

    def test_is_invariant_to_row_order(self, games: pd.DataFrame) -> None:
        shuffled = games.sample(frac=1.0, random_state=3).reset_index(drop=True)
        assert compute_p3ar(shuffled) == pytest.approx(compute_p3ar(games), rel=1e-12)


class TestEfficiencyVariance:
    def test_possession_model_matches_specification(self) -> None:
        expected = 60 + 0.44 * 25 + 12 - 8
        assert expected == pytest.approx(75.0, abs=1e-9)

    def test_matches_population_variance(self, games: pd.DataFrame) -> None:
        variances = compute_efficiency_variance(games)
        assert variances[ALPHA] == pytest.approx(
            statistics.pvariance(ALPHA_EFFICIENCIES), rel=1e-12
        )
        assert variances[BETA] == pytest.approx(
            statistics.pvariance(BETA_EFFICIENCIES), rel=1e-12
        )
        assert variances[GAMMA] == pytest.approx(
            statistics.pvariance(GAMMA_EFFICIENCIES), rel=1e-12
        )

    def test_is_not_sample_variance(self, games: pd.DataFrame) -> None:
        variances = compute_efficiency_variance(games)
        assert variances[ALPHA] != pytest.approx(
            statistics.variance(ALPHA_EFFICIENCIES), rel=1e-6
        )

    def test_single_game_team_has_zero_variance(self, games: pd.DataFrame) -> None:
        assert compute_efficiency_variance(games)[DELTA] == 0.0

    def test_aggregates_span_all_seasons(self, games: pd.DataFrame) -> None:
        assert compute_efficiency_variance(games)[BETA] != pytest.approx(
            statistics.pvariance(BETA_EFFICIENCIES_2023), rel=1e-9
        )

    def test_non_negative(self, games: pd.DataFrame) -> None:
        assert all(v >= 0.0 for v in compute_efficiency_variance(games).values())

    def test_pta_weight_is_honoured(self, games: pd.DataFrame) -> None:
        default = compute_efficiency_variance(games)
        alt = compute_efficiency_variance(games, fta_weight=0.0)
        assert alt[ALPHA] != pytest.approx(default[ALPHA], rel=1e-9)

    def test_invalid_weight_rejected(self, games: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="fta_weight"):
            compute_efficiency_variance(games, fta_weight=float("nan"))

    def test_zero_efficiency_variance_for_identical_games(self, tmp_path: Path) -> None:
        same = _stats(fga=60, fga3=20, fta=25, oreb=8, to=12)
        rows = [
            _game(2023, day, "H", ALPHA, 80, same, BETA, 70, same) for day in (0, 1, 2)
        ]
        path = _write_csv(tmp_path / data_loader.GAMES_FILENAME, GAMES_HEADER, rows)
        variances = compute_efficiency_variance(load_games(path))
        assert variances[ALPHA] == pytest.approx(0.0, abs=1e-12)
        assert variances[BETA] == pytest.approx(0.0, abs=1e-12)


class TestEloRatings:
    def test_matches_reference_implementation(self, games: pd.DataFrame) -> None:
        ratings = compute_elo_ratings(games, EloConfig(reset_each_season=True))
        expected = _reference_elo(_elo_spec())
        for team_id, value in expected.items():
            assert ratings[team_id] == pytest.approx(value, rel=1e-12)

    def test_first_game_uses_initial_rating_and_absolute_value(self) -> None:
        config = EloConfig(reset_each_season=True)
        expected_win = 1.0 / (1.0 + 10.0 ** ((1500.0 - 1565.0) / 400.0))
        rows = [_game(2023, 0, "H", ALPHA, 80, _stats(60, 20, 25, 8, 12),
                      BETA, 70, _stats(58, 18, 25, 6, 14))]
        ratings = compute_elo_ratings(pd.DataFrame(rows), config)
        gain = 20.0 * (1.0 - expected_win)
        assert ratings[ALPHA] == pytest.approx(1500.0 + gain, rel=1e-12)
        assert ratings[BETA] == pytest.approx(1500.0 - gain, rel=1e-12)
        assert ratings[ALPHA] == pytest.approx(1508.1506753883134, rel=1e-12)
        assert ratings[BETA] == pytest.approx(1491.8493246116866, rel=1e-12)

    def test_winner_gain_equals_loser_loss(self) -> None:
        """Equal-and-opposite movement is the defining property of Elo.

        Regression guard: computing the loser's update as ``-K * E_winner``
        instead of ``-K * (1 - E_winner)`` makes ratings diverge without bound.
        """
        for margin, site in ((1, "N"), (5, "H"), (40, "N"), (1, "A")):
            rows = [_game(2023, 0, site, ALPHA, 70 + margin, _stats(60, 20, 25, 8, 12),
                          BETA, 70, _stats(58, 18, 25, 6, 14))]
            ratings = compute_elo_ratings(pd.DataFrame(rows))
            winner_gain = ratings[ALPHA] - 1500.0
            loser_loss = 1500.0 - ratings[BETA]
            assert winner_gain == pytest.approx(loser_loss, rel=1e-12)

    def test_favourite_winning_is_worth_less_than_k_over_two(self) -> None:
        """A heavy favourite's win is unsurprising, so the Elo gain is small."""
        stats = _stats(fga=60, fga3=20, fta=25, oreb=8, to=12)
        seed = [_game(2023, day, "N", BETA, 75, stats, ALPHA, 70, stats) for day in range(40)]
        after_seed = compute_elo_ratings(pd.DataFrame(seed))
        assert after_seed[BETA] > after_seed[ALPHA] + 300.0

        next_game = [_game(2023, 40, "N", BETA, 75, stats, ALPHA, 70, stats)]
        combined = compute_elo_ratings(pd.DataFrame(seed + next_game))
        gain = combined[BETA] - after_seed[BETA]
        assert 0.0 < gain < EloConfig().k_factor / 2
        assert gain == pytest.approx(after_seed[ALPHA] - combined[ALPHA], rel=1e-12)

    def test_long_losing_streak_does_not_diverge(self, tmp_path: Path) -> None:
        """Ratings must stay bounded no matter how lopsided the schedule is."""
        stats = _stats(fga=60, fga3=20, fta=25, oreb=8, to=12)
        rows = [_game(2023, day, "N", GAMMA, 110, stats, ALPHA, 60, stats)
                for day in range(400)]
        path = _write_csv(tmp_path / data_loader.GAMES_FILENAME, GAMES_HEADER, rows)
        ratings = compute_elo_ratings(load_games(path))
        assert set(ratings) == {ALPHA, GAMMA}
        for value in ratings.values():
            assert -3000.0 < value < 3000.0

    def test_winner_gains_loser_loses(self, games_2023: pd.DataFrame) -> None:
        ratings = compute_elo_ratings(games_2023, EloConfig(reset_each_season=True))
        assert ratings[GAMMA] > 1500.0
        assert ratings[ALPHA] < 1500.0

    def _single_game_frame(self, site: str) -> pd.DataFrame:
        return pd.DataFrame([_game(2023, 0, site, ALPHA, 71, _stats(60, 20, 25, 8, 12),
                                   BETA, 70, _stats(58, 18, 25, 6, 14))])

    def test_home_advantage_makes_a_home_win_less_surprising(self) -> None:
        gains = {
            site: compute_elo_ratings(self._single_game_frame(site))[ALPHA] - 1500.0
            for site in ("H", "N", "A")
        }
        assert gains["N"] == pytest.approx(10.0, rel=1e-12)
        assert gains["H"] < gains["N"] < gains["A"]
        assert gains["H"] < 10.0 < gains["A"]

    def test_away_win_is_rewarded_more_than_neutral_win(self) -> None:
        away = compute_elo_ratings(self._single_game_frame("A"))[ALPHA]
        neutral = compute_elo_ratings(self._single_game_frame("N"))[ALPHA]
        assert away > neutral

    def test_home_win_is_rewarded_less_than_neutral_win(self) -> None:
        home = compute_elo_ratings(self._single_game_frame("H"))[ALPHA]
        neutral = compute_elo_ratings(self._single_game_frame("N"))[ALPHA]
        assert home < neutral

    def test_zero_home_advantage_removes_site_effect(self) -> None:
        config = EloConfig(reset_each_season=True, home_advantage=0.0)
        home = compute_elo_ratings(self._single_game_frame("H"), config)[ALPHA]
        neutral = compute_elo_ratings(self._single_game_frame("N"), config)[ALPHA]
        assert home == pytest.approx(neutral, rel=1e-12)
        assert home == pytest.approx(1500.0 + 10.0, rel=1e-12)

    def test_row_order_does_not_matter(self, games: pd.DataFrame) -> None:
        config = EloConfig(reset_each_season=True)
        baseline = compute_elo_ratings(games, config)
        shuffled = games.sample(frac=1.0, random_state=7).reset_index(drop=True)
        assert compute_elo_ratings(shuffled, config) == pytest.approx(baseline, rel=1e-12)

    def test_ratings_persist_across_seasons(self, games: pd.DataFrame) -> None:
        carried = compute_elo_ratings(games)
        only_2023 = compute_elo_ratings(games.loc[games["Season"] == 2023])
        for team_id in (ALPHA, BETA, GAMMA):
            assert carried[team_id] != pytest.approx(only_2023[team_id], rel=1e-9)

    def test_carry_over_differs_from_season_reset(self, games: pd.DataFrame) -> None:
        carried = compute_elo_ratings(games)
        reset = compute_elo_ratings(games, EloConfig(reset_each_season=True))
        assert carried != pytest.approx(reset, rel=1e-9)

    def test_seeded_team_inherits_prior_season_rating(self, games: pd.DataFrame) -> None:
        carried = compute_elo_ratings(games)
        only_2024 = compute_elo_ratings(games.loc[games["Season"] == 2024])
        assert carried[GAMMA] != pytest.approx(only_2024[GAMMA], rel=1e-9)

    def test_reset_each_season_matches_per_season_pass(self, games: pd.DataFrame) -> None:
        per_season: dict[int, float] = {}
        for season in (2023, 2024):
            subset = games.loc[games["Season"] == season]
            per_season.update(compute_elo_ratings(subset))
        combined = compute_elo_ratings(games, EloConfig(reset_each_season=True))
        for team_id, value in per_season.items():
            assert combined[team_id] == pytest.approx(value, rel=1e-12)

    def test_movement_is_bounded_by_k_factor(self, games: pd.DataFrame) -> None:
        config = EloConfig(reset_each_season=True)
        ratings = compute_elo_ratings(games, config)
        assert all(abs(v - config.initial_rating) <= config.k_factor for v in ratings.values())

    def test_mov_increases_blowout_impact(self) -> None:
        close = [_game(2023, 0, "H", ALPHA, 71, _stats(60, 20, 25, 8, 12),
                       BETA, 70, _stats(58, 18, 25, 6, 14))]
        blowout = [_game(2023, 0, "H", ALPHA, 110, _stats(60, 20, 25, 8, 12),
                         BETA, 40, _stats(58, 18, 25, 6, 14))]
        config = EloConfig(reset_each_season=True, mov_enabled=True)
        close_delta = compute_elo_ratings(pd.DataFrame(close), config)[ALPHA] - 1500.0
        blowout_delta = compute_elo_ratings(pd.DataFrame(blowout), config)[ALPHA] - 1500.0
        assert blowout_delta > close_delta

    def test_mov_disabled_by_default(self) -> None:
        assert EloConfig().mov_enabled is False

    def test_self_referential_game_rejected(self) -> None:
        row = _game(2023, 0, "H", ALPHA, 80, _stats(60, 20, 25, 8, 12),
                    ALPHA, 70, _stats(58, 18, 25, 6, 14))
        with pytest.raises(DataSchemaError, match="both winner and loser"):
            compute_elo_ratings(pd.DataFrame([row]))

    def test_empty_frame_yields_no_ratings(self) -> None:
        assert compute_elo_ratings(pd.DataFrame(columns=[
            "Season", "DayNum", "WTeamID", "WTeamScore", "LTeamID", "LTeamScore", "WLoc",
        ])) == {}

    @pytest.mark.parametrize("bad", [
        {"k_factor": 0.0},
        {"k_factor": -1.0},
        {"scale": 0.0},
        {"mov_reference": 0.0},
        {"mov_exponent": -1.0},
        {"mov_max_multiplier": 0.5},
        {"initial_rating": float("nan")},
        {"home_advantage": float("inf")},
    ])
    def test_invalid_config_rejected(self, bad: dict[str, float]) -> None:
        with pytest.raises(ValueError):
            EloConfig(**bad)

    def test_derive_elo_config_overrides_single_field(self) -> None:
        config = data_loader.derive_elo_config(k_factor=12.0)
        assert config.k_factor == 12.0
        assert config.initial_rating == EloConfig().initial_rating


class TestBuildParticleTeams:
    def test_keys_and_types(self, games: pd.DataFrame) -> None:
        teams = build_particle_teams(games, {ALPHA: "Alpha State", BETA: "Beta University",
                                             GAMMA: "Gamma College", DELTA: "Delta Institute"})
        assert list(teams) == sorted(teams)
        assert all(isinstance(t, ParticleTeam) for t in teams.values())

    def test_team_ids_match_keys(self, games: pd.DataFrame) -> None:
        for team_id, team in build_particle_teams(games).items():
            assert team.team_id == team_id

    def test_names_resolved_and_fallback_applied(self, games: pd.DataFrame) -> None:
        teams = build_particle_teams(games, {ALPHA: "Alpha State"})
        assert teams[ALPHA].team_name == "Alpha State"
        assert teams[BETA].team_name == "Team 1102"

    def test_features_match_individual_functions(self, games: pd.DataFrame) -> None:
        teams = build_particle_teams(games)
        elo = compute_elo_ratings(games)
        p3ar = compute_p3ar(games)
        variance = compute_efficiency_variance(games)
        for team_id, team in teams.items():
            assert team.elo_rating == pytest.approx(elo[team_id], rel=1e-12)
            assert team.p3ar == pytest.approx(p3ar[team_id], rel=1e-12)
            assert team.efficiency_variance == pytest.approx(variance[team_id], rel=1e-12)

    def test_all_fields_finite(self, games: pd.DataFrame) -> None:
        for team in build_particle_teams(games).values():
            for value in (team.elo_rating, team.p3ar, team.efficiency_variance):
                assert math.isfinite(value)

    def test_config_is_threaded_through(self, games: pd.DataFrame) -> None:
        default = build_particle_teams(games)
        custom = build_particle_teams(games, elo_config=EloConfig(k_factor=5.0))
        assert default[ALPHA].elo_rating != pytest.approx(custom[ALPHA].elo_rating, rel=1e-9)

    def test_empty_frame_yields_empty_container(self) -> None:
        assert build_particle_teams(pd.DataFrame(columns=[
            "Season", "DayNum", "WTeamID", "WTeamScore", "LTeamID", "LTeamScore", "WLoc",
            "WFGA", "WFGA3", "WFTA", "WOREB", "WTO",
            "LFGA", "LFGA3", "LFTA", "LOREB", "LTO",
        ])) == {}


class TestLoadParticleTeams:
    def test_end_to_end_from_directory(
        self, games_csv: Path, teams_csv: Path
    ) -> None:
        teams = load_particle_teams(games_csv, teams_csv)
        assert set(teams) == {ALPHA, BETA, GAMMA, DELTA}
        assert teams[GAMMA].team_name == "Gamma College"
        assert teams[DELTA].efficiency_variance == 0.0

    def test_teams_default_to_games_directory(
        self, games_csv: Path, teams_csv: Path
    ) -> None:
        teams = load_particle_teams(games_csv)
        assert teams[ALPHA].team_name == "Alpha State"

    def test_season_filter_applied_end_to_end(
        self, games_csv: Path, teams_csv: Path
    ) -> None:
        teams = load_particle_teams(games_csv, teams_csv, seasons=[2023])
        assert teams[BETA].efficiency_variance == pytest.approx(
            statistics.pvariance(BETA_EFFICIENCIES_2023), rel=1e-12
        )

    def test_container_is_reusable_for_simulation(
        self, games_csv: Path, teams_csv: Path
    ) -> None:
        teams = load_particle_teams(games_csv, teams_csv)
        volatilities = [t.internal_volatility for t in teams.values()]
        assert all(v > 0.0 for v in volatilities)
        assert all(math.isfinite(v) for v in volatilities)


class TestParticleTeam:
    def _team(self, **overrides: float) -> ParticleTeam:
        base = {
            "team_id": ALPHA,
            "team_name": "Alpha State",
            "elo_rating": 1500.0,
            "p3ar": 0.35,
            "efficiency_variance": 0.02,
        }
        return ParticleTeam(**{**base, **overrides})  # type: ignore[arg-type]

    def test_default_volatility_matches_specification(self) -> None:
        team = self._team()
        assert team.internal_volatility == pytest.approx(10.5 + 5.0 * 0.35 + 0.5 * 0.02, rel=1e-12)

    def test_default_coefficients(self) -> None:
        assert (DEFAULT_VOLATILITY.sigma_0, DEFAULT_VOLATILITY.alpha, DEFAULT_VOLATILITY.beta) \
            == (10.5, 5.0, 0.5)

    def test_volatility_accepts_custom_coefficients(self) -> None:
        team = self._team()
        params = VolatilityParams(sigma_0=1.0, alpha=2.0, beta=3.0)
        assert team.volatility(params) == pytest.approx(1.0 + 0.7 + 0.06, rel=1e-12)
        assert team.volatility() == pytest.approx(team.internal_volatility, rel=1e-12)

    def test_volatility_is_monotone_in_p3ar(self) -> None:
        low = self._team(p3ar=0.20).internal_volatility
        high = self._team(p3ar=0.50).internal_volatility
        assert high > low

    def test_volatility_rejects_bad_params_type(self) -> None:
        with pytest.raises(TypeError, match="VolatilityParams"):
            self._team().volatility({"sigma_0": 1.0})  # type: ignore[arg-type]

    def test_sigma_is_a_standard_deviation_in_elo(self) -> None:
        """sigma_0 = 10.5 means latent strength wanders +/-10.5 Elo, not 3.24."""
        team = ParticleTeam(
            team_id=ALPHA, team_name="Alpha", elo_rating=1500.0,
            p3ar=0.0, efficiency_variance=0.0,
        )
        assert team.internal_volatility == pytest.approx(10.5, rel=1e-12)
        assert team.internal_volatility_variance == pytest.approx(10.5**2, rel=1e-12)
        assert team.internal_volatility_variance == pytest.approx(110.25, rel=1e-12)

    def test_volatility_variance_is_square_of_volatility(self) -> None:
        for p3ar in (0.0, 0.28, 0.35, 0.50):
            for var in (0.0, 0.02, 0.06):
                team = self._team(p3ar=p3ar, efficiency_variance=var)
                assert team.internal_volatility_variance == pytest.approx(
                    team.internal_volatility**2, rel=1e-12
                )
                assert team.volatility_variance() == pytest.approx(
                    team.internal_volatility_variance, rel=1e-12
                )

    def test_volatility_variance_honours_custom_coefficients(self) -> None:
        team = self._team()
        params = VolatilityParams(sigma_0=4.0, alpha=0.0, beta=0.0)
        assert team.volatility(params) == pytest.approx(4.0, rel=1e-12)
        assert team.volatility_variance(params) == pytest.approx(16.0, rel=1e-12)

    def test_sigma_0_dominates_and_features_add_volatility(self) -> None:
        """Both features must contribute, and the baseline must stay dominant."""
        base = self._team(p3ar=0.0, efficiency_variance=0.0).internal_volatility
        from_p3ar = self._team(p3ar=0.5, efficiency_variance=0.0).internal_volatility - base
        from_var = self._team(p3ar=0.0, efficiency_variance=0.06).internal_volatility - base
        assert from_p3ar > 0.0
        assert from_var > 0.0
        assert base == pytest.approx(10.5, rel=1e-12)

    def test_frozen_immutability(self) -> None:
        with pytest.raises(dataclasses.FrozenInstanceError):
            self._team().elo_rating = 1600.0  # type: ignore[misc]

    def test_to_dict_round_trip(self) -> None:
        team = self._team()
        payload = team.to_dict()
        assert payload["team_id"] == ALPHA
        assert set(payload) == {
            "team_id", "team_name", "elo_rating", "p3ar",
            "efficiency_variance", "internal_volatility",
            "internal_volatility_variance",
        }
        assert payload["internal_volatility_variance"] == pytest.approx(
            float(payload["internal_volatility"]) ** 2, rel=1e-12
        )

    @pytest.mark.parametrize("overrides", [
        {"p3ar": 1.5},
        {"p3ar": -0.1},
        {"efficiency_variance": -1e-9},
        {"elo_rating": float("nan")},
        {"elo_rating": float("inf")},
        {"team_id": 0},
    ])
    def test_invalid_states_rejected(self, overrides: dict[str, float]) -> None:
        with pytest.raises(ValueError):
            self._team(**overrides)

    @pytest.mark.parametrize("overrides", [
        {"team_id": 1101.0},
        {"elo_rating": "1500"},
        {"p3ar": True},
    ])
    def test_wrong_types_rejected(self, overrides: dict[str, object]) -> None:
        with pytest.raises(TypeError):
            self._team(**overrides)  # type: ignore[arg-type]

    def test_blank_name_rejected(self) -> None:
        with pytest.raises(ValueError, match="team_name"):
            ParticleTeam(ALPHA, "   ", 1500.0, 0.35, 0.02)
