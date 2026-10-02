"""Ingestion and feature engineering for the March Madness particle model.

Reads Kaggle's ``MRegularSeasonDetailedResults.csv`` and ``MTeams.csv`` and turns
them into the end-of-season particle state variables consumed by the tournament
simulation (see :class:`src.models.ParticleTeam`).

Feature definitions
-------------------
``elo_rating``
    Sequential Elo rating updated in strict chronological order
    ``(Season, DayNum)``.  Ratings carry across seasons; a team is seeded with
    ``initial_rating`` on its very first game.

``p3ar``
    Season aggregate ``sum(FGA3) / sum(FGA)``.

``efficiency_variance``
    Population variance of game-level offensive efficiency
    ``Points / Possessions``.  Possessions are not published in the Kaggle
    files, so they are estimated with the standard sports-analytics
    approximation ``POS = FGA + w * FTA + TO - OREB`` (``w = 0.44``).

All coefficients are exposed as parameters rather than hard-coded literals.
"""

from __future__ import annotations

import math
from collections.abc import Collection, Iterable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, cast

import pandas as pd

from src.models import ParticleTeam

__all__ = [
    "DataSchemaError",
    "EloConfig",
    "DEFAULT_FTA_WEIGHT",
    "GAMES_FILENAME",
    "TEAMS_FILENAME",
    "load_games",
    "load_teams",
    "compute_elo_ratings",
    "compute_p3ar",
    "compute_efficiency_variance",
    "build_particle_teams",
    "load_particle_teams",
    "derive_elo_config",
]

GAMES_FILENAME = "MRegularSeasonDetailedResults.csv"
TEAMS_FILENAME = "MTeams.csv"

#: Weight applied to free-throw attempts in the possession estimate.
DEFAULT_FTA_WEIGHT = 0.44

_REQUIRED_GAME_COLUMNS: tuple[str, ...] = (
    "Season",
    "DayNum",
    "WTeamID",
    "WTeamScore",
    "LTeamID",
    "LTeamScore",
    "WLoc",
    "WFGA",
    "WFGA3",
    "WFTA",
    "WOREB",
    "WTO",
    "LFGA",
    "LFGA3",
    "LFTA",
    "LOREB",
    "LTO",
)

_TEAM_ID_COLUMNS: tuple[str, ...] = ("TeamID", "TeamName")

#: The NCAA API and Kaggle disagree on four column names.  Both encodings are
#: accepted so the same loader serves either source and a cross-check between
#: them is possible.  The keys are the names the *source* may use; the values
#: are this pipeline's canonical spelling.  A source name is only rewritten when
#: the canonical column is absent, so a file carrying both keeps its originals.
_COLUMN_ALIASES: dict[str, str] = {
    "WScore": "WTeamScore",
    "LScore": "LTeamScore",
    "WOR": "WOREB",
    "LOR": "LOREB",
}

_ID_COLUMNS: tuple[str, ...] = ("Season", "DayNum", "WTeamID", "LTeamID")
_NUMERIC_COLUMNS: tuple[str, ...] = tuple(
    c for c in _REQUIRED_GAME_COLUMNS if c not in ("WLoc",)
)

_POSSESSION_COLUMNS: tuple[str, ...] = (
    "team_id",
    "points",
    "fga",
    "fta",
    "oreb",
    "turnovers",
)


class DataSchemaError(ValueError):
    """Raised when an input file is missing, malformed, or lacks required columns."""


@dataclass(frozen=True)
class EloConfig:
    """Coefficients for the sequential Elo update.

    Defaults describe a standard logistic Elo model for Division I men's
    basketball: 1500-point start, 400-point logistic scale, ``K = 20`` and a
    65-point home-court advantage.  Margin of victory is *not* weighted by
    default; set ``mov_enabled`` to enable it.
    """

    initial_rating: float = 1500.0
    k_factor: float = 20.0
    scale: float = 400.0
    home_advantage: float = 65.0
    reset_each_season: bool = False
    mov_enabled: bool = False
    mov_reference: float = 12.0
    mov_exponent: float = 0.5
    mov_max_multiplier: float = 2.5

    def __post_init__(self) -> None:
        for name in ("initial_rating", "k_factor", "scale", "home_advantage",
                     "mov_reference", "mov_exponent", "mov_max_multiplier"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a real number, got {type(value).__name__}")
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite, got {value!r}")
        if self.k_factor <= 0.0:
            raise ValueError(f"k_factor must be positive, got {self.k_factor}")
        if self.scale <= 0.0:
            raise ValueError(f"scale must be positive, got {self.scale}")
        if self.mov_reference <= 0.0:
            raise ValueError(f"mov_reference must be positive, got {self.mov_reference}")
        if self.mov_exponent < 0.0:
            raise ValueError(f"mov_exponent must be non-negative, got {self.mov_exponent}")
        if self.mov_max_multiplier < 1.0:
            raise ValueError(
                f"mov_max_multiplier must be at least 1.0, got {self.mov_max_multiplier}"
            )


def _resolve(path: str | Path, filename: str) -> Path:
    """Accept either a directory or an explicit CSV path.

    When ``path`` is a directory the well-known ``filename`` is appended.
    """
    resolved = Path(path)
    if resolved.is_dir():
        resolved = resolved / filename
    if not resolved.exists():
        raise DataSchemaError(f"input file not found: {resolved}")
    if not resolved.is_file():
        raise DataSchemaError(f"expected a file but found: {resolved}")
    return resolved


def _require_columns(actual: Iterable[str], required: Iterable[str], source: Path) -> None:
    present = set(actual)
    missing = [c for c in required if c not in present]
    if missing:
        raise DataSchemaError(
            f"{source.name} is missing required column(s): {', '.join(missing)}"
        )


def _normalise_columns(games: pd.DataFrame) -> pd.DataFrame:
    """Rewrite source-specific column names onto the pipeline's canonical names.

    Accepts either the NCAA API spelling or Kaggle's without the caller
    caring which it has.  A name is only rewritten when the canonical column
    is absent, so a file carrying both spellings keeps its originals.
    """
    renames = {
        alias: canonical
        for alias, canonical in _COLUMN_ALIASES.items()
        if alias in games.columns and canonical not in games.columns
    }
    return games.rename(columns=renames) if renames else games


def _on_disk_name(canonical: str, available: Iterable[str]) -> str:
    """The name ``canonical`` is stored under in the file being read."""
    present = set(available)
    if canonical in present:
        return canonical
    for alias, canon in _COLUMN_ALIASES.items():
        if canon == canonical and alias in present:
            return alias
    raise DataSchemaError(f"column {canonical} is absent from the input")


def load_games(path: str | Path, *, seasons: Collection[int] | None = None) -> pd.DataFrame:
    """Load ``MRegularSeasonDetailedResults.csv`` and validate its schema.

    Args:
        path: Path to the CSV, or to the directory containing it.
        seasons: Optional season filter.  Only these ``Season`` values are kept.

    Returns:
        A validated frame containing only the columns the pipeline consumes,
        ordered by ``(Season, DayNum)``.

    Raises:
        DataSchemaError: If the file is absent, lacks required columns, or
            contains non-numeric values in a required numeric column.
    """
    resolved = _resolve(path, GAMES_FILENAME)
    # Two views of the header: `raw` names what is actually on disk (needed to
    # build usecols), `header` is normalised to the canonical spelling (needed
    # to validate).  Passing the normalised one to _on_disk_name would resolve
    # "WOREB" on a Kaggle file that only has "WOR".
    raw = pd.read_csv(resolved, nrows=0)
    header = _normalise_columns(raw)
    _require_columns(header.columns, _REQUIRED_GAME_COLUMNS, resolved)

    read_cols = [_on_disk_name(c, raw.columns) for c in _REQUIRED_GAME_COLUMNS]
    games = pd.read_csv(resolved, usecols=read_cols)
    games = _normalise_columns(games).loc[:, list(_REQUIRED_GAME_COLUMNS)]
    for column in _NUMERIC_COLUMNS:
        games[column] = pd.to_numeric(games[column], errors="coerce")

    na_counts = {c: int(games[c].isna().sum()) for c in _NUMERIC_COLUMNS}
    offending = {c: n for c, n in na_counts.items() if n}
    if offending:
        detail = ", ".join(f"{c} ({n} row(s))" for c, n in sorted(offending.items()))
        rows = games.loc[
            games.loc[:, list(offending)].isna().any(axis=1),
            ["Season", "DayNum", "WTeamID", "LTeamID"],
        ].head(5)
        raise DataSchemaError(
            f"{resolved.name} has missing or non-numeric values in {detail}. "
            f"First offending rows:\n{rows.to_string(index=False)}"
        )

    games["WLoc"] = games["WLoc"].astype(str).str.strip().str.upper()
    unknown_sites = sorted(set(games["WLoc"].unique()) - {"H", "A", "N"})
    if unknown_sites:
        raise DataSchemaError(
            f"{resolved.name} has unrecognised WLoc value(s): {', '.join(unknown_sites)}"
        )

    for column in _ID_COLUMNS:
        games[column] = games[column].astype("int64")

    if seasons is not None:
        wanted = {int(s) for s in seasons}
        games = games.loc[games["Season"].isin(wanted)]

    return games.sort_values(["Season", "DayNum"], kind="stable").reset_index(drop=True)


def load_teams(path: str | Path) -> dict[int, str]:
    """Load ``MTeams.csv`` into a ``{TeamID: TeamName}`` mapping.

    Raises:
        DataSchemaError: If the file is absent or lacks ``TeamID``/``TeamName``.
    """
    resolved = _resolve(path, TEAMS_FILENAME)
    header = pd.read_csv(resolved, nrows=0)
    _require_columns(header.columns, _TEAM_ID_COLUMNS, resolved)

    teams = pd.read_csv(resolved, usecols=list(_TEAM_ID_COLUMNS))
    teams["TeamID"] = pd.to_numeric(teams["TeamID"], errors="coerce").astype("Int64")
    teams = teams.loc[teams["TeamID"].notna()]
    teams["TeamName"] = teams["TeamName"].astype(str)

    if teams["TeamID"].duplicated().any():
        duplicated = sorted(teams.loc[teams["TeamID"].duplicated(), "TeamID"].unique())
        raise DataSchemaError(
            f"{resolved.name} contains duplicate TeamID(s): {', '.join(map(str, duplicated))}"
        )

    return {int(tid): name for tid, name in zip(teams["TeamID"], teams["TeamName"], strict=True)}


def _expected_score(rating: float, opponent_rating: float, advantage: float, scale: float) -> float:
    """Logistic Elo expectation, with ``advantage`` folded into ``rating``."""
    exponent = (opponent_rating - (rating + advantage)) / scale
    return 1.0 / (1.0 + math.pow(10.0, exponent))


def _mov_multiplier(margin: float, config: EloConfig) -> float:
    if not config.mov_enabled or margin <= 0.0:
        return 1.0
    raw = math.pow(margin / config.mov_reference, config.mov_exponent)
    return min(max(raw, 1.0), config.mov_max_multiplier)


def compute_elo_ratings(
    games: pd.DataFrame, config: EloConfig | None = None
) -> dict[int, float]:
    """Compute final Elo ratings by walking games in chronological order.

    Games are processed in ``(Season, DayNum)`` order; games sharing a
    ``DayNum`` are genuinely simultaneous, so they are resolved in the order
    they appear in the input file (a stable sort).  Ratings persist across
    seasons unless ``config.reset_each_season`` is set.

    Args:
        games: Frame produced by :func:`load_games`.
        config: Elo coefficients.  Defaults to :class:`EloConfig` defaults.

    Returns:
        ``{TeamID: final_elo_rating}`` for every team appearing in ``games``.
    """
    cfg = config if config is not None else EloConfig()
    ordered = games.sort_values(["Season", "DayNum"], kind="stable")
    ratings: dict[int, float] = {}
    previous_season: int | None = None

    for row in ordered.itertuples(index=False):
        season = cast("int", row.Season)
        if cfg.reset_each_season and season != previous_season:
            ratings.clear()
        previous_season = season

        winner = cast("int", row.WTeamID)
        loser = cast("int", row.LTeamID)
        if winner == loser:
            raise DataSchemaError(
                f"game on Season {season} DayNum {cast('int', row.DayNum)} lists team "
                f"{winner} as both winner and loser"
            )

        site = str(row.WLoc)
        if site == "H":
            winner_advantage = cfg.home_advantage
        elif site == "A":
            winner_advantage = -cfg.home_advantage
        else:
            winner_advantage = 0.0

        winner_rating = ratings.get(winner, cfg.initial_rating)
        loser_rating = ratings.get(loser, cfg.initial_rating)

        expected = _expected_score(winner_rating, loser_rating, winner_advantage, cfg.scale)
        margin = cast("float", row.WTeamScore) - cast("float", row.LTeamScore)
        k_eff = cfg.k_factor * _mov_multiplier(margin, cfg)

        # Standard Elo: the winner's gain and the loser's loss are equal and
        # opposite, because E_loser = 1 - E_winner.  Both sides therefore move by
        # K * (1 - E_winner).
        delta = k_eff * (1.0 - expected)
        ratings[winner] = winner_rating + delta
        ratings[loser] = loser_rating - delta

    return ratings


def _team_games(games: pd.DataFrame) -> pd.DataFrame:
    """Explode one row per game into two rows, one per team."""
    winner = pd.DataFrame(
        {
            "team_id": games["WTeamID"],
            "points": games["WTeamScore"],
            "fga": games["WFGA"],
            "fga3": games["WFGA3"],
            "fta": games["WFTA"],
            "oreb": games["WOREB"],
            "turnovers": games["WTO"],
        }
    )
    loser = pd.DataFrame(
        {
            "team_id": games["LTeamID"],
            "points": games["LTeamScore"],
            "fga": games["LFGA"],
            "fga3": games["LFGA3"],
            "fta": games["LFTA"],
            "oreb": games["LOREB"],
            "turnovers": games["LTO"],
        }
    )
    return pd.concat([winner, loser], ignore_index=True)


def compute_p3ar(games: pd.DataFrame) -> dict[int, float]:
    """Season 3-Point Attempt Rate ``sum(FGA3) / sum(FGA)`` per team.

    Teams that attempted no field goals are assigned ``0.0`` rather than NaN.
    """
    team_games = _team_games(games)
    grouped = team_games.groupby("team_id", sort=True)[["fga", "fga3"]].sum()
    totals_fga = grouped["fga"].astype("float64")
    totals_fga3 = grouped["fga3"].astype("float64")
    rates = totals_fga3.div(totals_fga.where(totals_fga > 0.0)).fillna(0.0)
    return {cast("int", tid): float(value) for tid, value in rates.items()}


def compute_efficiency_variance(
    games: pd.DataFrame, *, fta_weight: float = DEFAULT_FTA_WEIGHT
) -> dict[int, float]:
    """Population variance of game-level offensive efficiency per team.

    Possessions are estimated as ``FGA + fta_weight * FTA + TO - OREB`` and
    efficiency is ``Points / Possessions``.  Games whose possession estimate is
    non-positive are excluded.  Teams with fewer than two usable games get
    ``0.0``.

    Args:
        games: Frame produced by :func:`load_games`.
        fta_weight: Coefficient on free-throw attempts in the possession model.
    """
    if not math.isfinite(fta_weight):
        raise ValueError(f"fta_weight must be finite, got {fta_weight!r}")

    team_games = _team_games(games).loc[:, list(_POSSESSION_COLUMNS)]
    possessions = (
        team_games["fga"]
        + fta_weight * team_games["fta"]
        + team_games["turnovers"]
        - team_games["oreb"]
    )
    usable = possessions > 0.0
    efficiency = team_games["points"].astype("float64").div(
        possessions.where(usable)
    )
    variance = (
        efficiency.groupby(team_games["team_id"], sort=True)
        .var(ddof=0)
        .fillna(0.0)
        .clip(lower=0.0)
    )
    return {cast("int", tid): float(value) for tid, value in variance.items()}


def build_particle_teams(
    games: pd.DataFrame,
    team_names: Mapping[int, str] | None = None,
    *,
    elo_config: EloConfig | None = None,
    fta_weight: float = DEFAULT_FTA_WEIGHT,
) -> dict[int, ParticleTeam]:
    """Assemble the ``{TeamID: ParticleTeam}`` export container.

    Only teams that played at least one game in ``games`` are included, since a
    team with no games has no particle state.  Teams missing from
    ``team_names`` fall back to ``"Team <id>"``.

    Args:
        games: Frame produced by :func:`load_games`.
        team_names: Optional ``{TeamID: TeamName}`` mapping.
        elo_config: Elo coefficients.
        fta_weight: Free-throw weight in the possession estimate.

    Returns:
        Mapping of ``TeamID`` to :class:`~src.models.ParticleTeam`, keyed and
        sorted by team id.
    """
    if games.empty:
        return {}

    names = dict(team_names) if team_names else {}
    elo = compute_elo_ratings(games, elo_config)
    p3ar = compute_p3ar(games)
    variance = compute_efficiency_variance(games, fta_weight=fta_weight)

    return {
        team_id: ParticleTeam(
            team_id=team_id,
            team_name=names.get(team_id, f"Team {team_id}"),
            elo_rating=elo[team_id],
            p3ar=p3ar.get(team_id, 0.0),
            efficiency_variance=variance.get(team_id, 0.0),
        )
        for team_id in sorted(elo)
    }


def load_particle_teams(
    games_path: str | Path,
    teams_path: str | Path | None = None,
    *,
    seasons: Collection[int] | None = None,
    elo_config: EloConfig | None = None,
    fta_weight: float = DEFAULT_FTA_WEIGHT,
) -> dict[int, ParticleTeam]:
    """End-to-end entry point: CSVs in, particle state container out.

    Args:
        games_path: Path to ``MRegularSeasonDetailedResults.csv`` or its directory.
        teams_path: Path to ``MTeams.csv`` or its directory.  Defaults to the
            same directory as ``games_path``.
        seasons: Optional season filter applied before any feature is computed.
        elo_config: Elo coefficients.
        fta_weight: Free-throw weight in the possession estimate.

    Returns:
        Mapping of ``TeamID`` to :class:`~src.models.ParticleTeam`.
    """
    resolved_games = _resolve(games_path, GAMES_FILENAME)
    if teams_path is None:
        resolved_teams: Path | None = resolved_games.parent / TEAMS_FILENAME
    else:
        resolved_teams = _resolve(teams_path, TEAMS_FILENAME)

    games = load_games(resolved_games, seasons=seasons)
    team_names = load_teams(resolved_teams) if resolved_teams is not None else {}
    return build_particle_teams(
        games, team_names, elo_config=elo_config, fta_weight=fta_weight
    )


def derive_elo_config(**overrides: Any) -> EloConfig:
    """Return :class:`EloConfig` with ``overrides`` applied.

    Convenience helper so callers can tweak a single coefficient without
    restating the full set of defaults.
    """
    return replace(EloConfig(), **overrides)
