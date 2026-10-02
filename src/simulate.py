"""Monte Carlo tournament simulation over a validated bracket.

The bracket graph from :mod:`src.bracket` says who *can* meet.  This module
answers who *does*: it samples each matchup, advances winners through the
slots, and aggregates title probabilities over many runs.

Design notes
------------
Latent strength is drawn **once per team per simulation**, not per game.  A
team's hot (or cold) stretch spans its whole run, which is the point of a
particle model.  Resampling noise every game would make each team memoryless
and destroy the autocorrelation that makes upsets cluster.

Tournament games are neutral site, so ``venue_home_advantage`` is not applied.
A tied game is resolved by a coin flip at a seeded ``tie_prob`` rather than by
clipping the margin, so tuning tie handling never requires touching the
sampling code.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from random import Random
from typing import cast

import numpy as np

from src.bracket import TournamentBracket
from src.matchup import (
    MatchupParams,
    Venue,
    draw_latent_field,
    win_probability,
)
from src.models import ParticleTeam

__all__ = [
    "DEFAULT_TIE_PROB",
    "SimulationResult",
    "TournamentSimulator",
    "simulate_bracket",
    "simulate_seasons",
]

#: Probability a game ends tied.
#:
#: Zero by default, and zero is correct for the men's tournament: across all
#: 125,978 games in Kaggle's regular-season and tournament files there is not one
#: tie, because men's games go to overtime.  Ties are structurally impossible
#: rather than rare, so a non-zero default would inject noise that never occurs.
#: The knob remains for women's basketball, which does allow ties.
DEFAULT_TIE_PROB = 0.0


@dataclass(frozen=True)
class SimulationResult:
    """Aggregated output of many bracket simulations.

    Attributes:
        season: Championship year simulated.
        n_sims: Number of complete brackets played.
        champion_counts: Titles won per TeamID.
        round_counts: Wins at each round per TeamID (a win *in* round N means
            the team advanced past round N).
        appearance_counts: Teams present at the start of each round.
    """

    season: int
    n_sims: int
    champion_counts: Mapping[int, int]
    round_counts: Mapping[int, Mapping[int, int]]
    appearance_counts: Mapping[int, Mapping[int, int]]

    def title_probabilities(self) -> dict[int, float]:
        """Share of simulations won by each team."""
        if not self.n_sims:
            return {}
        return {t: c / self.n_sims for t, c in self.champion_counts.items()}

    def round_win_probabilities(self, round_no: int) -> dict[int, float]:
        """Chance a team wins a round, i.e. advances beyond it."""
        return self._rates(self.round_counts.get(round_no, {}))

    def appearance_probabilities(self, round_no: int) -> dict[int, float]:
        """Chance a team is present at the start of a round."""
        return self._rates(self.appearance_counts.get(round_no, {}))

    def _rates(self, counts: Mapping[int, int]) -> dict[int, float]:
        if not self.n_sims:
            return {}
        return {t: c / self.n_sims for t, c in counts.items()}

    def summary(self, names: Mapping[int, str], top: int = 10) -> str:
        """Human-readable leaderboard, used by tools/ and ad-hoc checks."""
        rows = sorted(self.title_probabilities().items(), key=lambda kv: -kv[1])[:top]
        lines = [f"{self.season}: {self.n_sims} sims"]
        for team, prob in rows:
            lines.append(f"  {prob * 100:5.2f}%  {names.get(team, str(team))}")
        return "\n".join(lines)


@dataclass
class TournamentSimulator:
    """Simulator bound to one bracket and one set of team particles.

    Args:
        bracket: Validated bracket structure.
        teams: Particle state keyed by TeamID.
        matchup: Calibrated matchup parameters.
        tie_prob: Probability a game ends tied.
        rng: Seeded RNG, so results are reproducible.
    """

    bracket: TournamentBracket
    teams: Mapping[int, ParticleTeam]
    matchup: MatchupParams
    tie_prob: float = DEFAULT_TIE_PROB
    rng: Random = field(default_factory=Random)
    draw_latent: Callable[[], Mapping[int, float]] | None = None

    def __post_init__(self) -> None:
        missing = sorted(s for s, t in self.bracket.seeds.items() if t not in self.teams)
        if missing:
            preview = ", ".join(missing[:8])
            more = "" if len(missing) <= 8 else f" (+{len(missing) - 8} more)"
            raise ValueError(
                f"{len(missing)} seeded team(s) missing from ParticleTeams: "
                f"{preview}{more}"
            )

    def play_once(self) -> tuple[int, dict[int, set[int]]]:
        """Play one complete bracket.

        Returns:
            ``(champion_id, appearances)`` where ``appearances[round]`` is the
            set of teams that took part in that round.  Round 0 is the play-in.
        """
        occupants: dict[str, int] = dict(self.bracket.seeds)
        appearances: dict[int, set[int]] = {}

        # One latent draw per team per simulation, so a hot team stays hot for
        # its whole run.  Resampling per game would make upsets independent.
        latent: Mapping[int, float] = self.draw_latent() if self.draw_latent else {}

        for round_no, slots in self.bracket.rounds().items():
            appearing = {
                self._resolve(occupants, side)
                for slot in slots
                for side in self.bracket.slots[slot]
            }
            appearances[round_no] = appearing

            winners: dict[str, int] = {}
            for slot in slots:
                strong, weak = self.bracket.slots[slot]
                a = self._resolve(occupants, strong)
                b = self._resolve(occupants, weak)
                winners[slot] = self._play(a, b, latent)
            occupants.update(winners)

        champion_slot = self.champion_slot()
        return self._resolve(occupants, champion_slot), appearances

    def champion_slot(self) -> str:
        """The single slot nothing feeds into - i.e. the final."""
        referenced = {
            side
            for pair in self.bracket.slots.values()
            for side in pair
            if side in self.bracket.slots
        }
        terminal = set(self.bracket.slots) - referenced
        if len(terminal) != 1:
            raise ValueError(
                f"expected exactly one terminal slot, found {sorted(terminal)}"
            )
        return terminal.pop()

    def _resolve(self, occupants: Mapping[str, int], key: str) -> int:
        """Turn a seed or slot reference into a TeamID."""
        if key in occupants:
            return occupants[key]
        return self.bracket.seeds[key]

    def _play(self, a: int, b: int, latent: Mapping[int, float] | None = None) -> int:
        """Pick a winner.  Teams are interchangeable by symmetry, so reorder."""
        p_a = win_probability(
            self.teams[a],
            self.teams[b],
            params=self.matchup,
            venue=Venue.NEUTRAL,
            latent=(latent.get(a, 0.0), latent.get(b, 0.0)) if latent else None,
        )
        p_b = 1.0 - p_a
        fav, dog = (a, b) if p_a >= p_b else (b, a)
        p_fav = max(p_a, p_b)
        # Two independent draws: reusing one would make the non-tie branch
        # conditional on having already failed the tie test, which quietly
        # skews the winner distribution whenever tie_prob is non-zero.
        if self.tie_prob and self.rng.random() < self.tie_prob:
            return fav if self.rng.random() < 0.5 else dog
        return fav if self.rng.random() < p_fav else dog


def simulate_bracket(
    bracket: TournamentBracket,
    teams: Mapping[int, ParticleTeam],
    matchup: MatchupParams,
    *,
    n_sims: int = 10_000,
    seed: int | None = 0,
    tie_prob: float = DEFAULT_TIE_PROB,
    use_latent: bool = False,
    latent_seed: int | None = None,
) -> SimulationResult:
    """Play ``n_sims`` brackets and aggregate probabilities.

    Args:
        bracket: Validated bracket for one season.
        teams: Particle state keyed by TeamID.
        matchup: Calibrated matchup parameters.
        n_sims: Number of complete brackets to play.
        seed: RNG seed; pass ``None`` for non-reproducible output.
        tie_prob: Probability a game ends tied.
        use_latent: Draw per-team latent strength offsets.  Requires
            NumPy, since the draws use `normal()` not
            `normalvariate()`.
        latent_seed: RNG seed for the latent draws, kept independent
            of `seed` so changing one does not shift the other.

    Returns:
        Aggregated probabilities across all simulations.
    """
    if n_sims < 1:
        raise ValueError(f"n_sims must be >= 1, got {n_sims}")

    latent_rng = None
    if use_latent:
        base = 0 if seed is None else seed if latent_seed is None else latent_seed
        latent_rng = np.random.default_rng(base + bracket.season)

    sim = TournamentSimulator(
        bracket=bracket,
        teams=teams,
        matchup=matchup,
        tie_prob=tie_prob,
        rng=Random(seed),
        draw_latent=(
            (lambda: draw_latent_field(teams, latent_rng, params=matchup))
            if latent_rng is not None
            else None
        ),
    )
    champion_counts: dict[int, int] = {}
    round_counts: dict[int, dict[int, int]] = {}
    appearance_counts: dict[int, dict[int, int]] = {}

    for _ in range(n_sims):
        champion, appearances = sim.play_once()
        champion_counts[champion] = champion_counts.get(champion, 0) + 1
        for round_no, teams_in_round in appearances.items():
            target = appearance_counts.setdefault(round_no, {})
            for team in teams_in_round:
                target[team] = target.get(team, 0) + 1

    # Wins in round N are exactly appearances in round N+1: every team present
    # in round N plays there, and the ones who win are the ones who show up
    # next.  Subtracting appearances(N) would be wrong - it made every count
    # negative, and no test asserted the sign.
    #
    # The final has no following round, so its single win is a title.
    # Round 0, when it exists, is the play-in, and the same rule holds: a
    # play-in winner is simply the participant who reaches round 1.
    last_round = max(appearance_counts, default=0)
    for round_no in sorted(appearance_counts):
        nxt = appearance_counts.get(round_no + 1, {})
        target = round_counts.setdefault(round_no, {})
        for team in appearance_counts[round_no]:
            won = (
                champion_counts.get(team, 0)
                if round_no == last_round
                else nxt.get(team, 0)
            )
            if won:
                target[team] = target.get(team, 0) + won

    return SimulationResult(
        season=bracket.season,
        n_sims=n_sims,
        champion_counts=champion_counts,
        round_counts=round_counts,
        appearance_counts=appearance_counts,
    )


def simulate_seasons(
    brackets: Mapping[int, TournamentBracket],
    teams: Mapping[int, object],
    matchup: Mapping[int, object] | MatchupParams,
    *,
    n_sims: int = 10_000,
    seed: int | None = 0,
    tie_prob: float = DEFAULT_TIE_PROB,
    use_latent: bool = False,
    latent_seed: int | None = None,
) -> dict[int, SimulationResult]:
    """Simulate several seasons, using an independent RNG stream per season.

    Args:
        brackets: Season to bracket.
        teams: Either `{season: {TeamID: ParticleTeam}}` for a true backtest,
            or one season-agnostic `{TeamID: ParticleTeam}}` map.  **Prefer
            the nested form.**  A single shared map silently feeds the
            end-of-season state of whichever season loaded last into every
            season, which leaks information backwards and makes a backtest
            meaningless.
        matchup: One `MatchupParams` shared by all seasons, or a
            `{season: MatchupParams}` mapping.
        n_sims: Brackets per season.
        seed: Base RNG seed; each season derives `seed + season`.
        tie_prob: Probability a game ends tied.
        use_latent: Draw per-team latent offsets.
        latent_seed: Base seed for the latent draws.  Defaults to `seed`, so
            the whole run stays reproducible from one number.

    Returns:
        `{season: SimulationResult}`.

    Raises:
        TypeError: If an entry has the wrong shape.
        ValueError: If a nested map is missing an entry for a requested season.
    """
    nested = _looks_nested(teams)

    out: dict[int, SimulationResult] = {}
    for season, bracket in sorted(brackets.items()):
        if nested:
            season_teams = teams.get(season)
            if season_teams is None:
                raise ValueError(f"season {season} missing from teams mapping")
        else:
            season_teams = teams
        season_matchup = (
            matchup.get(season) if isinstance(matchup, Mapping) else matchup
        )

        if not isinstance(season_teams, Mapping):
            raise TypeError(
                f"season {season} teams must be a mapping of TeamID to "
                f"ParticleTeam, got {type(season_teams).__name__}"
            )
        if not isinstance(season_matchup, MatchupParams):
            raise TypeError(
                f"season {season} matchup must be MatchupParams, got "
                f"{type(season_matchup).__name__}"
            )

        out[season] = simulate_bracket(
            bracket,
            cast("Mapping[int, ParticleTeam]", season_teams),
            season_matchup,
            n_sims=n_sims,
            seed=None if seed is None else seed + season,
            tie_prob=tie_prob,
            use_latent=use_latent,
            latent_seed=None if latent_seed is None else latent_seed + season,
        )
    return out


def _looks_nested(mapping: Mapping[int, object]) -> bool:
    """Heuristic: does this map go from season to per-season object?

    Distinguishes `{2025: {1100: ParticleTeam}}` from
    `{1100: ParticleTeam}` by inspecting the first value's type rather than
    the key range, because team ids and season numbers overlap.
    """
    for value in mapping.values():
        if isinstance(value, (ParticleTeam, MatchupParams)):
            return False
        if isinstance(value, Mapping):
            return True
    return False
