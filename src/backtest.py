"""Out-of-sample evaluation of the calibrated simulation.

Leakage is prevented by construction rather than by convention: strengths for
season ``S`` are fitted only on games from seasons strictly before ``S``,
while the team *features* (``p3ar``, ``efficiency_variance``) and the bracket
structure come from season ``S`` itself.  That mirrors the real forecasting
situation, where the current season's schedule is known but its results are
not.

Ground truth is reconstructed from ``MNCAATourneyDetailedResults.csv`` by
resolving the bracket's slot graph, so no season-specific ``DayNum`` cutoffs
are hardcoded.  That matters because the tourney is played on a different day
schedule in nearly every era.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import pandas as pd

from src.bracket import TournamentBracket
from src.simulate import SimulationResult

# Guard against a simulated probability of exactly zero, which would make a log
# loss infinite.  1e-12 is far below any probability a 10k-sim bracket produces,
# so it only clamps the pathological tail.
_MIN_PROB = 1e-12

#: Title log loss of a forecaster spreading its mass uniformly over a 68-team
#: field.  Round to 2dp, and reported next to the model's score so the headline
#: number always has a scale attached.
UNIFORM_TITLE_LOG_LOSS = math.log(68)

#: Mean squared error of a forecaster predicting the observed base rate.  Every
#: tournament game has one winner and one loser, so the per-team advancement
#: base rate is exactly 0.5 in every round and predicting 0.5 always scores
#: exactly this.  An advancement Brier at or above it is worth nothing, however
#: good it looks in isolation.
TRIVIAL_ADVANCEMENT_BRIER = 0.25

#: Mean ``-log p`` of a forecaster assigning 0.5 to every game's winner.
TRIVIAL_WIN_LOG_LOSS = -math.log(0.5)


class IncompleteTournamentError(ValueError):
    """Raised when a season's game log cannot cover every bracket matchup.

    Not hypothetical: Kaggle's ``MNCAATourneyDetailedResults.csv`` has 66 rows
    for 2021 instead of 67, omitting the first-round game for team 1433.  A
    single-elimination field of ``F`` teams always plays exactly ``F - 1`` games,
    so the row count alone detects the gap before any resolution is attempted.
    """


@dataclass(frozen=True)
class ActualOutcome:
    """Ground truth for a single season's tournament.

    Attributes:
        season: Championship year.
        champion: TeamID that won the title.
        round_teams: TeamIDs that played at least one game in each round,
            keyed by round number with 1 as the first round.  Presence in round
            ``r`` is precisely the event :class:`~src.simulate.SimulationResult`
            counts in ``appearance_counts``, so the two are comparable.
        bracket_teams: Every TeamID in the field, including play-in losers who
            never reached round 1.
    """

    season: int
    champion: int
    round_teams: Mapping[int, frozenset[int]]
    bracket_teams: frozenset[int]

    @property
    def n_rounds(self) -> int:
        """Number of rounds actually played."""
        return max(self.round_teams, default=0)

    def advancers(self, round_no: int) -> frozenset[int]:
        """Teams that won their round-``round_no`` game.

        Derived from ``round_teams`` rather than stored separately: a team
        present in round ``r + 1`` must have advanced out of round ``r``, and
        the last round is won by the champion alone.
        """
        if round_no < self.n_rounds:
            return self.round_teams.get(round_no + 1, frozenset())
        return frozenset({self.champion})


def _slot_round(slot: str) -> int | None:
    """Round number encoded in a slot name, or ``None`` for a play-in slot.

    Game slots are named ``R<round><region><index>`` (``R1W1``); the First Four
    slots carry the region name alone (``W16``), so they are excluded here.
    """
    if not slot.startswith("R") or len(slot) < 2 or not slot[1].isdigit():
        return None
    return int(slot[1])


def actual_outcome(
    tourney_games: pd.DataFrame, bracket: TournamentBracket
) -> ActualOutcome:
    """Recover the played-out bracket from the game log.

    Walks ``bracket.slots`` bottom-up: play-ins first, then rounds in ascending
    order, resolving each slot to the team that won it.  A slot is resolvable
    once both of its child slots are, which is guaranteed because a round-``r``
    slot's children are seeds (round 1), play-in winners, or round-``r - 1``
    winners.

    Args:
        tourney_games: Long frame of that season's tournament games.
        bracket: The bracket used for the simulation.

    Returns:
        The season's actual champion and per-round field.

    Raises:
        IncompleteTournamentError: If the log has the wrong number of rows for
            the field size, meaning it cannot be replayed at all.
        ValueError: If the log lacks a game for a bracket matchup, so the two
            sources disagree about the field itself.
    """
    expected = bracket.field_size - 1
    if len(tourney_games) != expected:
        raise IncompleteTournamentError(
            f"season {bracket.season} has {len(tourney_games)} tournament games "
            f"but a {bracket.field_size}-team single-elimination field needs "
            f"{expected}; cannot reconstruct ground truth"
        )

    winners: dict[tuple[int, int], int] = {}
    for win, lose in zip(
        tourney_games["WTeamID"], tourney_games["LTeamID"], strict=True
    ):
        w_id, l_id = int(win), int(lose)
        winners[(w_id, l_id)] = w_id
        winners[(l_id, w_id)] = w_id

    slot_team: dict[str, int] = {
        label: int(tid) for label, tid in bracket.seeds.items()
    }
    round_teams: dict[int, set[int]] = {}

    play_ins = [s for s in bracket.slots if _slot_round(s) is None]
    by_round: dict[int, list[str]] = {}
    for slot in bracket.slots:
        round_no = _slot_round(slot)
        if round_no is not None:
            by_round.setdefault(round_no, []).append(slot)

    ordered = play_ins + [s for r in sorted(by_round) for s in by_round[r]]
    for slot in ordered:
        child_a, child_b = bracket.slots[slot]
        team_a, team_b = slot_team.get(child_a), slot_team.get(child_b)
        if team_a is None or team_b is None:
            raise ValueError(
                f"slot {slot!r} could not be resolved before its children; "
                "bracket slot names are probably not ordered by round"
            )
        key = (team_a, team_b)
        if key not in winners:
            key = (team_b, team_a)
        if key not in winners:
            raise ValueError(
                f"no tournament game between teams {team_a} and {team_b}, "
                f"needed for slot {slot!r} in season {bracket.season}"
            )
        slot_team[slot] = winners[key]

        round_no = _slot_round(slot)
        if round_no is not None:
            round_teams.setdefault(round_no, set()).update((team_a, team_b))

    if not by_round:
        raise ValueError(f"bracket for season {bracket.season} has no game slots")

    final_round = max(by_round)
    if len(by_round[final_round]) != 1:
        raise ValueError(
            f"season {bracket.season} has {len(by_round[final_round])} games in "
            f"round {final_round}; expected exactly 1"
        )
    champion = slot_team[by_round[final_round][0]]

    return ActualOutcome(
        season=bracket.season,
        champion=champion,
        round_teams={r: frozenset(t) for r, t in round_teams.items()},
        bracket_teams=frozenset(bracket.seeds.values()),
    )


def title_log_loss(result: SimulationResult, champion: int) -> float:
    """Negative log likelihood the model assigned to the actual champion.

    Uniform over a 68-team field scores ``log(68) = 4.22``; a model that never
    gives the winner any mass diverges.  This is the single sharpest number a
    bracket forecaster can report, and it punishes both overconfidence and
    over-hedging.
    """
    p = result.title_probabilities().get(champion, 0.0)
    return -math.log(max(p, _MIN_PROB))


def advancement_brier(result: SimulationResult, outcome: ActualOutcome) -> float:
    """Mean squared error over every (round, team) advancement event.

    For each team present in round ``r``, the forecast is its probability of
    winning that round and the outcome is whether it actually advanced.  This is
    far more informative than a title-only score because it scores all 67 games
    instead of one, so a model that nails the champion but has no idea how the
    field narrows is still penalised appropriately.
    """
    total = 0.0
    count = 0
    for round_no, field in outcome.round_teams.items():
        probs = result.round_win_probabilities(round_no)
        advanced = outcome.advancers(round_no)
        for team in field:
            p = probs.get(team, 0.0)
            total += (p - (1.0 if team in advanced else 0.0)) ** 2
            count += 1
    return total / count if count else float("nan")


def reach_brier(result: SimulationResult, outcome: ActualOutcome) -> float:
    """Mean squared error over (round, team) *presence* events.

    Complements :func:`advancement_brier` by scoring whether a team was even in
    a given round, which the advancement score conditions away.  Reported
    alongside it because the two disagree when a model's field is mis-sized.
    """
    total = 0.0
    count = 0
    for round_no, field in outcome.round_teams.items():
        probs = result.appearance_probabilities(round_no)
        for team in outcome.bracket_teams:
            p = probs.get(team, 0.0)
            total += (p - (1.0 if team in field else 0.0)) ** 2
            count += 1
    return total / count if count else float("nan")


def win_log_loss_by_round(
    result: SimulationResult, outcome: ActualOutcome
) -> dict[int, tuple[float, int]]:
    """Per-round ``(mean, n_games)`` breakdown of :func:`win_log_loss`.

    Splitting by round is what makes the score interpretable.  The model can be
    genuinely informative early - it ranks a 15% team above a 5% team and wins
    - while being badly over-confident late, where it has too few games to tell
    and its probabilities are much more spread out.  A pooled mean hides that
    completely.
    """
    out: dict[int, tuple[float, int]] = {}
    for round_no in outcome.round_teams:
        probs = result.round_win_probabilities(round_no)
        total = 0.0
        count = 0
        for team in outcome.advancers(round_no):
            total -= math.log(max(probs.get(team, 0.0), _MIN_PROB))
            count += 1
        if count:
            out[round_no] = (total / count, count)
    return out


def win_log_loss(
    result: SimulationResult, outcome: ActualOutcome
) -> tuple[float, int]:
    """Mean ``-log p`` the model assigned to the team that actually won.

    One observation per *game* rather than per team, and keyed off
    :func:`ActualOutcome.advancers`, so it has a non-degenerate baseline: since
    every game has a winner, predicting 0.5 for each scores
    :data:`TRIVIAL_WIN_LOG_LOSS`.  That is what
    :func:`advancement_brier` cannot offer, because its base rate is pinned at
    0.5 by the bracket structure and it rewards a model for declining to
    discriminate.

    Play-in games are not scored, because :class:`ActualOutcome` records rounds
    from 1 upward and a First Four game leaves no trace once the field has been
    reconstructed.  :func:`reach_brier` is what covers reaching the round of 64
    in the first place.

    Returns:
        ``(mean, n_games)``, with ``n_games == 0`` if the outcome is empty.
    """
    per_round = win_log_loss_by_round(result, outcome)
    if not per_round:
        return float("nan"), 0
    total = sum(mean * n for mean, n in per_round.values())
    count = sum(n for _, n in per_round.values())
    return total / count, count


def favourite_forecasts(
    result: SimulationResult, outcome: ActualOutcome
) -> list[tuple[int, float, int]]:
    """One calibrated forecast per game: ``(round, P(favourite wins), 1 if it did)``.

    The reliability view that is *not* structurally degenerate.  Bucketing raw
    win probabilities is misleading here, because the two teams in one game
    always sum to 1: in a late round only 2n events exist and exactly n are
    wins, so the aggregate observed rate is forced to 0.5 no matter how good
    the model is, and a bucket that mixes lopsided underdogs with near-coin-flip
    losers drifts toward 0.5 for purely structural reasons.  Restricting to the
    favourite gives a forecast whose base rate is the model's own hit rate, so
    "the model said 80%" can actually be checked against "the favourite won 80%
    of the time".

    Ties at exactly ``p == 0.5`` score as an underdog loss, which is the
    convention :meth:`ActualOutcome.advancers` needs in order for
    :func:`win_log_loss` to see every game's winner exactly once.
    """
    rows: list[tuple[int, float, int]] = []
    for round_no in outcome.round_teams:
        probs = result.round_win_probabilities(round_no)
        for team in outcome.advancers(round_no):
            p = probs.get(team, 0.0)
            rows.append((round_no, max(p, 1.0 - p), 1 if p > 0.5 else 0))
    return rows


def reliability_forecasts(
    result: SimulationResult, outcome: ActualOutcome
) -> list[tuple[int, float, int]]:
    """Per-team advancement events as ``(round, P(advance), 1 if advanced)``."""
    rows: list[tuple[int, float, int]] = []
    for round_no, field in outcome.round_teams.items():
        probs = result.round_win_probabilities(round_no)
        advanced = outcome.advancers(round_no)
        for team in field:
            rows.append((round_no, probs.get(team, 0.0), 1 if team in advanced else 0))
    return rows


def round_trivial_baselines(outcome: ActualOutcome) -> dict[int, dict[str, float]]:
    """Best-possible constant prediction per round, for both Brier scores.

    Neither Brier has a single global floor.  Predicting each round's own
    observed base rate scores

    * ``p * (1 - p)`` for advancement, where ``p`` is the share of that round's
      field that advanced - and because a single-elimination round always splits
      its field evenly, that is exactly 0.25 in every round; and
    * ``p * (1 - p)`` again for reach, where ``p`` genuinely varies by round,
      since a 68-team field puts 64 teams in round 1 but only 2 in the final.

    Returns:
        ``{round_no: {"adv": ..., "reach": ...}}``.  The ``reach`` baseline
        scores the full ``bracket_teams`` field each round, matching
        :func:`reach_brier`, which also enumerates the whole field every round.
    """
    baselines: dict[int, dict[str, float]] = {}
    for round_no, field in outcome.round_teams.items():
        adv_p = len(outcome.advancers(round_no)) / len(field) if field else 0.0
        reach_p = len(field) / len(outcome.bracket_teams) if outcome.bracket_teams else 0.0
        baselines[round_no] = {
            "adv": adv_p * (1.0 - adv_p),
            "reach": reach_p * (1.0 - reach_p),
        }
    return baselines


# Buckets for the reliability diagram.  Wide at the extremes because a
# well-behaved model puts almost nothing out there, and narrow through the
# middle where most of the mass and most of the action is.
DEFAULT_EDGES = (0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0)


def calibration_table(
    forecasts: Sequence[tuple[float, int]],
    edges: Sequence[float] = DEFAULT_EDGES,
) -> list[tuple[float, float, float, int]]:
    """Reliability table: predicted probability versus observed frequency.

    Args:
        forecasts: ``(probability, outcome)`` pairs, one per forecastable
            event, where ``outcome`` is 1 for the event happening.
        edges: Ascending bucket edges on probability.

    Returns:
        One row per non-empty bucket as
        ``(midpoint, mean_predicted, observed_rate, n)``.  ``observed_rate`` is
        ``None`` when the bucket is too thin to interpret.

    This is the diagnostic that separates "the model ranks teams sensibly" from
    "the model's numbers mean what they say".  A Brier score near its floor is
    consistent both with genuine calibration and with a spread-out model that
    gets lucky on the ordering; only the reliability table distinguishes them.
    """
    rows: list[tuple[float, float, float, int]] = []
    if not forecasts:
        return rows
    for lo, hi in zip(edges, edges[1:], strict=False):
        bucket = [p for p, y in forecasts if lo <= p < hi]
        if not bucket:
            continue
        outcomes = [y for p, y in forecasts if lo <= p < hi]
        mean_p = sum(bucket) / len(bucket)
        observed = sum(outcomes) / len(outcomes)
        rows.append((0.5 * (lo + hi), mean_p, observed, len(bucket)))
    return rows


def brier(forecasts: Sequence[tuple[float, int]]) -> float:
    """Mean squared error over ``(probability, outcome)`` pairs."""
    if not forecasts:
        return float("nan")
    return sum((p - y) ** 2 for p, y in forecasts) / len(forecasts)
