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
from src.matchup import MatchupParams, Venue, win_probability
from src.models import ParticleTeam
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
class ActualGame:
    """One game that was actually played, with the matchup it was.

    Attributes:
        round_no: Round the game belonged to, with 1 as the first round.
        team_a: One of the two teams.
        team_b: The other team.
        winner: Which of the two won.
        venue: Where ``team_a`` played, from ``team_a``'s perspective.  Stored
            this way round because the source log records location from the
            *winner's* side, which is the wrong perspective for scoring the
            loser.
    """

    round_no: int
    team_a: int
    team_b: int
    winner: int
    venue: Venue


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
        games: Every round-``r`` game in the order the bracket resolved them.
            Play-ins are excluded, because once the field is reconstructed they
            leave no trace in ``round_teams``.
    """

    season: int
    champion: int
    round_teams: Mapping[int, frozenset[int]]
    bracket_teams: frozenset[int]
    games: tuple[ActualGame, ...] = ()

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


def _venue_from_perspective(loc: str, for_team_a: bool) -> Venue:
    """Convert a ``WLoc`` cell into a :class:`Venue` for ``team_a``.

    The source log records location from the winner's point of view, so when
    the winner is ``team_b`` the venue has to be flipped before it describes
    ``team_a``.  Anything unrecognised is treated as neutral, which is correct
    for tournament games and is what a frame without a ``WLoc`` column means.
    """
    venue = {"H": Venue.HOME, "A": Venue.AWAY}.get(loc.strip().upper(), Venue.NEUTRAL)
    if for_team_a or venue is Venue.NEUTRAL:
        return venue
    return Venue.AWAY if venue is Venue.HOME else Venue.HOME


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
        tourney_games: Long frame of that season's tournament games.  A
            ``WLoc`` column is used for the venue when present and assumed
            neutral when absent.
        bracket: The bracket used for the simulation.

    Returns:
        The season's actual champion, per-round field, and the games played.

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

    columns = tourney_games.columns
    winners: dict[tuple[int, int], tuple[int, str]] = {}
    for win, lose, loc in zip(
        tourney_games["WTeamID"],
        tourney_games["LTeamID"],
        tourney_games["WLoc"] if "WLoc" in columns else [""] * len(tourney_games),
        strict=True,
    ):
        w_id, l_id = int(win), int(lose)
        winners[(w_id, l_id)] = (w_id, str(loc))
        winners[(l_id, w_id)] = (w_id, str(loc))

    slot_team: dict[str, int] = {
        label: int(tid) for label, tid in bracket.seeds.items()
    }
    round_teams: dict[int, set[int]] = {}
    games: list[ActualGame] = []

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
        winner, loc = winners[key]
        slot_team[slot] = winner

        round_no = _slot_round(slot)
        if round_no is not None:
            round_teams.setdefault(round_no, set()).update((team_a, team_b))
            games.append(
                ActualGame(
                    round_no=round_no,
                    team_a=team_a,
                    team_b=team_b,
                    winner=winner,
                    venue=_venue_from_perspective(loc, winner == team_a),
                )
            )

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
        games=tuple(games),
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


def _opposite_venue(venue: Venue) -> Venue:
    if venue is Venue.HOME:
        return Venue.AWAY
    if venue is Venue.AWAY:
        return Venue.HOME
    return Venue.NEUTRAL


def game_win_probabilities(
    outcome: ActualOutcome,
    teams: Mapping[int, ParticleTeam],
    params: MatchupParams,
) -> list[tuple[int, float, float]]:
    """``(round_no, P(winner), P(loser))`` for every game actually played.

    The forecast scored here is the model's analytic probability for *that*
    matchup, ``Phi((Elo_a - Elo_b + hca) / tau_ab)``, at the venue the game was
    played.  Both sides come from the same matchup and the same parameters, so
    the pair sums to exactly 1.

    That is the whole reason this function exists rather than a per-team
    average.  :meth:`~src.simulate.SimulationResult.round_win_probabilities`
    reports each team's win rate against *whatever opponent the simulation gave
    it*, so the two teams in one game are averaged over different opponent draws
    and their numbers need not sum to 1 - measured over a 20-season backtest
    they failed to in 679 of 1260 games.  Scoring that marginal also measures
    something the model never claimed: it is a win rate *per opponent drawn*,
    not a probability of *this* game.  The analytic pair is what makes per-round
    numbers comparable to each other, which is the entire point of splitting
    them.

    Args:
        outcome: Ground truth, which carries the pairings and venues.
        teams: Team states for the evaluated season, carrying the fitted Elo and
            the volatility features that set ``tau``.
        params: Matchup parameters for that season's fit.

    Returns:
        One ``(round_no, p_winner, p_loser)`` per scored game, in bracket
        resolution order.  Play-ins are excluded, since
        :attr:`ActualOutcome.games` starts at round 1.

    Raises:
        ValueError: If a team that actually played has no entry in ``teams``,
            which would otherwise silently shrink the scored sample.
    """
    rows: list[tuple[int, float, float]] = []
    for game in outcome.games:
        team_a, team_b = teams.get(game.team_a), teams.get(game.team_b)
        if team_a is None or team_b is None:
            missing = game.team_a if team_a is None else game.team_b
            raise ValueError(
                f"team {missing} played in round {game.round_no} of season "
                f"{outcome.season} but has no team state; cannot score the matchup"
            )
        p_a = win_probability(team_a, team_b, params=params, venue=game.venue)
        p_b = win_probability(
            team_b, team_a, params=params, venue=_opposite_venue(game.venue)
        )
        if game.winner == game.team_a:
            rows.append((game.round_no, p_a, p_b))
        else:
            rows.append((game.round_no, p_b, p_a))
    return rows


def advancement_brier(result: SimulationResult, outcome: ActualOutcome) -> float:
    """Mean squared error over every (round, team) advancement event.

    For each team present in round ``r``, the forecast is its probability of
    winning that round and the outcome is whether it actually advanced.  This is
    far more informative than a title-only score because it scores all 67 games
    instead of one, so a model that nails the champion but has no idea how the
    field narrows is still penalised appropriately.

    The forecast is the *unconditional* advance probability, so this score
    partly measures how well the model predicts who is still alive - which is
    what :func:`reach_brier` isolates, and why the two are reported together.
    It is not a per-game score: it folds in the risk of never reaching the
    round, so it necessarily shrinks with depth.  To score the games themselves,
    use :func:`game_win_probabilities`.
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
    outcome: ActualOutcome,
    teams: Mapping[int, ParticleTeam],
    params: MatchupParams,
) -> dict[int, tuple[float, int]]:
    """Per-round ``(mean, n_games)`` breakdown of :func:`win_log_loss`.

    Splitting by round is what makes the score interpretable, because a model
    can be sharp early - it ranks a 15% team above a 5% team and wins - while
    drifting in the late rounds, where each round has fewer games and the
    bracket does more of the narrowing.  A pooled mean hides that completely.

    It is only comparable across rounds because every round is scored with the
    same analytic matchup forecast; see :func:`game_win_probabilities` for why
    a per-team average would not be.
    """
    out: dict[int, tuple[float, int]] = {}
    for round_no, p_win, _ in game_win_probabilities(outcome, teams, params):
        total, count = out.get(round_no, (0.0, 0))
        out[round_no] = (total - math.log(max(p_win, _MIN_PROB)), count + 1)
    return {r: (total / count, count) for r, (total, count) in out.items() if count}


def win_log_loss(
    outcome: ActualOutcome,
    teams: Mapping[int, ParticleTeam],
    params: MatchupParams,
) -> tuple[float, int]:
    """Mean ``-log p`` the model assigned to the team that actually won.

    One observation per *game* rather than per team, so it has a non-degenerate
    baseline: since every game has exactly one winner, predicting 0.5 for each
    scores :data:`TRIVIAL_WIN_LOG_LOSS`.  That is what :func:`advancement_brier`
    cannot offer, because its base rate is pinned at 0.5 by the bracket
    structure and it rewards a model for declining to discriminate.

    The probability scored is the analytic matchup forecast from
    :func:`game_win_probabilities` - the model's probability for the game that
    was actually played, not a per-team average over simulated opponents.
    Averaging over simulated opponents compresses the deep rounds toward 0.5 for
    a reason that has nothing to do with the forecast, which made the late
    rounds look like a collapse when the model was merely overconfident.

    Play-in games are not scored, because :attr:`ActualOutcome.games` starts at
    round 1 and a First Four game leaves no trace once the field has been
    reconstructed.  :func:`reach_brier` is what covers reaching the round of 64
    in the first place.

    Returns:
        ``(mean, n_games)``, with ``n_games == 0`` if the outcome is empty.
    """
    per_round = win_log_loss_by_round(outcome, teams, params)
    if not per_round:
        return float("nan"), 0
    total = sum(mean * n for mean, n in per_round.values())
    count = sum(n for _, n in per_round.values())
    return total / count, count


def favourite_forecasts(
    outcome: ActualOutcome,
    teams: Mapping[int, ParticleTeam],
    params: MatchupParams,
) -> list[tuple[int, float, int]]:
    """One calibrated forecast per game: ``(round, P(favourite wins), 1 if it did)``.

    The reliability view that is *not* structurally degenerate.  Pooling raw win
    probabilities is misleading, because in any round only ``2n`` events exist
    and exactly ``n`` are wins, so the aggregate observed rate is forced to 0.5
    no matter how good the model is, and a bucket that mixes lopsided underdogs
    with near-coin-flip losers drifts toward 0.5 for purely structural reasons.
    Restricting to the favourite gives a forecast whose base rate is the model's
    own hit rate, so "the model said 80%" can actually be checked against "the
    favourite won 80% of the time".

    Scoring the analytic matchup pair also makes this table honest in a way the
    old per-team average was not: because both sides come from one matchup they
    sum to 1 exactly, so ``max(p_winner, p_loser)`` really is one team's
    probability of winning the game that was played.

    Ties at exactly ``p == 0.5`` score as an underdog loss, so that
    :func:`win_log_loss` sees every game's winner exactly once.
    """
    rows: list[tuple[int, float, int]] = []
    for round_no, p_win, p_lose in game_win_probabilities(outcome, teams, params):
        rows.append((round_no, max(p_win, p_lose), 1 if p_win > 0.5 else 0))
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
