"""Tests for out-of-sample backtest construction and scoring.

The load-bearing class is ``TestRoundWinProbabilitiesAreSigned``.
``simulate_bracket`` once returned *negative* round-win probabilities: the
aggregation subtracted a team's appearances from its wins instead of reading
the next round's appearances.  Title and appearance tests all still passed,
because the bug was confined to that single accessor.  These tests pin the
identity that makes it visible.
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from src.backtest import (
    TRIVIAL_ADVANCEMENT_BRIER,
    TRIVIAL_WIN_LOG_LOSS,
    ActualOutcome,
    IncompleteTournamentError,
    actual_outcome,
    advancement_brier,
    brier,
    favourite_forecasts,
    game_win_probabilities,
    reach_brier,
    reliability_forecasts,
    round_trivial_baselines,
    title_log_loss,
    win_log_loss,
    win_log_loss_by_round,
)
from src.bracket import REGIONS, TournamentBracket
from src.data_loader import ParticleTeam
from src.matchup import MatchupParams
from src.models import VolatilityParams
from src.simulate import SimulationResult, simulate_bracket

# Which (region, seed) positions are play-ins in the modern 68-team format.
PLAY_INS = (("W", 16), ("X", 11), ("Y", 11), ("Y", 16))


def _team(team_id: int, elo: float) -> ParticleTeam:
    return ParticleTeam(
        team_id=team_id,
        team_name=f"T{team_id}",
        elo_rating=elo,
        p3ar=0.38,
        efficiency_variance=0.02,
    )


def _params(**kw) -> MatchupParams:
    return MatchupParams(
        elo_per_point=kw.get("elo_per_point", 30.0),
        home_advantage_points=0.0,
        volatility=kw.get("volatility", VolatilityParams(1.0, 0.0, 0.0)),
    )


def _is_round_slot(slot: str) -> bool:
    return slot.startswith("R") and len(slot) > 1 and slot[1].isdigit()


def _round_of(slot: str) -> int:
    return int(slot[1])


def _ordered_slots(bracket: TournamentBracket) -> list[str]:
    """Play-ins first, then rounds ascending - the order ground truth needs."""
    play_ins = [s for s in bracket.slots if not _is_round_slot(s)]
    by_round: dict[int, list[str]] = {}
    for slot in bracket.slots:
        if _is_round_slot(slot):
            by_round.setdefault(_round_of(slot), []).append(slot)
    return play_ins + [s for r in sorted(by_round) for s in by_round[r]]


def _flat_bracket(season: int = 2025) -> TournamentBracket:
    """A 64-team bracket with no play-in."""
    seeds = {}
    for region in REGIONS:
        for n in range(1, 17):
            seeds[f"{region}{n:02d}"] = len(seeds) + 1
    slots = {}
    for region in REGIONS:
        for i in range(1, 9):
            slots[f"R1{region}{i}"] = (
                f"{region}{(i - 1) * 2 + 1:02d}",
                f"{region}{(i - 1) * 2 + 2:02d}",
            )
        for i in range(1, 5):
            slots[f"R2{region}{i}"] = (
                f"R1{region}{(i - 1) * 2 + 1}",
                f"R1{region}{(i - 1) * 2 + 2}",
            )
        slots[f"R3{region}1"] = (f"R2{region}1", f"R2{region}2")
        slots[f"R3{region}2"] = (f"R2{region}3", f"R2{region}4")
        slots[f"R4{region}1"] = (f"R3{region}1", f"R3{region}2")
    slots["R5WX"] = ("R4W1", "R4X1")
    slots["R5YZ"] = ("R4Y1", "R4Z1")
    slots["R6CH"] = ("R5WX", "R5YZ")
    return TournamentBracket(season, seeds, slots)


def _playin_bracket(season: int = 2025) -> TournamentBracket:
    """A 68-team bracket with the four First Four play-ins, as in real 2025."""
    seeds: dict[str, int] = {}
    for region in REGIONS:
        for n in range(1, 17):
            if (region, n) in PLAY_INS:
                seeds[f"{region}{n}A"] = len(seeds) + 1
                seeds[f"{region}{n}B"] = len(seeds) + 1
            else:
                seeds[f"{region}{n:02d}"] = len(seeds) + 1
    slots: dict[str, tuple[str, str]] = {}
    for region in REGIONS:
        for n in range(1, 17):
            if (region, n) in PLAY_INS:
                slots[f"{region}{n}"] = (f"{region}{n}A", f"{region}{n}B")
        for i in range(1, 9):
            slots[f"R1{region}{i}"] = (
                f"{region}{(i - 1) * 2 + 1:02d}",
                f"{region}{(i - 1) * 2 + 2:02d}",
            )
        for i in range(1, 5):
            slots[f"R2{region}{i}"] = (
                f"R1{region}{(i - 1) * 2 + 1}",
                f"R1{region}{(i - 1) * 2 + 2}",
            )
        slots[f"R3{region}1"] = (f"R2{region}1", f"R2{region}2")
        slots[f"R3{region}2"] = (f"R2{region}3", f"R2{region}4")
        slots[f"R4{region}1"] = (f"R3{region}1", f"R3{region}2")
    slots["R5WX"] = ("R4W1", "R4X1")
    slots["R5YZ"] = ("R4Y1", "R4Z1")
    slots["R6CH"] = ("R5WX", "R5YZ")
    return TournamentBracket(season, seeds, slots)


def _make_winners(bracket: TournamentBracket) -> dict[str, int]:
    """Resolve every slot to the higher-id contestant, bottom-up."""
    team = dict(bracket.seeds)
    out: dict[str, int] = {}
    for slot in _ordered_slots(bracket):
        a, b = bracket.slots[slot]
        win = max(team[a], team[b])
        out[slot] = win
        team[slot] = win
    return out


def _played_games(
    bracket: TournamentBracket,
    winners: dict[str, int],
    locations: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Game log for a resolved bracket; ``locations`` maps slot -> ``WLoc``.

    ``WLoc`` is written from the *winner's* perspective, matching the real
    file.  An omitted location means neutral.
    """
    locs = locations or {}
    team = dict(bracket.seeds)
    rows = []
    for day, slot in enumerate(_ordered_slots(bracket), start=130):
        a, b = bracket.slots[slot]
        ta, tb = team[a], team[b]
        win = winners[slot]
        lose = tb if win == ta else ta
        rows.append({"Season": bracket.season, "DayNum": day,
                     "WTeamID": win, "LTeamID": lose,
                     "WLoc": locs.get(slot, "N")})
        team[slot] = win
    return pd.DataFrame(rows)


@pytest.fixture
def bracket() -> TournamentBracket:
    return _flat_bracket()


@pytest.fixture
def field() -> dict[int, ParticleTeam]:
    return {tid: _team(tid, 1500.0 + 60.0 * ((tid % 8) - 3.5)) for tid in range(1, 65)}


@pytest.fixture
def playin() -> TournamentBracket:
    return _playin_bracket()


@pytest.fixture
def playin_field() -> dict[int, ParticleTeam]:
    return {tid: _team(tid, 1500.0 + 60.0 * ((tid % 8) - 3.5)) for tid in range(1, 69)}


class TestRoundWinProbabilitiesAreSigned:
    """Regression cover for the negative-count bug in ``simulate_bracket``."""

    def test_all_round_win_probabilities_are_fractions(
        self, bracket: TournamentBracket, field: dict
    ) -> None:
        result = simulate_bracket(bracket, field, _params(), n_sims=800, seed=3)
        for round_no in result.appearance_counts:
            for team, prob in result.round_win_probabilities(round_no).items():
                assert 0.0 <= prob <= 1.0, (round_no, team, prob)

    def test_wins_equal_next_round_appearances(
        self, bracket: TournamentBracket, field: dict
    ) -> None:
        result = simulate_bracket(bracket, field, _params(), n_sims=800, seed=4)
        last = max(result.appearance_counts)
        for round_no in sorted(result.appearance_counts):
            if round_no == last:
                continue
            nxt = result.appearance_counts[round_no + 1]
            for team in result.appearance_counts[round_no]:
                expected = nxt.get(team, 0) / result.n_sims
                assert result.round_win_probabilities(round_no).get(team, 0.0) == (
                    pytest.approx(expected)
                ), (round_no, team)

    def test_each_round_awards_one_win_per_game(
        self, bracket: TournamentBracket, field: dict
    ) -> None:
        result = simulate_bracket(bracket, field, _params(), n_sims=600, seed=5)
        for round_no, games in ((1, 32), (2, 16), (3, 8), (4, 4), (5, 2), (6, 1)):
            total = sum(result.round_win_probabilities(round_no).values())
            assert total == pytest.approx(games, abs=0.01), (round_no, total)

    def test_final_round_wins_are_titles(
        self, bracket: TournamentBracket, field: dict
    ) -> None:
        result = simulate_bracket(bracket, field, _params(), n_sims=800, seed=6)
        last = max(result.appearance_counts)
        for team in result.appearance_counts[last]:
            assert result.round_win_probabilities(last).get(team, 0.0) == (
                pytest.approx(result.champion_counts.get(team, 0) / result.n_sims)
            )

    def test_play_in_round_zero_is_scored_too(
        self, playin: TournamentBracket, playin_field: dict
    ) -> None:
        result = simulate_bracket(playin, playin_field, _params(), n_sims=800, seed=11)
        assert 0 in result.appearance_counts
        participants = set(result.appearance_counts[0])
        assert len(participants) == 8
        # Exactly one winner per play-in game, and pairs must be complementary.
        assert sum(result.round_win_probabilities(0).values()) == pytest.approx(4.0)
        for team, prob in result.round_win_probabilities(0).items():
            assert 0.0 < prob < 1.0, (team, prob)


class TestActualOutcome:
    def test_reconstructs_a_known_bracket(self, playin: TournamentBracket) -> None:
        winners = _make_winners(playin)
        outcome = actual_outcome(_played_games(playin, winners), playin)
        assert outcome.champion == winners["R6CH"]
        assert outcome.season == playin.season
        assert outcome.n_rounds == 6

    def test_round_fields_halve_each_round(self, playin: TournamentBracket) -> None:
        outcome = actual_outcome(
            _played_games(playin, _make_winners(playin)), playin
        )
        sizes = [len(outcome.round_teams[r]) for r in sorted(outcome.round_teams)]
        assert sizes == [64, 32, 16, 8, 4, 2]

    def test_game_count_matches_field_size(self, playin: TournamentBracket) -> None:
        games = _played_games(playin, _make_winners(playin))
        assert len(games) == playin.field_size - 1

    def test_play_in_losers_never_reach_round_one(
        self, playin: TournamentBracket
    ) -> None:
        winners = _make_winners(playin)
        games = _played_games(playin, winners)
        outcome = actual_outcome(games, playin)
        losers = set()
        team = dict(playin.seeds)
        for slot in _ordered_slots(playin):
            if _is_round_slot(slot):
                break
            a, b = playin.slots[slot]
            losers.add(team[a] if winners[slot] == team[b] else team[b])
            team[slot] = winners[slot]
        assert len(losers) == 4
        assert losers & outcome.round_teams[1] == set()
        assert losers <= outcome.bracket_teams

    def test_short_log_is_rejected(self, playin: TournamentBracket) -> None:
        games = _played_games(playin, _make_winners(playin))
        with pytest.raises(IncompleteTournamentError, match="67"):
            actual_outcome(games.iloc[:-1], playin)

    def test_wrong_pairing_does_not_resolve_silently(
        self, playin: TournamentBracket
    ) -> None:
        games = _played_games(playin, _make_winners(playin))
        bad = games.copy()
        bad.loc[0, "LTeamID"] = 9999
        with pytest.raises(ValueError, match="no tournament game"):
            actual_outcome(bad, playin)

    def test_advancers_is_the_next_round(self, playin: TournamentBracket) -> None:
        outcome = actual_outcome(
            _played_games(playin, _make_winners(playin)), playin
        )
        for round_no in range(1, outcome.n_rounds):
            assert outcome.advancers(round_no) == outcome.round_teams[round_no + 1]
        assert outcome.advancers(outcome.n_rounds) == frozenset({outcome.champion})

    def test_bracket_teams_covers_the_whole_field(
        self, playin: TournamentBracket
    ) -> None:
        outcome = actual_outcome(
            _played_games(playin, _make_winners(playin)), playin
        )
        assert len(outcome.bracket_teams) == playin.field_size
        assert outcome.champion in outcome.bracket_teams


class TestScoring:
    def test_title_log_loss_of_a_flat_field(
        self, playin: TournamentBracket, playin_field: dict
    ) -> None:
        flat = {tid: _team(tid, 1500.0) for tid in playin_field}
        result = simulate_bracket(
            playin, flat, _params(volatility=VolatilityParams(50.0, 0.0, 0.0)),
            n_sims=4000, seed=7,
        )
        # A flat field is close to uniform, so the loss approaches log(68).
        assert 3.9 < title_log_loss(result, 1) < 4.5

    def test_title_log_loss_rewards_the_modelled_champion(
        self, bracket: TournamentBracket, field: dict
    ) -> None:
        result = simulate_bracket(bracket, field, _params(), n_sims=2000, seed=8)
        probs = result.title_probabilities()
        champ = max(probs, key=probs.get)
        # The top pick's loss is -log(p), not zero: it is a probability.
        assert title_log_loss(result, champ) == pytest.approx(
            -math.log(probs[champ])
        )
        worst = min(probs, key=probs.get)
        assert title_log_loss(result, champ) < title_log_loss(result, worst)
        # An out-of-field id gets no mass at all and is clamped, not 0.
        assert title_log_loss(result, 99999) > 20.0

    def test_briers_are_bounded(
        self, playin: TournamentBracket, playin_field: dict
    ) -> None:
        result = simulate_bracket(playin, playin_field, _params(), n_sims=1500, seed=9)
        outcome = actual_outcome(
            _played_games(playin, _make_winners(playin)), playin
        )
        assert 0.0 <= advancement_brier(result, outcome) <= 1.0
        assert 0.0 <= reach_brier(result, outcome) <= 1.0

    def test_a_perfect_forecast_scores_zero(self, playin: TournamentBracket) -> None:
        outcome = actual_outcome(
            _played_games(playin, _make_winners(playin)), playin
        )
        result = SimulationResult(
            season=playin.season,
            n_sims=1,
            champion_counts={outcome.champion: 1},
            round_counts={
                r: dict.fromkeys(outcome.advancers(r), 1)
                for r in range(1, outcome.n_rounds + 1)
            },
            appearance_counts={
                r: dict.fromkeys(field, 1) for r, field in outcome.round_teams.items()
            },
        )
        assert advancement_brier(result, outcome) == pytest.approx(0.0)
        assert reach_brier(result, outcome) == pytest.approx(0.0)
        assert title_log_loss(result, outcome.champion) == pytest.approx(0.0)

    def test_a_constant_zero_forecast_is_maximally_penalised(
        self, playin: TournamentBracket
    ) -> None:
        """Guards against a metric that only ever returns 0."""
        outcome = actual_outcome(
            _played_games(playin, _make_winners(playin)), playin
        )
        blind = SimulationResult(
            season=playin.season,
            n_sims=1,
            champion_counts={outcome.champion: 1},
            round_counts={},
            appearance_counts={},
        )
        # With p=0 everywhere, a team that advanced scores 1 and one that went
        # out scores 0.  Exactly half of each round's field advances, so the
        # mean squared error is 0.5.
        assert advancement_brier(blind, outcome) == pytest.approx(0.5)
        # For presence, only the teams actually in each round score 1.
        present = sum(len(f) for f in outcome.round_teams.values())
        rounds = outcome.n_rounds
        assert reach_brier(blind, outcome) == pytest.approx(
            present / (rounds * len(outcome.bracket_teams))
        )


def test_outcome_is_frozen() -> None:
    b = _playin_bracket()
    outcome = actual_outcome(_played_games(b, _make_winners(b)), b)
    assert isinstance(outcome, ActualOutcome)
    with pytest.raises(AttributeError):
        outcome.season = 1999  # type: ignore[misc]


def _coin_flip_result(outcome: ActualOutcome) -> SimulationResult:
    """Every team's win probability is exactly 0.5 in every round."""
    return SimulationResult(
        season=outcome.season,
        n_sims=2,
        champion_counts={outcome.champion: 2},
        round_counts={
            r: dict.fromkeys(field, 1) for r, field in outcome.round_teams.items()
        },
        appearance_counts={
            r: dict.fromkeys(outcome.bracket_teams, 2) for r in outcome.round_teams
        },
    )


def _outcome(playin: TournamentBracket) -> ActualOutcome:
    return actual_outcome(_played_games(playin, _make_winners(playin)), playin)


def _confident_result(outcome: ActualOutcome, wins: int, n_sims: int) -> SimulationResult:
    """Every team wins its round with probability ``wins / n_sims``.

    Presence is certain, so the per-game conditional equals the unconditional
    value and these fixtures exercise the win-log-loss path without the
    dilution that :func:`conditional_win_probabilities` exists to undo.
    """
    return SimulationResult(
        season=outcome.season,
        n_sims=n_sims,
        champion_counts={outcome.champion: n_sims},
        round_counts={
            r: dict.fromkeys(field, wins) for r, field in outcome.round_teams.items()
        },
        appearance_counts={
            r: dict.fromkeys(field, n_sims) for r, field in outcome.round_teams.items()
        },
    )


def _team_average(result: SimulationResult, round_no: int, team: int) -> float:
    """The old per-team per-game forecast: unconditional advance / presence."""
    return (
        result.round_win_probabilities(round_no)[team]
        / result.appearance_probabilities(round_no)[team]
    )


class TestPerTeamAveragesAreNotPerGameForecasts:
    def test_two_teams_in_one_game_need_not_sum_to_one(
        self, playin: TournamentBracket
    ) -> None:
        """The defect: a marginal is not a distribution over outcomes.

        The two teams are averaged over *different* opponent draws, so their
        numbers are two separate forecasts that never complement each other.
        Across a real 20-season backtest this failed in 679 of 1260 games.
        """
        outcome = _outcome(playin)
        _, result = _simulated_field(playin, outcome, elo_step=25.0, n_sims=3000)
        for round_no in outcome.round_teams:
            pairs = [
                _team_average(result, round_no, team_a)
                + _team_average(result, round_no, team_b)
                for team_a, team_b in _games_of(outcome, round_no)
            ]
            # Round 1 has a single possible opponent, so it is coherent.  Every
            # later round mixes opponents and the two numbers drift apart.
            expected = 0 if round_no == 1 else len(pairs)
            assert sum(abs(pair - 1.0) > 1e-9 for pair in pairs) >= expected

    def test_the_per_team_average_is_not_the_actual_pairing(
        self, playin: TournamentBracket
    ) -> None:
        """A different quantity, not merely a noisy copy of the right one.

        Round 1 happens to be coherent because a team faces one opponent.  From
        round 2 on the average is spread over the whole field, so it drifts away
        from the probability of the game that actually happened.
        """
        outcome = _outcome(playin)
        params = _params(volatility=VolatilityParams(30.0, 0.0, 0.0))
        field, result = _simulated_field(playin, outcome, elo_step=25.0, n_sims=3000)
        analytic = game_win_probabilities(outcome, field, params)
        gaps = [
            abs(_team_average(result, game.round_no, game.winner) - p_win)
            for game, (_, p_win, _) in zip(outcome.games, analytic, strict=True)
        ]
        assert max(gaps) > 0.05
        assert max(gaps[32:]) > max(gaps[:32])

    def test_analytic_pair_is_exhaustive(
        self, playin: TournamentBracket, playin_field: dict
    ) -> None:
        """The replacement really is a two-outcome distribution."""
        outcome = _outcome(playin)
        for _, p_win, p_lose in game_win_probabilities(outcome, playin_field, _params()):
            assert p_win + p_lose == pytest.approx(1.0)

    def test_per_game_score_ignores_the_simulation_entirely(
        self, playin: TournamentBracket, playin_field: dict
    ) -> None:
        """A flat field and a sharp one must score differently.

        The score is a property of the matchup model, so team strengths alone
        drive it; no :class:`SimulationResult` is involved.
        """
        outcome = _outcome(playin)
        mean_flat, n_flat = win_log_loss(outcome, _flat_field(outcome), _params())
        mean_aligned, n_aligned = win_log_loss(
            outcome, _aligned_field(outcome), _params()
        )
        assert n_flat == n_aligned == _scored_games(outcome)
        assert mean_flat == pytest.approx(TRIVIAL_WIN_LOG_LOSS)
        assert mean_aligned < mean_flat

    def test_missing_team_state_is_rejected_not_skipped(
        self, playin: TournamentBracket, playin_field: dict
    ) -> None:
        """A silently shortened sample would flatter the score."""
        outcome = _outcome(playin)
        broken = dict(playin_field)
        broken.pop(outcome.games[0].team_a)
        with pytest.raises(ValueError, match="no team state"):
            game_win_probabilities(outcome, broken, _params())


def _scored_games(outcome: ActualOutcome) -> int:
    """Games this metric scores: 67 total minus the 4 unscored play-ins."""
    return sum(len(f) // 2 for f in outcome.round_teams.values())


def _games_of(outcome: ActualOutcome, round_no: int) -> list[tuple[int, int]]:
    """The actual pairings in one round, in bracket resolution order."""
    return [(g.team_a, g.team_b) for g in outcome.games if g.round_no == round_no]


def _step_field(outcome: ActualOutcome, step: float) -> dict[int, ParticleTeam]:
    """A hierarchy ordered by TeamID, spaced ``step`` Elo apart.

    ``_make_winners`` resolves every slot to the higher team id, so ordering
    Elo the same way makes the actual winners the favourites.  A larger step
    means a sharper field and a more confident forecast.
    """
    low = min(outcome.bracket_teams)
    return {tid: _team(tid, 1000.0 + step * (tid - low))
            for tid in outcome.bracket_teams}


def _aligned_field(outcome: ActualOutcome) -> dict[int, ParticleTeam]:
    return _step_field(outcome, 50.0)


def _flat_field(outcome: ActualOutcome) -> dict[int, ParticleTeam]:
    """Every team identical: the matchup model has nothing to go on."""
    return {tid: _team(tid, 1500.0) for tid in outcome.bracket_teams}


def _simulated_field(
    playin: TournamentBracket, outcome: ActualOutcome, elo_step: float, n_sims: int
):
    """A sharp field plus the bracket simulation run against it."""
    field = _step_field(outcome, elo_step)
    params = _params(volatility=VolatilityParams(30.0, 0.0, 0.0))
    result = simulate_bracket(playin, field, params, n_sims=n_sims, seed=11)
    return field, result


class TestPerGameWinLogLoss:
    """One observation per game, so the trivial baseline is well defined.

    Scored off the analytic matchup, so a flat field is the coin flip and a
    field aligned with the actual winners beats it.
    """

    def test_a_flat_field_scores_exactly_the_trivial_baseline(
        self, playin: TournamentBracket
    ) -> None:
        outcome = _outcome(playin)
        flat = _flat_field(outcome)
        mean, n_games = win_log_loss(outcome, flat, _params())
        assert mean == pytest.approx(TRIVIAL_WIN_LOG_LOSS)
        # One event per game.  A 68-team field plays 67 games, but the four
        # First Four games leave no trace in round_teams, so 63 are scored.
        assert n_games == _scored_games(outcome) == 63

    def test_a_coin_flip_advancement_score_is_exactly_quarter(
        self, playin: TournamentBracket
    ) -> None:
        """The advancement floor is 0.25, not 0."""
        outcome = _outcome(playin)
        assert advancement_brier(
            _coin_flip_result(outcome), outcome
        ) == pytest.approx(TRIVIAL_ADVANCEMENT_BRIER)

    def test_being_confident_about_the_winner_beats_a_coin_flip(
        self, playin: TournamentBracket
    ) -> None:
        outcome = _outcome(playin)
        mean, _ = win_log_loss(outcome, _aligned_field(outcome), _params())
        assert mean < TRIVIAL_WIN_LOG_LOSS
        # The metric has a gradient, not a single cliff at 0.5: a weaker
        # separation lands between the coin flip and the confident forecast.
        mean_weak, _ = win_log_loss(outcome, _step_field(outcome, 4.0), _params())
        assert TRIVIAL_WIN_LOG_LOSS > mean_weak > mean

    def test_the_score_is_always_finite_and_in_range(
        self, playin: TournamentBracket
    ) -> None:
        """An absurd separation must still produce a finite mean."""
        outcome = _outcome(playin)
        silly = {tid: _team(tid, 1500.0 + 5000.0 * tid)
                 for tid in outcome.bracket_teams}
        mean, n_games = win_log_loss(outcome, silly, _params())
        assert math.isfinite(mean)
        assert n_games == _scored_games(outcome)

    def test_empty_outcome_reports_nan_rather_than_dividing_by_zero(
        self, playin: TournamentBracket
    ) -> None:
        empty = ActualOutcome(
            season=2025,
            champion=1,
            round_teams={},
            bracket_teams=frozenset({1}),
        )
        flat = _flat_field(empty)
        mean, n_games = win_log_loss(empty, flat, _params())
        assert math.isnan(mean)
        assert n_games == 0

    def test_per_round_breakdown_aggregates_back_to_the_pooled_mean(
        self, playin: TournamentBracket
    ) -> None:
        outcome = _outcome(playin)
        aligned = _aligned_field(outcome)
        per_round = win_log_loss_by_round(outcome, aligned, _params())
        assert per_round
        for round_no, (_, count) in per_round.items():
            # Every scored round splits its field evenly, so half the games.
            assert count == len(outcome.round_teams[round_no]) // 2
        weighted = sum(m * n for m, n in per_round.values()) / sum(
            n for _, n in per_round.values()
        )
        pooled, pooled_n = win_log_loss(outcome, aligned, _params())
        assert pooled == pytest.approx(weighted)
        assert pooled_n == sum(n for _, n in per_round.values())


class TestFavouriteForecasts:
    """The calibration diagnostic that is not structurally pinned to 0.5."""

    def test_one_forecast_per_game_and_none_below_a_coin_flip(
        self, playin: TournamentBracket
    ) -> None:
        outcome = _outcome(playin)
        aligned = _aligned_field(outcome)
        rows = favourite_forecasts(outcome, aligned, _params())
        assert len(rows) == _scored_games(outcome)
        assert min(p for _, p, _ in rows) >= 0.5

    def test_observed_rate_is_the_models_hit_rate_not_a_fixed_half(
        self, playin: TournamentBracket
    ) -> None:
        """The point of the metric: the base rate is free to move."""
        outcome = _outcome(playin)
        aligned = _aligned_field(outcome)
        rows = favourite_forecasts(outcome, aligned, _params())
        # The winners are the higher Elo, so the favourite should win every one.
        assert sum(y for _, _, y in rows) / len(rows) == pytest.approx(1.0)

    def test_a_coin_flip_is_scored_as_an_underdog_loss(
        self, playin: TournamentBracket
    ) -> None:
        """p == 0.5 resolves to underdog, so winners are counted exactly once."""
        outcome = _outcome(playin)
        flat = _flat_field(outcome)
        rows = favourite_forecasts(outcome, flat, _params())
        assert all(p == pytest.approx(0.5) for _, p, _ in rows)
        assert sum(y for _, _, y in rows) == 0


class TestTrivialBaselines:
    def test_advancement_floor_is_quarter_in_every_round(
        self, playin: TournamentBracket
    ) -> None:
        outcome = _outcome(playin)
        baselines = round_trivial_baselines(outcome)
        assert set(baselines) == set(outcome.round_teams)
        for round_no, base in baselines.items():
            field = outcome.round_teams[round_no]
            share = len(outcome.advancers(round_no)) / len(field)
            assert base["adv"] == pytest.approx(share * (1.0 - share))
            assert base["adv"] == pytest.approx(TRIVIAL_ADVANCEMENT_BRIER)

    def test_reach_floor_shrinks_as_the_field_narrows(
        self, playin: TournamentBracket
    ) -> None:
        outcome = _outcome(playin)
        baselines = round_trivial_baselines(outcome)
        final = outcome.n_rounds
        share = len(outcome.round_teams[final]) / len(outcome.bracket_teams)
        assert baselines[final]["reach"] == pytest.approx(share * (1.0 - share))
        # The final is the deepest round, so it has the tightest floor.  A
        # single global reach floor would be wrong, which is why it is reported
        # per round.
        tightest = min(base["reach"] for base in baselines.values())
        assert tightest == pytest.approx(baselines[final]["reach"])


class TestReliabilityForecasts:
    def test_reproduce_the_advancement_brier_exactly(
        self, playin: TournamentBracket
    ) -> None:
        outcome = _outcome(playin)
        result = _confident_result(outcome, 3, 4)
        rows = reliability_forecasts(result, outcome)
        assert len(rows) == sum(len(f) for f in outcome.round_teams.values())
        assert brier([(p, y) for _, p, y in rows]) == pytest.approx(
            advancement_brier(result, outcome)
        )

    def test_include_the_teams_that_went_out_too(self, playin: TournamentBracket) -> None:
        """Every participant is scored, not only the winners."""
        outcome = _outcome(playin)
        rows = reliability_forecasts(_confident_result(outcome, 3, 4), outcome)
        assert sum(y for _, _, y in rows) == _scored_games(outcome)
        assert sum(1 for _, _, y in rows if y == 0) > 0

    def test_observed_rate_is_pinned_to_half_in_every_round(
        self, playin: TournamentBracket
    ) -> None:
        """Documents why the pooled advancement table cannot show skill.

        Each round splits its field evenly, so the aggregate outcome rate is
        0.5 whether the model is right or wrong.  This is a property of
        single elimination, not a defect in the forecast, and it is the reason
        the report leans on :func:`favourite_forecasts` for calibration.
        """
        outcome = _outcome(playin)
        rows = reliability_forecasts(_confident_result(outcome, 3, 4), outcome)
        for round_no in outcome.round_teams:
            band = [y for r, _, y in rows if r == round_no]
            assert sum(band) / len(band) == pytest.approx(0.5)

