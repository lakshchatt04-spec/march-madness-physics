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
    ActualOutcome,
    IncompleteTournamentError,
    actual_outcome,
    advancement_brier,
    reach_brier,
    title_log_loss,
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


def _played_games(bracket: TournamentBracket, winners: dict[str, int]) -> pd.DataFrame:
    team = dict(bracket.seeds)
    rows = []
    for day, slot in enumerate(_ordered_slots(bracket), start=130):
        a, b = bracket.slots[slot]
        ta, tb = team[a], team[b]
        win = winners[slot]
        lose = tb if win == ta else ta
        rows.append({"Season": bracket.season, "DayNum": day,
                     "WTeamID": win, "LTeamID": lose})
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
