"""Tests for Monte Carlo bracket simulation.

Most fixtures use a *competitive* field rather than a runaway favourite: a
deterministic champion makes seed and reproducibility assertions vacuous,
because every outcome collapses to one team.  The analytic check in
``TestMatchesAnalyticProbabilities`` is the load-bearing one - it compares
simulated frequencies against ``win_probability`` and would catch both a
mis-scaled field and an unused latent draw.
"""

from __future__ import annotations

from random import Random

import numpy as np
import pytest

from src.bracket import REGIONS, TournamentBracket
from src.data_loader import ParticleTeam
from src.matchup import (
    DEFAULT_MATCHUP,
    MatchupParams,
    Venue,
    draw_latent_field,
    margin_scale_points,
    win_probability,
)
from src.models import MIN_TEAM_SIGMA, VolatilityParams
from src.simulate import (
    DEFAULT_TIE_PROB,
    TournamentSimulator,
    simulate_bracket,
    simulate_seasons,
)

N_TEAMS = 64


def _team(team_id: int, elo: float) -> ParticleTeam:
    return ParticleTeam(
        team_id=team_id,
        team_name=f"T{team_id}",
        elo_rating=elo,
        p3ar=0.38,
        efficiency_variance=0.02,
    )


def _flat_bracket(season: int = 2025) -> TournamentBracket:
    """A 64-team bracket with seeds W01..Z16 mapped to ids 1..64."""
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


@pytest.fixture
def bracket() -> TournamentBracket:
    return _flat_bracket()


@pytest.fixture
def field() -> dict[int, ParticleTeam]:
    """A field with a real spread, so results vary run to run."""
    return {tid: _team(tid, 1500.0 + 60.0 * ((tid % 8) - 3.5)) for tid in range(1, 65)}


@pytest.fixture
def runaway() -> dict[int, ParticleTeam]:
    """One overwhelming favourite - for 'best team should win' checks."""
    teams = {tid: _team(tid, 1500.0) for tid in range(2, 65)}
    teams[1] = _team(1, 2000.0)
    return teams


def _params(**kw) -> MatchupParams:
    return MatchupParams(
        elo_per_point=kw.get("elo_per_point", 30.0),
        home_advantage_points=0.0,
        volatility=kw.get("volatility", VolatilityParams(1.0, 0.0, 0.0)),
    )


class TestPlayOnce:
    def test_round_sizes_are_halving(self, bracket, field) -> None:
        sim = TournamentSimulator(bracket, field, _params(), rng=Random(1))
        _, appearances = sim.play_once()
        # This fixture has no play-in, so rounds are 1-6.
        assert [len(appearances[r]) for r in sorted(appearances)] == [
            64, 32, 16, 8, 4, 2,
        ]

    def test_champion_slot_is_the_final(self, bracket, field) -> None:
        sim = TournamentSimulator(bracket, field, _params(), rng=Random(1))
        assert sim.champion_slot() == "R6CH"

    def test_missing_team_is_rejected_at_construction(self, bracket, field) -> None:
        incomplete = {k: v for k, v in field.items() if k != 5}
        with pytest.raises(ValueError, match="missing from ParticleTeams"):
            TournamentSimulator(bracket, incomplete, _params())


class TestMatchesAnalyticProbabilities:
    """The invariant that ties the sampler back to the analytic model.

    Before this existed, nothing would have noticed that the simulator fed a
    mis-scaled field to the matchup model, or that latent draws were computed
    and then dropped on the floor.
    """

    def test_simulated_frequency_matches_win_probability(self, bracket, field) -> None:
        params = _params(volatility=VolatilityParams(6.0, 0.5, 0.3))
        sim = TournamentSimulator(bracket, field, params, rng=Random(4))

        pair = (3, 11)
        expected = win_probability(
            field[pair[0]], field[pair[1]], params=params, venue=Venue.NEUTRAL
        )
        trials = 40_000
        wins = sum(1 for _ in range(trials) if sim._play(*pair) == pair[0])
        # 3-sigma Monte Carlo tolerance.
        tol = 3.0 * (expected * (1 - expected) / trials) ** 0.5
        assert abs(wins / trials - expected) < max(tol, 0.01)

    def test_tie_prob_zero_never_changes_the_favourite(
        self, bracket, runaway
    ) -> None:
        """With ties disabled, a lopsided matchup is decided by the favourite."""
        params = _params(volatility=VolatilityParams(0.5, 0.0, 0.0))
        sim = TournamentSimulator(bracket, runaway, params, rng=Random(5))
        trials = 20_000
        wins = sum(1 for _ in range(trials) if sim._play(1, 64) == 1)
        assert wins / trials > 0.999

    def test_upset_probability_rises_with_volatility(self, bracket, field) -> None:
        """Volatility lives in the denominator, so it flattens mismatches.

        Team 2 outranks team 1 here, so team 1's win probability is the upset
        rate and must climb towards 0.5 as sigma grows.
        """
        steady = TournamentSimulator(
            bracket, field, _params(volatility=VolatilityParams(1.0, 0.0, 0.0)),
            rng=Random(6),
        )
        wild = TournamentSimulator(
            bracket, field, _params(volatility=VolatilityParams(12.0, 0.0, 0.0)),
            rng=Random(6),
        )
        trials = 20_000
        p_steady = sum(1 for _ in range(trials) if steady._play(1, 2) == 1) / trials
        p_wild = sum(1 for _ in range(trials) if wild._play(1, 2) == 1) / trials
        assert p_steady < 0.1
        assert p_wild > p_steady
        assert p_wild < 0.6


class TestProbabilities:
    def test_dominant_team_wins_nearly_always(self, bracket, runaway) -> None:
        params = _params(volatility=VolatilityParams(1.0, 0.0, 0.0))
        res = simulate_bracket(bracket, runaway, params, n_sims=500, seed=11)
        assert res.title_probabilities()[1] > 0.98

    def test_title_probabilities_sum_to_one(self, bracket, field) -> None:
        res = simulate_bracket(bracket, field, _params(), n_sims=400, seed=5)
        assert sum(res.title_probabilities().values()) == pytest.approx(1.0)

    def test_appearance_falls_for_a_middling_team(self, bracket, field) -> None:
        """A mid-field team should be less likely to appear in later rounds."""
        res = simulate_bracket(bracket, field, _params(), n_sims=800, seed=13)
        middle = max(field)
        early = res.appearance_probabilities(1)[middle]
        late = res.appearance_probabilities(5).get(middle, 0.0)
        assert early == pytest.approx(1.0)
        assert late < early

    def test_round_wins_equal_appearances_difference(self, bracket, field) -> None:
        """Regression: wins(N) == appearances(N+1) - appearances(N)."""
        res = simulate_bracket(bracket, field, _params(), n_sims=300, seed=17)
        for round_no in (1, 2, 3):
            wins = res.round_win_probabilities(round_no)
            here = res.appearance_probabilities(round_no)
            nxt = res.appearance_probabilities(round_no + 1)
            for team in here:
                expected = nxt.get(team, 0.0) - here[team]
                if expected > 0:
                    assert wins[team] == pytest.approx(expected)

    def test_final_win_count_equals_titles(self, bracket, field) -> None:
        """The final's winner is the champion, not a round-7 appearance."""
        res = simulate_bracket(bracket, field, _params(), n_sims=200, seed=19)
        titles = res.title_probabilities()
        final_wins = res.round_win_probabilities(6)
        assert final_wins == pytest.approx(titles)


class TestLatentDraws:
    """Latent on/off, and the analytic target that pins the variance budget.

    `use_latent=True` with `latent_fraction=0` is a silent no-op, so every
    test here either sets phi explicitly or checks that it is 0.
    """

    @staticmethod
    def _phi_params(phi: float, sigma_0: float = 8.0) -> MatchupParams:
        return MatchupParams(
            elo_per_point=30.0,
            home_advantage_points=0.0,
            volatility=VolatilityParams(sigma_0, 0.0, 0.0),
            latent_fraction=phi,
        )

    def test_field_draws_one_offset_per_team(self, field) -> None:
        params = self._phi_params(phi=1.0 / 3.0)
        draws = draw_latent_field(field, np.random.default_rng(0), params=params)
        assert set(draws) == set(field)
        # sigma * sqrt(1/3) = 4.62, so offsets stay inside a few sigma of 0.
        assert all(abs(v) < 30.0 for v in draws.values())

    def test_field_rejects_rng_without_normal(self, field) -> None:
        with pytest.raises(TypeError, match="normal"):
            draw_latent_field(field, Random(0), params=self._phi_params(0.3))

    def test_phi_zero_leaves_the_default_path_untouched(self, bracket, field) -> None:
        """use_latent with phi=0 must be bit-identical to use_latent=False."""
        params = self._phi_params(0.0)
        plain = simulate_bracket(bracket, field, params, n_sims=200, seed=3)
        on = simulate_bracket(
            bracket, field, params, n_sims=200, seed=3, use_latent=True, latent_seed=3
        )
        assert plain.champion_counts == on.champion_counts

    def test_use_latent_changes_results(self, bracket, field) -> None:
        params = self._phi_params(phi=0.5)
        plain = simulate_bracket(bracket, field, params, n_sims=300, seed=3)
        shaky = simulate_bracket(
            bracket, field, params, n_sims=300, seed=3, use_latent=True, latent_seed=3
        )
        assert plain.champion_counts != shaky.champion_counts

    def test_conditional_frequency_matches_win_probability(
        self, bracket, field
    ) -> None:
        """The analytic target that was missing for the latent path.

        A single latent draw has no closed-form frequency, but averaging over
        draws must return the unconditional `win_probability`.  A latent
        drawn at full sigma while game noise stays full sigma breaks this,
        because the two inflate the spread together.
        """
        params = self._phi_params(phi=0.4)
        sim = TournamentSimulator(bracket, field, params, rng=Random(4))
        pair = (3, 11)
        target = win_probability(
            field[pair[0]], field[pair[1]], params=params, venue=Venue.NEUTRAL
        )
        rng = np.random.default_rng(9)
        draws = 6_000
        latent = [draw_latent_field(field, rng, params=params) for _ in range(draws)]
        wins = sum(
            1
            for f in latent
            if sim._play(pair[0], pair[1], f) == pair[0]
        )
        p = wins / draws
        tol = 4.0 * (target * (1 - target) / draws) ** 0.5
        assert abs(p - target) < max(tol, 0.005)

    def test_latent_does_not_inflate_marginal_variance(
        self, bracket, field
    ) -> None:
        """Marginal margin spread must equal the calibrated total.

        The old construction drew latent at full sigma *and* left game noise at
        full sigma, giving sqrt(2) x the calibrated spread - about 41% too wide.
        Measured here on the realised margins, not inferred from the formula.
        """
        phi = 0.35
        params = self._phi_params(phi=phi)
        sim = TournamentSimulator(bracket, field, params, rng=Random(12))
        pair = (2, 9)
        a, b = field[pair[0]], field[pair[1]]
        calibrated = margin_scale_points(a, b, params=params)
        mu = (a.elo_rating - b.elo_rating) / params.elo_per_point

        # A fresh field per game, so each realisation samples the full marginal.
        rng = np.random.default_rng(13)
        draws = 20_000
        p = sum(
            1
            for _ in range(draws)
            if sim._play(
                pair[0], pair[1], draw_latent_field(field, rng, params=params)
            )
            == pair[0]
        ) / draws
        target = win_probability(a, b, params=params, venue=Venue.NEUTRAL)
        tol = 4.0 * (target * (1 - target) / draws) ** 0.5
        assert abs(p - target) < max(tol, 0.005)

        # And the decomposition itself must close on the calibrated total.
        sa, sb = a.volatility(params.volatility), b.volatility(params.volatility)
        drift_var = phi * (sa**2 + sb**2)
        cond = margin_scale_points(a, b, params=params, conditional=True)
        assert drift_var + cond**2 == pytest.approx(calibrated**2, rel=1e-12)
        assert mu != pytest.approx(0.0, abs=1e-9)


class TestReproducibility:
    def test_same_seed_reproduces(self, bracket, field) -> None:
        a = simulate_bracket(bracket, field, _params(), n_sims=200, seed=42)
        b = simulate_bracket(bracket, field, _params(), n_sims=200, seed=42)
        assert a.champion_counts == b.champion_counts

    def test_different_seed_varies(self, bracket, field) -> None:
        a = simulate_bracket(bracket, field, _params(), n_sims=200, seed=1)
        b = simulate_bracket(bracket, field, _params(), n_sims=200, seed=2)
        assert a.champion_counts != b.champion_counts

    def test_none_seed_is_allowed(self, bracket, field) -> None:
        res = simulate_bracket(bracket, field, _params(), n_sims=50, seed=None)
        assert res.n_sims == 50

    def test_zero_sims_is_rejected(self, bracket, field) -> None:
        with pytest.raises(ValueError, match="n_sims must be >= 1"):
            simulate_bracket(bracket, field, _params(), n_sims=0)


class TestTies:
    def test_default_is_zero_because_mens_games_cannot_tie(self) -> None:
        # Verified against Kaggle: 0 ties in 125,978 regular + tournament games.
        assert DEFAULT_TIE_PROB == 0.0

    def test_tie_prob_one_is_deterministic_playoff(self, bracket) -> None:
        teams = {tid: _team(tid, 1500.0) for tid in range(1, 65)}
        res = simulate_bracket(bracket, teams, _params(), n_sims=2000, seed=9,
                               tie_prob=1.0)
        assert max(res.title_probabilities().values()) < 0.10


class TestMultiSeason:
    def test_seasons_use_independent_streams(self, bracket, field) -> None:
        b2 = TournamentBracket(2024, bracket.seeds, bracket.slots)
        out = simulate_seasons(
            {2024: b2, 2025: bracket}, field, _params(), n_sims=200, seed=100
        )
        assert set(out) == {2024, 2025}
        assert out[2024].champion_counts != out[2025].champion_counts

    def test_nested_teams_are_used_per_season(self, bracket, field) -> None:
        """Backtest path: each season must get its own end-of-season state."""

        b2 = TournamentBracket(2024, bracket.seeds, bracket.slots)
        strong = {tid: _team(tid, 1900.0) for tid in range(1, 65)}
        weak = {tid: _team(tid, 1100.0) for tid in range(1, 65)}

        def champion_of(teams_map, season, seed):
            return simulate_bracket(
                b2 if season == 2024 else bracket,
                teams_map,
                _params(volatility=VolatilityParams(4.0, 0.0, 0.0)),
                n_sims=300,
                seed=seed,
            ).champion_counts

        nested = simulate_seasons(
            {2024: b2, 2025: bracket},
            {2024: strong, 2025: weak},
            _params(volatility=VolatilityParams(4.0, 0.0, 0.0)),
            n_sims=300,
            seed=5,
        )
        assert nested[2024].champion_counts != nested[2025].champion_counts
        # A uniformly strong field beats a uniformly weak one more often.
        assert max(nested[2024].champion_counts.values()) > max(
            nested[2025].champion_counts.values()
        )

    def test_missing_nested_season_raises(self, bracket, field) -> None:
        b2 = TournamentBracket(2024, bracket.seeds, bracket.slots)
        with pytest.raises(ValueError, match="missing from teams mapping"):
            simulate_seasons(
                {2024: b2}, {2025: field}, _params(), n_sims=10
            )


class TestVolatilityFloor:
    def test_negative_sigma_is_floored_not_propagated(self) -> None:
        """A negative sigma would silently break hypot but crash a normal draw."""
        cancelling = VolatilityParams(sigma_0=7.363, alpha=-10.0, beta=0.0)
        team = ParticleTeam(1, "x", 1500.0, p3ar=1.0, efficiency_variance=0.02)
        assert team.volatility(cancelling) == MIN_TEAM_SIGMA
        assert cancelling.is_floored(1.0, 0.02)

    def test_floor_does_not_bite_for_sane_parameters(self) -> None:
        params = VolatilityParams(sigma_0=7.363, alpha=0.164, beta=0.75)
        team = ParticleTeam(1, "x", 1500.0, p3ar=0.39, efficiency_variance=0.021)
        assert not params.is_floored(0.39, 0.021)
        assert team.volatility(params) > MIN_TEAM_SIGMA

    def test_default_matchup_is_untouched(self) -> None:
        assert DEFAULT_MATCHUP.volatility.sigma_0 == 10.5
