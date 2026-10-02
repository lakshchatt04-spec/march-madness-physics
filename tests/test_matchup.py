"""Tests for the matchup win-probability machinery."""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from src.matchup import (
    DEFAULT_MATCHUP,
    MatchupParams,
    Venue,
    draw_latent_field,
    draw_latent_strength,
    expected_margin_points,
    margin_scale_points,
    normal_cdf,
    win_probability,
)
from src.models import MIN_TEAM_SIGMA, ParticleTeam, VolatilityParams

KANSAS = 1246
DUKE = 1251
UNC = 1254
GONZAGA = 1229
FLAT = VolatilityParams(sigma_0=10.0, alpha=0.0, beta=0.0)


def team(team_id: int, elo: float, *, name: str | None = None) -> ParticleTeam:
    return ParticleTeam(
        team_id=team_id,
        team_name=name or f"Team {team_id}",
        elo_rating=elo,
        p3ar=0.35,
        efficiency_variance=0.02,
    )


class TestNormalCdf:
    @pytest.mark.parametrize(
        ("x", "expected"),
        [
            (0.0, 0.5),
            (1.0, 0.841_344_746),
            (1.644_853_627, 0.95),
            (1.959_963_985, 0.975),
            (-1.959_963_985, 0.025),
            (2.575_829_304, 0.995),
        ],
    )
    def test_matches_known_quantiles(self, x: float, expected: float) -> None:
        assert normal_cdf(x) == pytest.approx(expected, abs=1e-9)

    def test_symmetric_about_zero(self) -> None:
        for x in (0.3, 1.1, 2.9):
            assert normal_cdf(x) + normal_cdf(-x) == pytest.approx(1.0, abs=1e-12)

    def test_monotone_increasing(self) -> None:
        xs = np.linspace(-6, 6, 241)
        vals = [normal_cdf(float(x)) for x in xs]
        assert all(b >= a for a, b in zip(vals, vals[1:], strict=False))


class TestMatchupParams:
    def test_rejects_non_positive_elo_per_point(self) -> None:
        for bad in (0.0, -1.0, -30.0):
            with pytest.raises(ValueError, match="elo_per_point"):
                MatchupParams(elo_per_point=bad)

    def test_rejects_non_finite_home_advantage(self) -> None:
        with pytest.raises(ValueError, match="home_advantage_points"):
            MatchupParams(home_advantage_points=math.inf)

    def test_rejects_bad_volatility_type(self) -> None:
        with pytest.raises(TypeError, match="VolatilityParams"):
            MatchupParams(volatility={"sigma_0": 1.0})  # type: ignore[arg-type]

    def test_hca_sign_per_venue(self) -> None:
        p = MatchupParams(home_advantage_points=2.0)
        assert p.hca_for(Venue.HOME) == pytest.approx(2.0)
        assert p.hca_for(Venue.AWAY) == pytest.approx(-2.0)
        assert p.hca_for(Venue.NEUTRAL) == 0.0

    def test_rejects_bad_venue(self) -> None:
        with pytest.raises(TypeError, match="Venue"):
            DEFAULT_MATCHUP.hca_for("H")  # type: ignore[arg-type]

    def test_is_frozen(self) -> None:
        with pytest.raises(dataclasses.FrozenInstanceError):
            DEFAULT_MATCHUP.elo_per_point = 1.0  # type: ignore[misc]


class TestMarginAndScale:
    def test_expected_margin_is_elo_gap_over_scale(self) -> None:
        p = MatchupParams(elo_per_point=30.0, home_advantage_points=0.0)
        assert expected_margin_points(
            team(KANSAS, 1800.0), team(DUKE, 1500.0), params=p
        ) == pytest.approx(10.0)

    def test_neutral_expected_margin_is_antisymmetric(self) -> None:
        a, b = team(KANSAS, 1800.0), team(DUKE, 1500.0)
        p = MatchupParams(home_advantage_points=0.0)
        assert expected_margin_points(a, b, params=p) == pytest.approx(
            -expected_margin_points(b, a, params=p)
        )

    def test_home_advantage_adds_points(self) -> None:
        a, b = team(KANSAS, 1800.0), team(DUKE, 1500.0)
        neutral = expected_margin_points(a, b, params=MatchupParams(home_advantage_points=0.0))
        home = expected_margin_points(
            a, b, params=MatchupParams(home_advantage_points=2.0), venue=Venue.HOME
        )
        away = expected_margin_points(
            a, b, params=MatchupParams(home_advantage_points=2.0), venue=Venue.AWAY
        )
        assert home == pytest.approx(neutral + 2.0)
        assert away == pytest.approx(neutral - 2.0)

    def test_away_venue_matches_swapped_home_venue(self) -> None:
        """P(i beats j, i away) + P(j beats i, j home) must be 1."""
        p = MatchupParams(home_advantage_points=2.0)
        a, b = team(KANSAS, 1750.0), team(DUKE, 1650.0)
        a_wins_away = win_probability(a, b, params=p, venue=Venue.AWAY)
        b_wins_home = win_probability(b, a, params=p, venue=Venue.HOME)
        assert a_wins_away + b_wins_home == pytest.approx(1.0, abs=1e-12)
        # a is the stronger team, so it is favoured even while away
        assert a_wins_away > 0.5

    def test_scale_is_hypot_of_two_sigmas(self) -> None:
        p = MatchupParams(volatility=FLAT)
        a, b = team(KANSAS, 1500.0), team(DUKE, 1500.0)
        assert margin_scale_points(a, b, params=p) == pytest.approx(
            math.hypot(10.0, 10.0)
        )

    def test_scale_uses_each_teams_own_features(self) -> None:
        vol = VolatilityParams(sigma_0=10.0, alpha=5.0, beta=0.0)
        p = MatchupParams(volatility=vol)
        low = ParticleTeam(KANSAS, "a", 1500.0, p3ar=0.10, efficiency_variance=0.0)
        high = ParticleTeam(DUKE, "b", 1500.0, p3ar=0.90, efficiency_variance=0.0)
        assert margin_scale_points(low, high, params=p) == pytest.approx(
            math.hypot(10.5, 14.5)
        )

    def test_rejects_non_particle_teams(self) -> None:
        a = team(KANSAS, 1500.0)
        with pytest.raises(TypeError, match="ParticleTeam"):
            margin_scale_points(a, {"elo": 1500})  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="ParticleTeam"):
            expected_margin_points({"elo": 1500}, a)  # type: ignore[arg-type]


class TestWinProbability:
    def test_equal_elo_neutral_is_exactly_half(self) -> None:
        p = MatchupParams(volatility=FLAT)
        assert win_probability(
            team(KANSAS, 1500.0), team(DUKE, 1500.0), params=p
        ) == pytest.approx(0.5, abs=1e-12)

    @pytest.mark.parametrize("sigma", [1.0, 10.0, 50.0, 200.0])
    def test_equal_elo_is_half_at_every_volatility(self, sigma: float) -> None:
        """The documented limitation: volatility cannot act on an even game."""
        p = MatchupParams(
            elo_per_point=30.0,
            home_advantage_points=0.0,
            volatility=VolatilityParams(sigma_0=sigma, alpha=0.0, beta=0.0),
        )
        assert win_probability(
            team(KANSAS, 1500.0), team(DUKE, 1500.0), params=p
        ) == pytest.approx(0.5, abs=1e-12)

    def test_sum_to_one_for_neutral_venue(self) -> None:
        p = MatchupParams(home_advantage_points=0.0)
        a, b = team(KANSAS, 1780.0), team(DUKE, 1610.0)
        assert win_probability(a, b, params=p) + win_probability(b, a, params=p) == \
            pytest.approx(1.0, abs=1e-12)

    def test_matches_hand_computed_normal(self) -> None:
        p = MatchupParams(elo_per_point=30.0, home_advantage_points=0.0, volatility=FLAT)
        a, b = team(KANSAS, 1560.0), team(DUKE, 1500.0)
        # mu = 60/30 = 2 points ; tau = hypot(10,10) = 14.142 ; z = 2/14.142
        expected = normal_cdf(2.0 / math.hypot(10.0, 10.0))
        assert win_probability(a, b, params=p) == pytest.approx(expected, abs=1e-12)

    def test_more_volatile_opponent_lowers_favourite_probability(self) -> None:
        p = MatchupParams(home_advantage_points=0.0, volatility=FLAT)
        fav, dog = team(KANSAS, 1800.0), team(DUKE, 1400.0)
        steady = win_probability(fav, dog, params=p)
        chaotic = win_probability(
            fav,
            ParticleTeam(DUKE, "d", 1400.0, p3ar=0.35, efficiency_variance=0.06),
            params=MatchupParams(
                home_advantage_points=0.0,
                volatility=VolatilityParams(sigma_0=10.0, alpha=0.0, beta=200.0),
            ),
        )
        assert chaotic < steady

    def test_monotone_in_own_elo(self) -> None:
        p = MatchupParams(home_advantage_points=0.0)
        prev = -1.0
        for elo in (1200.0, 1400.0, 1500.0, 1600.0, 1800.0, 2000.0):
            val = win_probability(team(KANSAS, elo), team(DUKE, 1500.0), params=p)
            assert val > prev
            prev = val

    def test_monotone_against_opponent_elo(self) -> None:
        p = MatchupParams(home_advantage_points=0.0)
        prev = 2.0
        for elo in (1200.0, 1400.0, 1500.0, 1600.0, 1800.0):
            val = win_probability(team(KANSAS, 1500.0), team(DUKE, elo), params=p)
            assert val < prev
            prev = val

    def test_home_advantage_raises_home_win_probability(self) -> None:
        a, b = team(KANSAS, 1500.0), team(DUKE, 1500.0)
        p0 = win_probability(a, b, params=MatchupParams(home_advantage_points=0.0))
        p2 = win_probability(
            a, b, params=MatchupParams(home_advantage_points=2.0), venue=Venue.HOME
        )
        assert 0.5 < p2 < 1.0
        assert p2 > p0

    def test_output_always_in_unit_interval(self) -> None:
        vol = VolatilityParams(sigma_0=0.001, alpha=0.0, beta=0.0)
        p = MatchupParams(volatility=vol, home_advantage_points=0.0)
        for gap in (-3000.0, -100.0, 0.0, 100.0, 3000.0):
            v = win_probability(team(KANSAS, 1500.0 + gap), team(DUKE, 1500.0), params=p)
            assert 0.0 <= v <= 1.0

    def test_collapsing_features_floor_instead_of_zeroing(self) -> None:
        """Features can drive sigma negative; it is floored, never returned as 0.

        Returning exactly 0 made `win_probability` raise on realistic inputs
        (alpha=-10 with a high p3ar), which is a crash rather than a model
        behaviour.  The floor keeps the matchup well defined.
        """
        cancelling = VolatilityParams(sigma_0=1.0, alpha=-10.0, beta=0.0)
        p = MatchupParams(volatility=cancelling)
        a = ParticleTeam(KANSAS, "a", 1500.0, p3ar=0.1, efficiency_variance=0.0)
        b = ParticleTeam(DUKE, "b", 1400.0, p3ar=0.1, efficiency_variance=0.0)
        assert cancelling.is_floored(0.1, 0.0)
        scale = margin_scale_points(a, b, params=p)
        assert scale == pytest.approx(MIN_TEAM_SIGMA * math.sqrt(2))
        assert 0.0 < win_probability(a, b, params=p) < 1.0

    def test_volatility_params_rejects_non_positive_sigma_0(self) -> None:
        for bad in (0.0, -1.0):
            with pytest.raises(ValueError, match="sigma_0 must be positive"):
                VolatilityParams(sigma_0=bad, alpha=0.0, beta=0.0)

    def test_volatility_compresses_towards_one_half(self) -> None:
        """The headline mechanism: bigger sigma, more upset-prone."""
        fav, dog = team(KANSAS, 1800.0), team(DUKE, 1400.0)
        probs = []
        for sigma in (1.0, 5.0, 12.0, 20.0, 35.0, 60.0):
            p = MatchupParams(
                elo_per_point=30.0,
                home_advantage_points=0.0,
                volatility=VolatilityParams(sigma_0=sigma, alpha=0.0, beta=0.0),
            )
            probs.append(win_probability(fav, dog, params=p))
        # sigma=1.0 makes the favourite an overwhelming near-certainty
        assert probs[0] == pytest.approx(1.0, abs=1e-9)
        assert all(b < a for a, b in zip(probs, probs[1:], strict=False))
        assert probs[-1] > 0.4
        assert probs[-1] < 0.75

    def test_known_value_matches_reference_distribution(self) -> None:
        """Cross-check against a Monte Carlo simulation of the same model."""
        p = MatchupParams(
            elo_per_point=30.0, home_advantage_points=0.0,
            volatility=VolatilityParams(sigma_0=9.0, alpha=0.0, beta=0.0),
        )
        rng = np.random.default_rng(12345)
        a, b = team(KANSAS, 1700.0), team(DUKE, 1550.0)
        n = 400_000
        mu = expected_margin_points(a, b, params=p)
        tau = margin_scale_points(a, b, params=p)
        sim = float((rng.normal(mu, tau, n) > 0.0).mean())
        assert win_probability(a, b, params=p) == pytest.approx(sim, abs=0.004)


class TestLatentDraws:
    """Every case pins ``phi`` explicitly.

    ``DEFAULT_MATCHUP.latent_fraction`` is 0, which makes the field identically
    zero.  That is the right default for the latent-*off* path, but it would
    make every sd assertion below vacuous, so these tests opt in.
    """

    @staticmethod
    def _full(phi: float = 1.0 / 3.0, **kw) -> MatchupParams:
        return dataclasses.replace(DEFAULT_MATCHUP, latent_fraction=phi, **kw)

    def test_draw_is_reproducible_with_seed(self) -> None:
        a = team(KANSAS, 1500.0)
        p = self._full()
        first = draw_latent_strength(a, np.random.default_rng(7), params=p)
        second = draw_latent_strength(a, np.random.default_rng(7), params=p)
        assert first == second

    def test_draw_centres_on_zero(self) -> None:
        # The draw is an *offset*, so its centre is 0 regardless of the rating.
        # this team has p3ar=0.35, effvar=0.02, so the default coefficients give
        # sigma = 10.5 + 5.0*0.35 + 0.5*0.02 = 12.26, not the bare sigma_0
        a = team(KANSAS, 1500.0)
        phi = 1.0 / 3.0
        p = self._full(phi)
        expected_sd = (10.5 + 5.0 * 0.35 + 0.5 * 0.02) * math.sqrt(phi)
        draws = np.array(
            [
                draw_latent_strength(a, np.random.default_rng(i), params=p)
                for i in range(20_000)
            ]
        )
        assert draws.mean() == pytest.approx(0.0, abs=1.0)
        assert draws.std() == pytest.approx(expected_sd, rel=0.03)

    def test_default_phi_zero_draws_nothing(self) -> None:
        """The shipped default must not perturb anything."""
        a = team(KANSAS, 1500.0)
        assert DEFAULT_MATCHUP.latent_fraction == 0.0
        assert draw_latent_strength(a, np.random.default_rng(1)) == 0.0

    def test_draw_offset_is_independent_of_rating(self) -> None:
        """A rating change must not move the offset: it is a perturbation, not a rating."""
        p = self._full()
        a = team(KANSAS, 1500.0)
        b = ParticleTeam(KANSAS, "a", 2000.0, p3ar=0.35, efficiency_variance=0.02)
        assert draw_latent_strength(a, np.random.default_rng(4), params=p) == \
            draw_latent_strength(b, np.random.default_rng(4), params=p)

    def test_offset_composes_in_points(self) -> None:
        """Regression guard: the offset must be added in points, not to the rating.

        A 400 Elo gap is 13.33 points.  If the offset were treated as Elo the
        expected margin would move by ~32x more than intended.
        """
        p = MatchupParams(elo_per_point=30.0, home_advantage_points=0.0)
        fav = team(KANSAS, 1900.0)
        dog = team(DUKE, 1500.0)
        base = expected_margin_points(fav, dog, params=p)
        shifted = expected_margin_points(fav, dog, params=p, latent=(2.0, 0.0))
        assert base == pytest.approx(400.0 / 30.0)
        assert shifted - base == pytest.approx(2.0)

    def test_draw_respects_team_specific_volatility(self) -> None:
        vol = VolatilityParams(sigma_0=10.0, alpha=5.0, beta=0.0)
        p = MatchupParams(volatility=vol, latent_fraction=1.0 / 3.0)
        low = ParticleTeam(KANSAS, "a", 1500.0, p3ar=0.10, efficiency_variance=0.0)
        high = ParticleTeam(DUKE, "b", 1500.0, p3ar=0.90, efficiency_variance=0.0)
        lo = np.array([draw_latent_strength(low, np.random.default_rng(i), params=p) for i in range(20_000)])
        hi = np.array([draw_latent_strength(high, np.random.default_rng(i), params=p) for i in range(20_000)])
        assert hi.std() > lo.std() + 1.0

    def test_field_draws_exactly_once_per_team(self) -> None:
        """Persistence is the whole point: a second call must differ."""
        p = self._full()
        teams = {i: team(i, 1500.0 + i) for i in (KANSAS, DUKE, UNC, GONZAGA)}
        first = draw_latent_field(teams, np.random.default_rng(21), params=p)
        second = draw_latent_field(teams, np.random.default_rng(21), params=p)
        assert set(first) == set(teams)
        assert first == second  # same seed -> same tournament
        third = draw_latent_field(teams, np.random.default_rng(22), params=p)
        assert third != first

    def test_field_is_internally_consistent(self) -> None:
        p = self._full()
        teams = {i: team(i, 1500.0) for i in (KANSAS, DUKE, UNC, GONZAGA)}
        field = draw_latent_field(teams, np.random.default_rng(3), params=p)
        assert len(set(field.values())) == len(teams)

    def test_rejects_bad_rng(self) -> None:
        with pytest.raises(TypeError, match="normal"):
            draw_latent_strength(team(KANSAS, 1500.0), object(), params=self._full())


class TestLatentVarianceBudget:
    """The latent field and the game noise must split one variance budget.

    Regression: the latent draw used to be taken at the full ``sigma`` while the
    per-game noise was also left at the full ``sigma``, so the predictive spread
    came out ``sqrt(2)`` too wide - about 41% - and every underdog was flattered.
    These tests pin the invariant that makes that impossible to reintroduce.
    """

    @staticmethod
    def _params(phi: float, sigma_0: float = 8.0) -> MatchupParams:
        return MatchupParams(
            elo_per_point=30.0,
            home_advantage_points=0.0,
            volatility=VolatilityParams(sigma_0=sigma_0, alpha=0.0, beta=0.0),
            latent_fraction=phi,
        )

    @pytest.mark.parametrize("phi", [0.0, 0.25, 0.5, 0.9])
    def test_total_variance_is_preserved(self, phi: float) -> None:
        """Latent variance plus game variance must equal the calibrated total."""
        p = self._params(phi)
        a, b = team(KANSAS, 1500.0), team(DUKE, 1400.0)
        sa = a.volatility(p.volatility)
        sb = b.volatility(p.volatility)
        total = margin_scale_points(a, b, params=p)
        conditional = margin_scale_points(a, b, params=p, conditional=True)
        # Var(z_a - z_b) = phi * (sa^2 + sb^2), each offset drawn once per field.
        drift_var = phi * (sa**2 + sb**2)
        assert drift_var + conditional**2 == pytest.approx(total**2, rel=1e-12)

    def test_full_sigma_latent_would_double_count(self) -> None:
        """Documents the magnitude of the old error, so it cannot come back."""
        a, b = team(KANSAS, 1500.0), team(DUKE, 1400.0)
        phi = 1.0 / 3.0
        p = self._params(phi)
        sa = a.volatility(p.volatility)
        sb = b.volatility(p.volatility)
        total = margin_scale_points(a, b, params=p)
        old = math.sqrt(total**2 + sa**2 + sb**2)
        assert old / total == pytest.approx(math.sqrt(2.0), rel=1e-9)

    def test_latent_draw_sd_is_scaled_by_sqrt_phi(self) -> None:
        """The empirical sd of the draw must be ``sqrt(phi) * sigma``."""
        phi = 0.25
        p = self._params(phi)
        a = team(KANSAS, 1500.0)
        expected = a.volatility(p.volatility) * math.sqrt(phi)
        draws = np.array(
            [
                draw_latent_strength(a, np.random.default_rng(i), params=p)
                for i in range(20_000)
            ]
        )
        assert float(draws.std(ddof=0)) == pytest.approx(expected, rel=0.03)
        # The old behaviour drew at the full sigma.
        assert float(draws.std(ddof=0)) < a.volatility(p.volatility) * 0.75

    def test_phi_zero_means_a_silent_no_op(self) -> None:
        """With phi=0 the field is exactly zero, not merely small."""
        p = self._params(0.0)
        teams = {i: team(i, 1500.0 + i) for i in (KANSAS, DUKE)}
        assert draw_latent_field(teams, np.random.default_rng(0), params=p) == {
            KANSAS: 0.0,
            DUKE: 0.0,
        }
        a, b = team(KANSAS, 1500.0), team(DUKE, 1400.0)
        assert margin_scale_points(a, b, params=p, conditional=True) == pytest.approx(
            margin_scale_points(a, b, params=p)
        )

    def test_marginal_probability_is_unbiased(self) -> None:
        """Averaging over the latent field must return the unconditional P.

        This is the analytic target the latent path previously lacked: draw a
        field, condition on it, then average.  A double-counted latent makes
        the average drift away from the fitted probability.
        """
        p = self._params(0.3)
        a, b = team(KANSAS, 1500.0), team(DUKE, 1400.0)
        teams = {KANSAS: a, DUKE: b}
        target = win_probability(a, b, params=p)
        rng = np.random.default_rng(11)
        estimates = [
            win_probability(
                a,
                b,
                params=p,
                latent=(lambda f: (f[KANSAS], f[DUKE]))(
                    draw_latent_field(teams, rng, params=p)
                ),
            )
            for _ in range(4_000)
        ]
        mean = float(np.mean(estimates))
        tol = 4.0 * float(np.std(estimates, ddof=0)) / math.sqrt(len(estimates))
        assert abs(mean - target) < max(tol, 0.004)

    def test_latent_fraction_is_validated(self) -> None:
        for bad in (-0.1, 1.0, 1.5, math.nan, math.inf):
            with pytest.raises(ValueError, match="latent_fraction must be in"):
                dataclasses.replace(DEFAULT_MATCHUP, latent_fraction=bad)
