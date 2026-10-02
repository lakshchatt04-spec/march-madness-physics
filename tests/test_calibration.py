"""Tests for the dyadic calibration: parameter recovery and calibration.

The synthetic fixtures are generated from a known ground truth so that the
estimator can be checked for *bias*, not merely for producing finite numbers.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pandas as pd
import pytest

from src.calibration import (
    DyadicFit,
    FeatureScales,
    build_particle_teams_from_fit,
    estimate_latent_fraction,
    fit_dyadic,
    floored_team_fraction,
    strength_from_elo,
    summarise_fit,
)
from src.matchup import Venue, win_probability
from src.models import IDENTITY_SCALES, ParticleTeam, VolatilityParams

N_TEAMS = 100
N_GAMES = 2600
TRUTH = {"sigma_0": 10.0, "alpha": 1.5, "beta": 1.0, "hca": 2.0, "elo_per_point": 30.0}
SEASONS_PER_FIT = 20


def _simulate(
    seed: int, n_games: int, phi: float = 0.0
) -> tuple[dict[int, ParticleTeam], pd.DataFrame, np.ndarray, np.ndarray]:
    """Generate games and team states from a known dyadic truth.

    The ``W`` side must genuinely win every row, so a negative realised margin
    is relabelled by swapping the two team ids *and* the venue.  Skipping that
    swap leaves ``W`` losing roughly half the time, which silently turns every
    win-probability assertion into a test of nothing.

    `phi` splits each team scale into a persistent latent offset, redrawn
    every `SEASONS_PER_FIT` seasons, and per-game noise.  `phi=0` means
    every residual is fresh noise, which is the default.
    """
    rng = np.random.default_rng(seed)
    ids = np.arange(1100, 1100 + N_TEAMS)
    strength = rng.normal(0.0, 8.0, N_TEAMS)
    z = rng.normal(size=N_TEAMS)
    w = rng.normal(size=N_TEAMS)
    sigma = TRUTH["sigma_0"] + TRUTH["alpha"] * z + TRUTH["beta"] * w

    teams = {
        int(t): ParticleTeam(
            team_id=int(t),
            team_name=f"Team {t}",
            elo_rating=1500.0 + strength[i] * TRUTH["elo_per_point"],
            p3ar=float(0.35 + z[i] * 0.06),
            efficiency_variance=float(0.033 + w[i] * 0.008),
        )
        for i, t in enumerate(ids)
    }

    w_idx = rng.integers(0, N_TEAMS, n_games)
    l_idx = rng.integers(0, N_TEAMS, n_games)
    keep = w_idx != l_idx
    w_idx, l_idx = w_idx[keep], l_idx[keep]

    loc = rng.choice(["H", "A", "N"], w_idx.size, p=[0.4, 0.4, 0.2])
    home = np.where(loc == "H", 1.0, np.where(loc == "A", -1.0, 0.0))
    season = rng.integers(0, SEASONS_PER_FIT, w_idx.size)
    latent = (
        rng.normal(0.0, np.sqrt(phi) * sigma, (SEASONS_PER_FIT, N_TEAMS))
        if phi
        else np.zeros((SEASONS_PER_FIT, N_TEAMS))
    )
    game_sd = np.sqrt(1.0 - phi) * np.sqrt(sigma[w_idx] ** 2 + sigma[l_idx] ** 2)

    margin = (
        strength[w_idx]
        - strength[l_idx]
        + latent[season, w_idx]
        - latent[season, l_idx]
        + TRUTH["hca"] * home
        + rng.normal(0.0, game_sd)
    )

    # relabel so the W side always wins; flip the venue with the swap
    flip = margin < 0.0
    w_idx, l_idx = np.where(flip, l_idx, w_idx), np.where(flip, w_idx, l_idx)
    loc = np.where(flip, np.where(loc == "H", "A", np.where(loc == "A", "H", "N")), loc)
    margin = np.abs(margin)

    games = pd.DataFrame(
        {
            "WTeamID": ids[w_idx],
            "LTeamID": ids[l_idx],
            "WTeamScore": np.rint(72.0 + margin / 2.0).astype(int),
            "LTeamScore": np.rint(72.0 - margin / 2.0).astype(int),
            "WLoc": loc,
            "Season": 2000 + season,
        }
    )
    games = games[games.WTeamScore > games.LTeamScore].reset_index(drop=True)
    return teams, games, strength, z


@pytest.fixture(scope="module")
def fitted() -> tuple[DyadicFit, dict[int, ParticleTeam], np.ndarray]:
    teams, games, strength, _ = _simulate(0, N_GAMES)
    fit = fit_dyadic(games, teams, ridge=1.0, elo_per_point=TRUTH["elo_per_point"], max_iter=30)
    return fit, teams, strength


class TestFeatureScales:
    def test_z_scores_standardise(self) -> None:
        s = FeatureScales(p3ar_mean=0.35, p3ar_sd=0.06, effvar_mean=0.033, effvar_sd=0.008)
        assert s.z_p3ar(0.35) == pytest.approx(0.0)
        assert s.z_p3ar(0.41) == pytest.approx(1.0)
        assert s.z_effvar(0.041) == pytest.approx(1.0)

    @pytest.mark.parametrize("bad", [0.0, -1.0, math.nan, math.inf])
    def test_rejects_non_positive_sd(self, bad: float) -> None:
        with pytest.raises(ValueError, match="sd"):
            FeatureScales(p3ar_mean=0.35, p3ar_sd=bad, effvar_mean=0.03, effvar_sd=0.008)

    def test_rejects_non_finite_mean(self) -> None:
        with pytest.raises(ValueError, match="finite"):
            FeatureScales(p3ar_mean=math.nan, p3ar_sd=0.06, effvar_mean=0.03, effvar_sd=0.008)


class TestValidation:
    def test_rejects_missing_columns(self) -> None:
        teams, games, _, _ = _simulate(1, 200)
        with pytest.raises(ValueError, match="missing required columns"):
            fit_dyadic(games.drop(columns=["WLoc"]), teams)

    def test_rejects_empty_games(self) -> None:
        teams, games, _, _ = _simulate(1, 200)
        with pytest.raises(ValueError, match="at least one row"):
            fit_dyadic(games.iloc[0:0], teams)

    def test_rejects_unknown_teams(self) -> None:
        teams, games, _, _ = _simulate(1, 200)
        games = games.copy()
        games.loc[0, "WTeamID"] = 9999
        with pytest.raises(ValueError, match="absent from teams"):
            fit_dyadic(games, teams)

    def test_rejects_empty_teams(self) -> None:
        _, games, _, _ = _simulate(1, 200)
        with pytest.raises(ValueError, match="must not be empty"):
            fit_dyadic(games, {})

    @pytest.mark.parametrize("bad", [0.0, -30.0, math.inf])
    def test_rejects_bad_elo_per_point(self, bad: float) -> None:
        teams, games, _, _ = _simulate(1, 200)
        with pytest.raises(ValueError, match="elo_per_point"):
            fit_dyadic(games, teams, elo_per_point=bad)

    def test_rejects_negative_ridge(self) -> None:
        teams, games, _, _ = _simulate(1, 200)
        with pytest.raises(ValueError, match="ridge"):
            fit_dyadic(games, teams, ridge=-1.0)

    def test_rejects_unknown_team_on_lookup(self, fitted: tuple[DyadicFit, dict, np.ndarray]) -> None:
        fit, _, _ = fitted
        with pytest.raises(KeyError, match="was not present in the fit"):
            fit.strength_of(4242)


class TestParameterRecovery:
    """The fit must be unbiased, so these are bias checks, not smoke tests."""

    def test_strengths_track_truth(self, fitted: tuple[DyadicFit, dict, np.ndarray]) -> None:
        fit, teams, truth = fitted
        est = np.array([fit.strength_of(tid) for tid in sorted(teams)])
        assert np.corrcoef(est, truth)[0, 1] > 0.95
        assert float(np.mean(np.abs(est - truth))) < 2.5

    @pytest.mark.parametrize(
        ("attr", "truth", "tol"),
        [("sigma_0", TRUTH["sigma_0"], 1.5), ("alpha", TRUTH["alpha"], 1.0),
         ("beta", TRUTH["beta"], 1.0), ("home_advantage_points", TRUTH["hca"], 0.8)],
    )
    def test_parameter_near_truth(
        self, fitted: tuple[DyadicFit, dict, np.ndarray], attr: str, truth: float, tol: float
    ) -> None:
        fit, _, _ = fitted
        value = getattr(fit.volatility, attr) if attr != "home_advantage_points" else getattr(fit, attr)
        assert value == pytest.approx(truth, abs=tol), f"{attr}: {value} vs {truth}"

    def test_volatility_coefficients_are_unbiased_across_seeds(self) -> None:
        """alpha and beta must average out to truth, not sit systematically low."""
        alphas, betas = [], []
        for seed in (10, 11, 12, 13, 14):
            teams, games, _, _ = _simulate(seed, N_GAMES)
            fit = fit_dyadic(games, teams, ridge=1.0, elo_per_point=TRUTH["elo_per_point"], max_iter=30)
            alphas.append(fit.volatility.alpha)
            betas.append(fit.volatility.beta)
        assert float(np.mean(alphas)) == pytest.approx(TRUTH["alpha"], abs=0.6)
        assert float(np.mean(betas)) == pytest.approx(TRUTH["beta"], abs=0.6)

    def test_converges(self, fitted: tuple[DyadicFit, dict, np.ndarray]) -> None:
        fit, _, _ = fitted
        assert fit.converged
        assert fit.iterations < 30

    def test_log_likelihood_is_finite(self, fitted: tuple[DyadicFit, dict, np.ndarray]) -> None:
        fit, _, _ = fitted
        assert math.isfinite(fit.log_likelihood)

    def test_elo_per_point_is_reported_as_supplied(self, fitted: tuple[DyadicFit, dict, np.ndarray]) -> None:
        fit, _, _ = fitted
        assert fit.elo_per_point == pytest.approx(TRUTH["elo_per_point"])


class TestRidgeRole:
    def test_heavy_ridge_shrinks_strengths(self) -> None:
        teams, games, truth, _ = _simulate(0, N_GAMES)
        loose = fit_dyadic(games, teams, ridge=1e-6, elo_per_point=30.0, max_iter=30)
        tight = fit_dyadic(games, teams, ridge=5000.0, elo_per_point=30.0, max_iter=30)
        loose_sd = float(np.std([loose.strength_of(t) for t in sorted(teams)]))
        tight_sd = float(np.std([tight.strength_of(t) for t in sorted(teams)]))
        assert tight_sd < loose_sd

    def test_ridge_is_what_breaks_the_flat_direction(self) -> None:
        """Without the anchor the level of the strengths is arbitrary."""
        teams, games, _, _ = _simulate(0, N_GAMES)
        a = fit_dyadic(games, teams, ridge=0.1, elo_per_point=30.0, max_iter=30)
        b = fit_dyadic(games, teams, ridge=0.1, elo_per_point=30.0, max_iter=30)
        assert a.strengths == b.strengths
        assert abs(float(np.mean(list(a.strengths.values())))) < 1e-6


class TestHeldOutCalibration:
    def test_predicted_probabilities_match_observed_frequency(self) -> None:
        """The real test: are the win probabilities honest on unseen games?"""
        teams, train, _, _ = _simulate(0, N_GAMES)
        _, test, _, _ = _simulate(0, N_GAMES)  # same field, same truth
        fit = fit_dyadic(train, teams, ridge=1.0, elo_per_point=TRUTH["elo_per_point"], max_iter=30)
        params = fit.matchup_params()
        fitted_teams = build_particle_teams_from_fit(fit, teams)

        # The games frame lists the winner as W, so scoring only the W side would
        # make every observation a win and the test vacuous.  Score both sides.
        opposite = {"H": Venue.AWAY, "A": Venue.HOME, "N": Venue.NEUTRAL}
        predicted: list[float] = []
        actual: list[float] = []
        for row in test.itertuples(index=False):
            venue = Venue(str(row.WLoc))
            winner = fitted_teams[int(row.WTeamID)]
            loser = fitted_teams[int(row.LTeamID)]
            predicted.append(win_probability(winner, loser, params=params, venue=venue))
            actual.append(1.0)
            predicted.append(win_probability(loser, winner, params=params, venue=opposite[str(row.WLoc)]))
            actual.append(0.0)

        predicted_arr = np.array(predicted)
        actual_arr = np.array(actual)
        assert 0.05 < actual_arr.mean() < 0.95  # a degenerate test set proves nothing

        edges = [0.0, 0.45, 0.55, 0.65, 0.85, 1.0]
        gaps: list[float] = []
        for lo, hi in zip(edges[:-1], edges[1:], strict=True):
            mask = (predicted_arr >= lo) & (predicted_arr < hi)
            if mask.sum() < 60:
                continue
            gaps.append(float(actual_arr[mask].mean() - predicted_arr[mask].mean()))
        assert len(gaps) >= 2, "not enough populated bins to judge calibration"
        # per-bin slack, plus a tighter check on the average gap so that a
        # single lucky bin cannot carry a badly biased model
        assert all(abs(g) < 0.08 for g in gaps), f"calibration gaps {gaps}"
        assert float(np.mean(np.abs(gaps))) < 0.04, f"mean |gap| too high: {gaps}"

    def test_beats_a_coin_flip_on_log_loss(self, fitted: tuple[DyadicFit, dict, np.ndarray]) -> None:
        """A model that merely reproduces the base rate must not pass."""
        teams, _, _, _ = _simulate(0, N_GAMES)
        _, test, _, _ = _simulate(0, N_GAMES)
        fit = fit_dyadic(test, teams, ridge=1.0, elo_per_point=30.0, max_iter=30)
        params = fit.matchup_params()
        ft = build_particle_teams_from_fit(fit, teams)
        opposite = {"H": Venue.AWAY, "A": Venue.HOME, "N": Venue.NEUTRAL}
        p, a = [], []
        for row in test.itertuples(index=False):
            winner, loser = ft[int(row.WTeamID)], ft[int(row.LTeamID)]
            p.append(win_probability(winner, loser, params=params, venue=Venue(str(row.WLoc))))
            a.append(1.0)
            p.append(
                win_probability(loser, winner, params=params, venue=opposite[str(row.WLoc)])
            )
            a.append(0.0)
        p_arr, a_arr = np.clip(np.array(p), 1e-9, 1 - 1e-9), np.array(a)
        log_loss = float(
            -np.mean(a_arr * np.log(p_arr) + (1 - a_arr) * np.log(1 - p_arr))
        )
        assert log_loss < math.log(2.0) - 0.10, f"log loss {log_loss} is near coin-flip"

    def test_predictions_decrease_as_opponent_grows_stronger(
        self, fitted: tuple[DyadicFit, dict, np.ndarray]
    ) -> None:
        fit, teams, _ = fitted
        params = fit.matchup_params()
        focal = build_particle_teams_from_fit(fit, teams)[1100]
        probs = [
            win_probability(
                focal,
                ParticleTeam(1299, "opp", 1500.0 + gap, 0.35, 0.033),
                params=params,
            )
            for gap in (0.0, 100.0, 200.0, 300.0, 400.0)
        ]
        assert all(b < a for a, b in zip(probs, probs[1:], strict=False))

    def test_predictions_increase_as_team_grows_stronger(
        self, fitted: tuple[DyadicFit, dict, np.ndarray]
    ) -> None:
        fit, _, _ = fitted
        params = fit.matchup_params()
        opponent = ParticleTeam(1299, "opp", 1500.0, 0.35, 0.033)
        probs = [
            win_probability(
                ParticleTeam(1199, "t", 1500.0 + gap, 0.35, 0.033), opponent, params=params
            )
            for gap in (0.0, 100.0, 200.0, 300.0, 400.0)
        ]
        assert all(b > a for a, b in zip(probs, probs[1:], strict=False))


class TestFitAccessors:
    def test_matchup_params_carry_the_fit(self, fitted: tuple[DyadicFit, dict, np.ndarray]) -> None:
        fit, _, _ = fitted
        p = fit.matchup_params()
        assert p.elo_per_point == pytest.approx(fit.elo_per_point)
        assert p.home_advantage_points == pytest.approx(fit.home_advantage_points)
        assert p.volatility is fit.volatility

    def test_build_teams_preserves_identity_and_features(
        self, fitted: tuple[DyadicFit, dict, np.ndarray]
    ) -> None:
        fit, teams, _ = fitted
        out = build_particle_teams_from_fit(fit, teams)
        assert set(out) == set(teams)
        for tid, team in teams.items():
            assert out[tid].team_name == team.team_name
            assert out[tid].p3ar == pytest.approx(team.p3ar)
            assert out[tid].efficiency_variance == pytest.approx(team.efficiency_variance)

    def test_build_teams_recovers_elo_contrasts(self, fitted: tuple[DyadicFit, dict, np.ndarray]) -> None:
        fit, teams, _ = fitted
        out = build_particle_teams_from_fit(fit, teams)
        ids = sorted(teams)
        prior = np.array([teams[t].elo_rating for t in ids])
        derived = np.array([out[t].elo_rating for t in ids])
        assert np.corrcoef(prior, derived)[0, 1] > 0.9
        # the conversion must be monotonic: bigger strength, bigger Elo
        assert np.array_equal(np.argsort(derived), np.argsort(
            np.array([fit.strength_of(t) for t in ids])
        ))

    def test_strength_from_elo_helper(self) -> None:
        assert strength_from_elo(1530.0, elo_mean=1500.0, elo_per_point=30.0) == pytest.approx(1.0)

    def test_summarise_fit_is_readable(self, fitted: tuple[DyadicFit, dict, np.ndarray]) -> None:
        fit, _, _ = fitted
        text = "\n".join(summarise_fit(fit))
        assert "elo_per_point" in text
        assert "sigma_0" in text
        assert "converged=True" in text


class TestDeterminism:
    def test_same_input_same_output(self) -> None:
        teams, games, _, _ = _simulate(5, 1200)
        a = fit_dyadic(games, teams, ridge=1.0, elo_per_point=30.0, max_iter=20)
        b = fit_dyadic(games, teams, ridge=1.0, elo_per_point=30.0, max_iter=20)
        assert a.strengths == b.strengths
        assert a.volatility == b.volatility
        assert a.home_advantage_points == b.home_advantage_points

    def test_supplied_scales_are_used(self) -> None:
        teams, games, _, _ = _simulate(6, 800)
        scales = FeatureScales(p3ar_mean=0.35, p3ar_sd=0.06, effvar_mean=0.033, effvar_sd=0.008)
        fit = fit_dyadic(games, teams, scales=scales, elo_per_point=30.0, max_iter=20)
        assert fit.scales is scales

    def test_neutral_only_schedule_gives_zero_home_advantage(self) -> None:
        """With no home games the home-advantage column is all zeros."""
        teams, games, _, _ = _simulate(7, 1500)
        neutral = games.copy()
        neutral["WLoc"] = "N"
        fit = fit_dyadic(neutral, teams, ridge=1.0, elo_per_point=30.0, max_iter=20)
        assert abs(fit.home_advantage_points) < 1e-6


def test_volatility_params_still_validate() -> None:
    with pytest.raises(ValueError):
        VolatilityParams(sigma_0=-1.0, alpha=0.0, beta=0.0)
    with pytest.raises(TypeError):
        VolatilityParams(sigma_0=1.0, alpha=True, beta=0.0)  # type: ignore[arg-type]


def test_volatility_params_require_all_coefficients() -> None:
    """Partial specification is an error, not a silent default.

    Defaulting ``alpha``/``beta`` would let a caller write ``VolatilityParams(
    sigma_0=7.4)`` and get coefficients they never asked for - the same class
    of implicit-value bug this refactor removes.
    """
    with pytest.raises(TypeError):
        VolatilityParams(sigma_0=7.4)  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        VolatilityParams(sigma_0=7.4, alpha=0.1)  # type: ignore[call-arg]


class TestScalePropagation:
    """Regression tests for the z-score leak.

    ``fit_dyadic`` estimated ``alpha``/``beta`` against Z-scores, but
    ``DyadicFit.matchup_params()`` used to drop the ``FeatureScales``.  Callers
    then re-derived sigma from raw features, collapsing a fitted 0.787 point
    spread down to 0.009 and silently making every team equally volatile.
    """

    SCALES = FeatureScales(p3ar_mean=0.39, p3ar_sd=0.054,
                           effvar_mean=0.021, effvar_sd=0.0057)

    def test_volatility_params_uses_z_scored_features(self) -> None:
        v = VolatilityParams(sigma_0=7.371, alpha=0.141, beta=0.750,
                             scales=self.SCALES)
        s = self.SCALES
        # A team sitting exactly at both league means must land on sigma_0.
        assert v.sigma(s.p3ar_mean, s.effvar_mean) == pytest.approx(7.371)
        # One sd high on effvar adds exactly beta.
        assert v.sigma(s.p3ar_mean, s.effvar_mean + s.effvar_sd) == \
            pytest.approx(7.371 + 0.750)

    def test_identity_scales_make_the_formula_literal(self) -> None:
        v = VolatilityParams(sigma_0=1.0, alpha=2.0, beta=3.0)
        assert v.scales is IDENTITY_SCALES
        assert v.sigma(0.35, 0.02) == pytest.approx(1.0 + 2.0 * 0.35 + 3.0 * 0.02)

    def test_fit_round_trips_its_scales_into_matchup_params(self, fitted) -> None:
        fit = fitted[0]
        params = fit.matchup_params()
        assert params.volatility.scales == fit.scales

    def test_scales_actually_change_particle_volatility(self, fitted) -> None:
        """The end-to-end symptom: fitted heterogeneity must survive into teams."""
        fit, teams, _ = fitted
        spread = np.std([t.volatility(fit.volatility) for t in teams.values()])
        assert spread > 0.5, "heterogeneity collapsed: scales were dropped"

    def test_raw_features_would_reproduce_the_old_bug(self, fitted) -> None:
        """Applying z-scale coefficients to raw features is what we removed.

        Compared as a ratio, not an absolute: the synthetic coefficients are
        larger than the real fit's, so the collapsed spread is ~0.09 points
        rather than the ~0.009 seen on 2025 data.  The *ratio* is the invariant.
        """
        fit, teams, _ = fitted
        v = fit.volatility
        z_spread = np.std([t.volatility(v) for t in teams.values()])
        raw_spread = np.std([
            v.sigma_0 + v.alpha * t.p3ar + v.beta * t.efficiency_variance
            for t in teams.values()
        ])
        assert z_spread > 10.0 * raw_spread


class TestLatentFraction:
    """phi recovery: the estimator must find a synthetic truth, not just any number.

    The latent share is what stops the simulation double counting team variance,
    so the estimator feeding it needs an accuracy test of its own.
    """

    @pytest.mark.parametrize("truth", [0.0, 0.2, 0.5])
    def test_recovers_synthetic_phi(self, truth: float) -> None:
        teams, games, _, _ = _simulate(0, 9000, phi=truth)
        fit = fit_dyadic(
            games, teams, ridge=1.0, elo_per_point=TRUTH["elo_per_point"], max_iter=30
        )
        assert estimate_latent_fraction(games, teams, fit) == pytest.approx(
            truth, abs=0.12
        )

    def test_returns_zero_without_enough_history(self) -> None:
        """Too little history must yield a no-op, not a guess."""
        teams, games, _, _ = _simulate(0, 400, phi=0.4)
        fit = fit_dyadic(
            games, teams, ridge=1.0, elo_per_point=TRUTH["elo_per_point"], max_iter=10
        )
        # 400 games over 100 teams is ~4 per team, far below min_seasons.
        assert estimate_latent_fraction(games, teams, fit) == 0.0

    def test_rejects_games_without_season(self) -> None:
        teams, games, _, _ = _simulate(0, 600)
        fit = fit_dyadic(
            games, teams, ridge=1.0, elo_per_point=TRUTH["elo_per_point"], max_iter=10
        )
        with pytest.raises(ValueError, match="missing required columns"):
            estimate_latent_fraction(games.drop(columns=["Season"]), teams, fit)


class TestConvergenceCriterion:
    """The convergence flag has to mean something.

    The old criterion compared strengths between sweeps.  The ridge step is a
    direct linear solve, so strengths reach their fixed point in one sweep, and
    the flag then read ``True in 2 iters`` on every fit regardless of the data -
    reporting the shape of the algorithm rather than the behaviour of this fit.
    """

    def test_fixed_point_reached_is_genuine(self, fitted) -> None:
        fit, teams, _ = fitted
        assert fit.converged
        assert fit.iterations >= 2
        # Extra sweeps must not move anything, which is what makes stopping
        # at two legitimate rather than premature.
        teams2, games2, _, _ = _simulate(0, N_GAMES, phi=0.0)
        long_fit = fit_dyadic(
            games2, teams2, ridge=1.0, elo_per_point=TRUTH["elo_per_point"], max_iter=40
        )
        drift = max(
            abs(long_fit.strength_of(k) - fit.strength_of(k)) for k in fit.strengths
        )
        assert drift == pytest.approx(0.0, abs=1e-9)

    def test_not_converged_when_the_iteration_cap_bites(self) -> None:
        teams, games, _, _ = _simulate(0, 1500)
        fit = fit_dyadic(
            games,
            teams,
            ridge=1.0,
            elo_per_point=TRUTH["elo_per_point"],
            max_iter=1,
            tolerance=0.0,
        )
        assert not fit.converged
        assert fit.iterations == 1

    def test_zero_tolerance_still_stops_at_the_fixed_point(self) -> None:
        """An impossible tolerance must not spin to max_iter forever."""
        teams, games, _, _ = _simulate(0, 1200)
        fit = fit_dyadic(
            games,
            teams,
            ridge=1.0,
            elo_per_point=TRUTH["elo_per_point"],
            max_iter=25,
            tolerance=0.0,
        )
        assert fit.converged


class TestFlooredTeamFraction:
    """The floor is invisible in sigma_0/alpha/beta, so it needs its own report."""

    def test_zero_for_a_sane_fit(self, fitted) -> None:
        fit, teams, _ = fitted
        assert floored_team_fraction(teams, fit) == 0.0

    def test_reports_teams_whose_linear_form_goes_negative(self, fitted) -> None:
        """alpha=-10 against a high p3ar drives the raw form below zero."""
        fit, teams, _ = fitted
        hot = dict(teams)
        for tid in list(hot)[:30]:
            x = hot[tid]
            hot[tid] = ParticleTeam(x.team_id, x.team_name, x.elo_rating, 0.90, 0.02)
        broken = dataclasses.replace(
            fit, volatility=VolatilityParams(1.0, -10.0, 0.0, scales=fit.scales)
        )
        assert 0.0 < floored_team_fraction(hot, broken) < 1.0

    def test_summarise_fit_includes_the_line_only_with_teams(self, fitted) -> None:
        fit, teams, _ = fitted
        assert "floored" not in "\n".join(summarise_fit(fit))
        assert "floored teams=0.00%" in "\n".join(summarise_fit(fit, teams))
