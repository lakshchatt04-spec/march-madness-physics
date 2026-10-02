"""Dyadic (plus-one) calibration of team strength, scale and volatility.

The fitted model is a Gaussian margin with game-dependent scale::

    margin_ij  ~  N( s_i - s_j + hca * h_ij ,  sigma_i^2 + sigma_j^2 )
    sigma_i     =  sigma_0 + alpha * z_p3ar_i + beta * z_effvar_i

where ``z_*`` are league-wide Z-scores of the two features and ``h_ij`` is
``+1`` when the winner was at home, ``-1`` when the loser was, and ``0`` at a
neutral site.

Why a dyadic fit rather than Elo alone
--------------------------------------
A single regression of margin on Elo difference pins down the Elo-to-point
conversion, the home advantage and the residual spread in one shot.  Fitting
the mean and the scale separately also dissolves the ``c``/``sigma_0``
degeneracy, in which the two enter the likelihood only as the product
``c * sigma_0`` and so cannot be estimated apart.

Why the ridge term is mandatory, not merely regularising
--------------------------------------------------------
Each game contributes a design row of ``+1`` for the winner and ``-1`` for the
loser, so ``A^T A`` is singular: adding a constant to every strength leaves all
margins unchanged.  The strengths are therefore only identified *relative to a
fixed anchor*.  The Elo ratings supply that anchor via the penalty
``lambda * ||s - s_elo||^2``.  With no penalty the fit has a flat direction
and the solver returns an arbitrary level.

Consequence: the absolute level of every strength, and hence every fitted
home advantage, is inherited from the sequential Elo ratings.  Only the
*contrasts* between teams are learned from margins.  That is a real, and
deliberate, division of labour between the two estimators.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .matchup import MatchupParams
from .models import FeatureScales, ParticleTeam, VolatilityParams

__all__ = [
    "FeatureScales",
    "DyadicFit",
    "fit_dyadic",
    "build_particle_teams_from_fit",
    "estimate_latent_fraction",
    "floored_team_fraction",
           "unseen_strengths",
    "strength_from_elo",
    "summarise_fit",
]


@dataclass(frozen=True)
class DyadicFit:
    """Result of the dyadic calibration.

    Attributes:
        strengths: Team strength in points, keyed by ``team_id``.
        elo_per_point: Elo units per point of margin, anchored on the prior.
        home_advantage_points: Fitted home advantage in points.
        volatility: Fitted volatility coefficients, on the **points** scale.
        scales: League-wide feature standardisation used during the fit.
        ridge: Penalty weight applied to deviation from the Elo prior.
        log_likelihood: Final heteroscedastic Gaussian log-likelihood.
        iterations: Alternating-sweeps performed.
    """

    strengths: dict[int, float]
    elo_per_point: float
    home_advantage_points: float
    volatility: VolatilityParams
    scales: FeatureScales
    ridge: float
    log_likelihood: float
    iterations: int = 0
    converged: bool = False
    feature_summary: dict[str, float] = field(default_factory=dict)

    def matchup_params(self) -> MatchupParams:
        """A :class:`~src.matchup.MatchupParams` wired to this fit."""
        return MatchupParams(
            elo_per_point=self.elo_per_point,
            home_advantage_points=self.home_advantage_points,
            volatility=self.volatility,
        )

    def strength_of(self, team_id: int) -> float:
        """Fitted strength in points for ``team_id``."""
        try:
            return self.strengths[int(team_id)]
        except KeyError:
            raise KeyError(f"team {team_id} was not present in the fit") from None


def _hetero_loglik(
    resid: np.ndarray, tau: np.ndarray
) -> float:
    """Gaussian log-likelihood with per-game scale ``tau``."""
    var = tau**2
    if not np.all(np.isfinite(var)) or np.any(var <= 0.0):
        return -math.inf
    return float(-0.5 * np.sum(np.log(2.0 * math.pi * var) + resid**2 / var))


def _solve_ridge(
    design: np.ndarray, target: np.ndarray, prior: np.ndarray, ridge: float
) -> np.ndarray:
    """Ridge least squares ``min ||A s - t||^2 + ridge*||s - prior||^2``."""
    gram = design.T @ design + ridge * np.eye(design.shape[1])
    moment = design.T @ target + ridge * prior
    try:
        return np.linalg.solve(gram, moment)
    except np.linalg.LinAlgError:
        fallback = np.linalg.lstsq(gram, moment, rcond=None)[0]
        return np.asarray(fallback, dtype=float)


def _solve_ridge_with_home(
    design: np.ndarray,
    home_sign: np.ndarray,
    margin: np.ndarray,
    prior: np.ndarray,
    ridge: float,
) -> tuple[np.ndarray, float]:
    """Jointly solve for strengths and home advantage.

    Minimises ``||A s + hca*h - m||^2 + ridge*||s - prior||^2``.

    The penalty applies to the strengths only, never to ``hca``.  The
    home-advantage column is not in ``col(A)``, so it is already identified by
    the data and shrinking it would only bias it.  Estimating ``hca`` from an
    unconditional mean of margins instead - the obvious shortcut - is wrong
    unless the schedule is venue-balanced, because it silently multiplies the
    estimate by ``mean(home_sign)``, which is exactly zero for a neutral-site
    schedule.
    """
    n_teams = design.shape[1]
    augmented = np.column_stack([design, home_sign])
    penalty = np.diag([ridge] * n_teams + [0.0])
    gram = augmented.T @ augmented + penalty
    moment = augmented.T @ margin + penalty @ np.append(prior, 0.0)
    try:
        solution = np.linalg.solve(gram, moment)
    except np.linalg.LinAlgError:
        solution = np.asarray(np.linalg.lstsq(gram, moment, rcond=None)[0], dtype=float)
    return solution[:n_teams], float(solution[n_teams])



def _grid_volatility(
    resid: np.ndarray,
    z_i: np.ndarray,
    z_j: np.ndarray,
    w_i: np.ndarray,
    w_j: np.ndarray,
    start: tuple[float, float, float],
    span: tuple[float, float, float],
    steps: int,
) -> tuple[tuple[float, float, float], float]:
    """Multi-resolution grid search for ``(sigma_0, alpha, beta)``.

    A dependency-free stand-in for a nonlinear optimiser.  The three-parameter
    surface is smooth and low-dimensional, so a coarse-to-fine grid is both
    adequate and much harder to get subtly wrong than hand-rolled gradients.

    The scale for a game is ``sqrt(sigma_i^2 + sigma_j^2)`` with each team's
    sigma built from *its own* features::

        sigma_i = sigma_0 + alpha*z_i + beta*w_i
        sigma_j = sigma_0 + alpha*z_j + beta*w_j

    It is emphatically not ``sqrt(2) * (sigma_0 + alpha*z_i + beta*w_j)``:
    that applies each coefficient to the pair's combined features instead of
    to each team separately, and attenuates ``alpha`` and ``beta`` toward zero
    whenever the two teams' features differ.
    """
    best = start
    best_ll = -math.inf
    cur_span = span
    cur_start = start
    for _ in range(4):
        axes = [
            np.linspace(cur_start[k] - cur_span[k], cur_start[k] + cur_span[k], steps)
            for k in range(3)
        ]
        for s0 in axes[0]:
            if s0 <= 0.05:
                continue
            for a in axes[1]:
                for b in axes[2]:
                    tau = np.sqrt(
                        (s0 + a * z_i + b * w_i) ** 2 + (s0 + a * z_j + b * w_j) ** 2
                    )
                    ll = _hetero_loglik(resid, tau)
                    if ll > best_ll:
                        best_ll, best = ll, (float(s0), float(a), float(b))
        cur_start = best
        cur_span = tuple(s / max(steps - 1, 1) * 2.0 for s in cur_span)  # type: ignore[assignment]
    return best, best_ll


def estimate_latent_fraction(
    games: pd.DataFrame,
    teams: Mapping[int, ParticleTeam],
    fit: DyadicFit,
    *,
    min_games_per_season: int = 5,
    min_seasons: int = 12,
) -> float:
    """Estimate `phi`, the share of team variance that persists between seasons.

    A team's fitted strength is a *season-average* estimate.  If some of its
    margin scatter is a stable, season-long property rather than fresh noise
    each game, then its per-season mean residual overstates how well the
    fitted strength describes any one season.

    Method of moments, per team `k`.  Let `m_ks` be that team's mean
    residual over season `s`, over `n_ks` games, and let `v_ks` be the
    variance of those residuals.  Then

    * `Var_s(m_ks) = phi * sigma_k^2 + v_ks / n_ks` - the persistent part
      plus the sampling noise of a season mean, and
    * `sigma_k^2` is half of a two-sided game's residual variance.

    So the excess of the observed between-season variance over the average
    sampling term is `phi * sigma_k^2`.

    Two details that each cost about `1 / n_games_per_season` of accuracy if
    ignored, which is large at realistic sample sizes:

    * `v_ks` is estimated *empirically* from the cell residuals rather than
      predicted as `sigma_k^2`.  A game's residual carries both teams'
      variance, so a cell mean carries roughly `2 * sigma_k^2 / n`.  Using
      the team's own sigma understates the sampling term and inflates `phi`
      - by ~0.19 on a synthetic fixture at true `phi = 0`.
    * the variance of `T` season means is taken with `ddof=1`, since
      `T` sample means understate their own spread by `(T-1)/T`.

    Returns:
        `phi` clipped to `[0, 0.95]`.  The upper bound leaves a sliver of
        per-game noise, so `1 - phi` never reaches zero.  Returns `0.0`
        when too few teams have enough seasons to estimate anything, which
        makes the latent mechanism a no-op rather than a guess.
    """
    required = {"WTeamID", "LTeamID", "WTeamScore", "LTeamScore", "WLoc", "Season"}
    missing = required - set(games.columns)
    if missing:
        raise ValueError(f"games is missing required columns: {sorted(missing)}")
    if "Season" not in games.columns:
        raise ValueError("games must carry a Season column to estimate phi")

    margin = (
        games["WTeamScore"].to_numpy(dtype=float)
        - games["LTeamScore"].to_numpy(dtype=float)
    )
    loc = games["WLoc"].astype(str).str.strip().str.upper()
    home_sign = np.where(loc.eq("H"), 1.0, np.where(loc.eq("A"), -1.0, 0.0))
    winners = games["WTeamID"].to_numpy()
    losers = games["LTeamID"].to_numpy()
    seasons = games["Season"].to_numpy()

    try:
        pred = np.array([fit.strength_of(int(t)) for t in winners]) - np.array(
            [fit.strength_of(int(t)) for t in losers]
        )
    except KeyError as exc:
        raise ValueError(f"games reference teams absent from the fit: {exc}") from None
    resid = margin - (pred + fit.home_advantage_points * home_sign)

    per_cell: dict[tuple[int, int], list[float]] = {}
    for season, win, lose, r in zip(seasons, winners, losers, resid, strict=True):
        per_cell.setdefault((int(win), int(season)), []).append(float(r))
        per_cell.setdefault((int(lose), int(season)), []).append(-float(r))

    by_team: dict[int, list[tuple[int, float, float, int]]] = {}
    for (team_id, season), values in per_cell.items():
        if team_id not in teams:
            continue
        # A cell needs at least 2 games for its own variance to exist.
        if len(values) < max(min_games_per_season, 2):
            continue
        by_team.setdefault(team_id, []).append(
            (
                season,
                float(np.mean(values)),
                float(np.var(values, ddof=1)),
                len(values),
            )
        )

    volatility = fit.volatility
    ratios: list[float] = []
    for team_id, cells in by_team.items():
        if len(cells) < min_seasons:
            continue
        means = np.array([c[1] for c in cells], dtype=float)
        cell_var = np.array([c[2] for c in cells], dtype=float)
        counts = np.array([c[3] for c in cells], dtype=float)
        sigma2 = teams[team_id].volatility(volatility) ** 2
        if sigma2 <= 0.0:
            continue
        observed = float(np.var(means, ddof=1))
        sampling = float(np.mean(cell_var / counts))
        ratios.append((observed - sampling) / sigma2)

    if not ratios:
        return 0.0
    return float(min(max(float(np.mean(ratios)), 0.0), 0.95))


def fit_dyadic(
    games: pd.DataFrame,
    teams: Mapping[int, ParticleTeam],
    *,
    ridge: float = 1.0,
    elo_per_point: float = 30.0,
    max_iter: int = 60,
    tolerance: float = 1e-7,
    initial_volatility: VolatilityParams | None = None,
    scales: FeatureScales | None = None,
) -> DyadicFit:
    """Fit team strengths, Elo scale, home advantage and volatility jointly.

    Args:
        games: Long frame of completed games.  Must contain ``WTeamID``,
            ``LTeamID``, ``WTeamScore``, ``LTeamScore`` and ``WLoc``.
        teams: The :class:`~src.models.ParticleTeam` states supplying the
            Elo prior and the two volatility features.
        ridge: Penalty weight on deviation from the Elo prior.  Also the
            parameter that makes the strengths identifiable at all; values
            near zero leave a flat direction in the likelihood.
        elo_per_point: Elo units per point of margin.  This converts the Elo
            prior into the point scale on which the strengths are fitted.  It
            is deliberately a *separate* argument from any volatility
            coefficient: the Elo-to-point conversion and the volatility scale
            are unrelated quantities and must not be conflated.
        max_iter: Cap on alternating sweeps.
        tolerance: Convergence threshold on the *relative* change in the
            heteroscedastic log-likelihood between sweeps.  A strength-based
            threshold would be vacuous here, since the ridge step is a direct
            solve that reaches its fixed point immediately.
        initial_volatility: Starting coefficients.  Must be on the points
            scale, since it seeds the scale that the margin likelihood sees.
        scales: Pre-computed feature standardisation.  Recomputed from
            ``teams`` when omitted.

    Returns:
        A :class:`DyadicFit`.

    Raises:
        ValueError: If ``games`` is empty, lacks required columns, or refers to
            teams absent from ``teams``.
    """
    if not math.isfinite(elo_per_point) or elo_per_point <= 0.0:
        raise ValueError(f"elo_per_point must be positive and finite, got {elo_per_point!r}")
    if ridge < 0.0 or not math.isfinite(ridge):
        raise ValueError(f"ridge must be non-negative and finite, got {ridge!r}")

    required = {"WTeamID", "LTeamID", "WTeamScore", "LTeamScore", "WLoc"}
    missing = required - set(games.columns)
    if missing:
        raise ValueError(f"games is missing required columns: {sorted(missing)}")
    if len(games) == 0:
        raise ValueError("games must contain at least one row")

    ids = np.array(sorted(int(t) for t in teams), dtype=np.int64)
    if ids.size == 0:
        raise ValueError("teams must not be empty")
    index = {int(t): i for i, t in enumerate(ids)}

    winners = games["WTeamID"].to_numpy()
    losers = games["LTeamID"].to_numpy()
    unknown = ({int(x) for x in winners} | {int(x) for x in losers}) - set(index)
    if unknown:
        raise ValueError(f"games reference teams absent from teams: {sorted(unknown)}")

    margin = (
        games["WTeamScore"].to_numpy(dtype=float)
        - games["LTeamScore"].to_numpy(dtype=float)
    )
    loc = games["WLoc"].astype(str).str.strip().str.upper()
    home_sign = np.where(
        loc.eq("H"), 1.0, np.where(loc.eq("A"), -1.0, 0.0)
    )

    w_idx = np.array([index[int(x)] for x in winners], dtype=np.int64)
    l_idx = np.array([index[int(x)] for x in losers], dtype=np.int64)

    design = np.zeros((margin.size, ids.size), dtype=float)
    design[np.arange(margin.size), w_idx] = 1.0
    design[np.arange(margin.size), l_idx] = -1.0

    if scales is None:
        p3ar = np.array([float(teams[int(t)].p3ar) for t in ids])
        effvar = np.array([float(teams[int(t)].efficiency_variance) for t in ids])
        scales = FeatureScales(
            p3ar_mean=float(p3ar.mean()),
            p3ar_sd=float(p3ar.std(ddof=0)) or 1.0,
            effvar_mean=float(effvar.mean()),
            effvar_sd=float(effvar.std(ddof=0)) or 1.0,
        )
    z_p3ar = np.array([scales.z_p3ar(float(teams[int(t)].p3ar)) for t in ids])
    z_effvar = np.array([scales.z_effvar(float(teams[int(t)].efficiency_variance)) for t in ids])

    # The Elo prior, expressed in points via the Elo-to-point conversion.
    # This is the anchor that resolves the additive non-identifiability.
    elo = np.array([float(teams[int(t)].elo_rating) for t in ids])
    elo_centered = elo - float(elo.mean())
    initial = initial_volatility or VolatilityParams(sigma_0=10.0, alpha=0.0, beta=0.0)
    prior_strength = elo_centered / float(elo_per_point)

    hca = 0.0
    params = (float(initial.sigma_0), float(initial.alpha), float(initial.beta))

    strengths = prior_strength.copy()
    converged = False
    iters = 0
    loglik = -math.inf
    previous_loglik = -math.inf

    for iteration in range(1, max_iter + 1):
        iters = iteration
        strengths, hca = _solve_ridge_with_home(
            design, home_sign, margin, prior_strength, ridge
        )

        resid = margin - (design @ strengths + hca * home_sign)
        params, loglik = _grid_volatility(
            resid,
            z_p3ar[w_idx],
            z_p3ar[l_idx],
            z_effvar[w_idx],
            z_effvar[l_idx],
            start=params,
            span=(max(params[0] * 0.75, 1.0), 2.0, 2.0),
            steps=9,
        )

        # Convergence is judged on the log-likelihood, not on the strengths.
        # The ridge sweep is a direct linear solve, so the strengths reach their
        # fixed point in one step and the volatility grid barely perturbs them:
        # on the full 2003-2025 history the strengths move by exactly 0.0 between
        # sweeps.  A strength-change criterion therefore reports "converged" on
        # the second iteration no matter what, which is why this fit always
        # claimed 2.  The objective being maximised is the log-likelihood, so
        # that is what gets watched.  Requiring at least two iterations means
        # "the objective stopped moving" is never claimed without a prior value
        # to compare against.
        ll_shift = abs(loglik - previous_loglik)
        if iteration >= 2 and ll_shift <= tolerance * max(abs(loglik), 1.0):
            converged = True
            break
        previous_loglik = loglik

    elo_per_point = float(elo_per_point)
    # The scales travel inside the volatility object.  alpha and beta were fitted
    # against Z-scores, so carrying them without these would silently apply
    # z-scale coefficients to raw features.
    volatility = VolatilityParams(
        sigma_0=params[0],
        alpha=params[1],
        beta=params[2],
        scales=scales,
    )
    return DyadicFit(
        strengths={int(t): float(s) for t, s in zip(ids, strengths, strict=True)},
        elo_per_point=elo_per_point,
        home_advantage_points=float(hca),
        volatility=volatility,
        scales=scales,
        ridge=ridge,
        log_likelihood=float(loglik),
        iterations=iters,
        converged=converged,
        feature_summary={
            "n_teams": float(ids.size),
            "n_games": float(margin.size),
            "p3ar_mean": scales.p3ar_mean,
            "p3ar_sd": scales.p3ar_sd,
            "effvar_mean": scales.effvar_mean,
            "effvar_sd": scales.effvar_sd,
        },
    )


def floored_team_fraction(
    teams: Mapping[int, ParticleTeam], fit: DyadicFit
) -> float:
    """Share of teams whose fitted volatility was clamped by the floor.

    A team is floored when the linear form ``sigma_0 + alpha*z + beta*w`` comes
    out non-positive, meaning the fitted heterogeneity is extreme enough to
    leave the valid range.  Worth reporting: it is the only sign that the
    volatility surface is being extrapolated rather than interpolated, and it
    is invisible in ``sigma_0``/``alpha``/``beta`` alone.
    """
    if not teams:
        return 0.0
    vol = fit.volatility
    return float(
        sum(
            1
            for team in teams.values()
            if vol.is_floored(team.p3ar, team.efficiency_variance)
        )
        / len(teams)
    )


def strength_from_elo(
    elo: float, *, elo_mean: float, elo_per_point: float
) -> float:
    """Convert a raw Elo rating to a point-scale strength."""
    return (float(elo) - float(elo_mean)) / float(elo_per_point)


def unseen_strengths(
    fit: DyadicFit, teams: Mapping[int, ParticleTeam]
) -> set[int]:
    """TeamIDs in ``teams`` that the fit never observed."""
    return set(teams) - set(fit.strengths)


def build_particle_teams_from_fit(
    fit: DyadicFit,
    teams: Mapping[int, ParticleTeam],
    *,
    unseen: str = "raise",
) -> dict[int, ParticleTeam]:
    """Return copies of ``teams`` whose ``elo_rating`` carries fitted strength.

    Lets the fitted dyadic strengths flow through the existing
    :class:`~src.models.ParticleTeam` contract without changing the downstream
    matchup code: strength in points is converted back to Elo units.
    """
    if unseen not in ("raise", "prior"):
        raise ValueError(
            f"unseen must be 'raise' or 'prior', got {unseen!r}"
        )
    out: dict[int, ParticleTeam] = {}
    for tid, team in teams.items():
        if unseen == "prior" and tid not in fit.strengths:
            # The ridge penalty pulls an unobserved strength to exactly the
            # field mean, which is 0 in the centred gauge the fit uses.
            strength_pts = 0.0
        else:
            strength_pts = fit.strength_of(tid)
        out[tid] = ParticleTeam(
            team_id=team.team_id,
            team_name=team.team_name,
            elo_rating=strength_pts * fit.elo_per_point,
            p3ar=team.p3ar,
            efficiency_variance=team.efficiency_variance,
        )
    return out


def summarise_fit(
    fit: DyadicFit, teams: Mapping[int, ParticleTeam] | None = None
) -> Sequence[str]:
    """Short human-readable report of a fit, for calibration diagnostics.

    Pass ``teams`` to include the floored-share line, which is the only signal
    that the volatility surface was extrapolated outside its valid range.
    """
    lines = [
        f"teams={int(fit.feature_summary.get('n_teams', 0))} "
        f"games={int(fit.feature_summary.get('n_games', 0))}",
        f"elo_per_point={fit.elo_per_point:.3f} "
        f"hca={fit.home_advantage_points:+.3f} pts",
        f"sigma_0={fit.volatility.sigma_0:.3f} "
        f"alpha={fit.volatility.alpha:+.3f} beta={fit.volatility.beta:+.3f} (points)",
        f"loglik={fit.log_likelihood:.1f} iters={fit.iterations} "
        f"converged={fit.converged}",
    ]
    if teams:
        lines.append(f"floored teams={floored_team_fraction(teams, fit):.2%}")
    return lines
