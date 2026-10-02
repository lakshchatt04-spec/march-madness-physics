"""State-variable containers for the thermodynamics-inspired tournament model.

A team is treated as a particle whose end-of-season state is a small vector of
observables.  Those observables are produced by :mod:`src.data_loader` and are
consumed downstream by the tournament simulation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

__all__ = ["FeatureScales", "IDENTITY_SCALES", "VolatilityParams",
           "DEFAULT_VOLATILITY", "MIN_TEAM_SIGMA", "ParticleTeam"]


@dataclass(frozen=True)
class FeatureScales:
    """League-wide location and spread of the two volatility features.

    Standardising is what lets one penalty treat ``alpha`` and ``beta`` fairly;
    it does **not** reduce collinearity, since VIF is invariant to rescaling.

    These live here rather than in :mod:`src.calibration` so that
    :class:`VolatilityParams` can carry them without a circular import.  A
    calibration module must never be needed just to *interpret* a fitted
    coefficient.
    """

    p3ar_mean: float
    p3ar_sd: float
    effvar_mean: float
    effvar_sd: float

    def __post_init__(self) -> None:
        for name in ("p3ar_sd", "effvar_sd"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite, got {value!r}")
        for name in ("p3ar_mean", "p3ar_sd", "effvar_mean", "effvar_sd"):
            if not math.isfinite(float(getattr(self, name))):
                raise ValueError(f"{name} must be finite")

    def z_p3ar(self, value: float) -> float:
        return (float(value) - self.p3ar_mean) / self.p3ar_sd

    def z_effvar(self, value: float) -> float:
        return (float(value) - self.effvar_mean) / self.effvar_sd


IDENTITY_SCALES = FeatureScales(p3ar_mean=0.0, p3ar_sd=1.0,
                                effvar_mean=0.0, effvar_sd=1.0)
"""Neutral standardisation: ``z(x) == x``.  The default for hand-built
coefficients, so that a literal ``sigma_0 + alpha*p3ar + beta*effvar`` still
means exactly what it says."""


@dataclass(frozen=True)
class VolatilityParams:
    """Coefficients for the heteroscedastic margin scale, in **points**.

    ``sigma_i = sigma_0 + alpha * z_p3ar_i + beta * z_effvar_i``

    Units
    -----
    ``sigma`` is a standard deviation of the **margin in points**.  This is the
    only unit the coefficient set supports.  A single scale means the rest of
    the pipeline cannot receive an Elo-denominated sigma by accident, and
    :func:`ParticleTeam.volatility` always yields points.

    The Elo-to-point conversion is a *separate* concern and lives in
    :class:`src.matchup.MatchupParams.elo_per_point`.  Keeping the two apart is
    not cosmetic: in a margin likelihood they enter only through the product
    ``elo_per_point * sigma_0`` and so are not separately identifiable.

    Standardisation
    ---------------
    The coefficients are defined against the Z-scores produced by ``scales``.
    The scales therefore travel *inside* this object rather than being passed
    alongside it, so a fitted ``alpha``/``beta`` cannot be applied to raw
    features by a caller who forgets them.  The default is
    :data:`IDENTITY_SCALES`, which makes the formula literal.
    """

    sigma_0: float
    alpha: float
    beta: float
    scales: FeatureScales = field(default=IDENTITY_SCALES)

    def __post_init__(self) -> None:
        for name in ("sigma_0", "alpha", "beta"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise TypeError(f"{name} must be a real number, got {type(value).__name__}")
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite, got {value!r}")
        # alpha and beta are free to be negative - a feature may genuinely
        # reduce volatility - but the baseline must stay a usable scale.
        if self.sigma_0 <= 0.0:
            raise ValueError(f"sigma_0 must be positive, got {self.sigma_0}")
        if not isinstance(self.scales, FeatureScales):
            raise TypeError(f"scales must be a FeatureScales, got {type(self.scales).__name__}")

    def sigma(self, p3ar: float, efficiency_variance: float) -> float:
        """The point-scale standard deviation for a team with these features.

        The linear form can go negative for an out-of-range feature pair, and a
        negative standard deviation is silently *wrong* rather than merely
        inconvenient: `math.hypot` squares it away, so callers that only
        combine two sigmas see no error, while a direct normal draw raises
        `ValueError: scale < 0`.  Both paths consume this value, so it is
        floored here to keep them consistent.
        """
        raw = (
            self.sigma_0
            + self.alpha * self.scales.z_p3ar(p3ar)
            + self.beta * self.scales.z_effvar(efficiency_variance)
        )
        return max(raw, MIN_TEAM_SIGMA)

    def is_floored(self, p3ar: float, efficiency_variance: float) -> bool:
        """True when :meth:sigma had to clamp a non-positive raw value.

        Exposed so calibration can report how often the linear form leaves the
        valid range rather than silently producing a floored number.
        """
        raw = (
            self.sigma_0
            + self.alpha * self.scales.z_p3ar(p3ar)
            + self.beta * self.scales.z_effvar(efficiency_variance)
        )
        return raw < MIN_TEAM_SIGMA


MIN_TEAM_SIGMA = 0.5
"""Floor for a per-team volatility, in points.

Small but positive.  A team whose game-to-game margin scatter genuinely
vanishes would be a modelling curiosity, not a real team, and the floor keeps
every consumer (Gaussian sampling, `math.hypot`, log-loss) on one consistent
scale.
"""


DEFAULT_VOLATILITY = VolatilityParams(sigma_0=10.5, alpha=5.0, beta=0.5)
"""The project-brief prior, in points, with identity standardisation.

This is a deliberately *inflated* placeholder: real D-I data fits
``sigma_0 ~ 7.4`` points.  Production runs should use the output of
:func:`src.calibration.fit_dyadic`, which supplies both the coefficients and
the matching :class:`FeatureScales`.
"""


@dataclass(frozen=True)
class ParticleTeam:
    """End-of-season state variables for a single team.

    Attributes:
        team_id: Kaggle-assigned 4-digit team identifier.
        team_name: Human readable team name from ``MTeams.csv``.
        elo_rating: Regular-season Elo rating as of the final game processed.
        p3ar: 3-Point Attempt Rate, ``sum(FGA3) / sum(FGA)`` over the season.
        efficiency_variance: Population variance of game-level offensive
            efficiency (``Points / Possessions``) across the season.
    """

    team_id: int
    team_name: str
    elo_rating: float
    p3ar: float
    efficiency_variance: float

    def __post_init__(self) -> None:
        if isinstance(self.team_id, bool) or not isinstance(self.team_id, (int,)):
            raise TypeError(f"team_id must be an int, got {type(self.team_id).__name__}")
        if self.team_id <= 0:
            raise ValueError(f"team_id must be positive, got {self.team_id}")
        if not isinstance(self.team_name, str) or not self.team_name.strip():
            raise ValueError(f"team_name must be a non-empty string, got {self.team_name!r}")

        for name in ("elo_rating", "p3ar", "efficiency_variance"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a real number, got {type(value).__name__}")
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite, got {value!r}")

        if not 0.0 <= float(self.p3ar) <= 1.0:
            raise ValueError(f"p3ar must lie in [0, 1], got {self.p3ar}")
        if self.efficiency_variance < 0.0:
            raise ValueError(
                f"efficiency_variance must be non-negative, got {self.efficiency_variance}"
            )

    @property
    def internal_volatility(self) -> float:
        """Intra-team margin standard deviation ``sigma_i``, in points.

        ``sigma_i = sigma_0 + alpha * z_p3ar + beta * z_effvar`` under the
        default coefficients.  A standard deviation, not a variance; see
        :attr:`internal_volatility_variance` for the squared form.
        """
        return self.volatility(DEFAULT_VOLATILITY)

    @property
    def internal_volatility_variance(self) -> float:
        """Intra-team margin variance in points-squared, under the defaults.

        Equivalent to ``internal_volatility ** 2``.  This is the quantity the
        margin likelihood uses, since a normal likelihood needs a variance.
        """
        return self.internal_volatility**2

    def volatility(self, params: VolatilityParams | None = None) -> float:
        """Intra-team margin standard deviation ``sigma_i``, in points.

        Args:
            params: Coefficients and their feature standardisation.  Defaults
                to :data:`DEFAULT_VOLATILITY`.

        Returns:
            ``sigma_i`` for this team, in points.
        """
        effective = DEFAULT_VOLATILITY if params is None else params
        if not isinstance(effective, VolatilityParams):
            raise TypeError(
                f"params must be a VolatilityParams, got {type(effective).__name__}"
            )
        return effective.sigma(self.p3ar, self.efficiency_variance)

    def volatility_variance(self, params: VolatilityParams | None = None) -> float:
        """Intra-team margin variance in points-squared, under explicit coefficients.

        Args:
            params: Coefficients and their feature standardisation.  Defaults
                to :data:`DEFAULT_VOLATILITY`.

        Returns:
            ``sigma_i ** 2`` for this team, in points-squared.
        """
        return self.volatility(params) ** 2

    def to_dict(self) -> dict[str, object]:
        """Flat mapping of the state variables, including both volatility forms."""
        return {
            "team_id": self.team_id,
            "team_name": self.team_name,
            "elo_rating": float(self.elo_rating),
            "p3ar": float(self.p3ar),
            "efficiency_variance": float(self.efficiency_variance),
            "internal_volatility": self.internal_volatility,
            "internal_volatility_variance": self.internal_volatility_variance,
        }
