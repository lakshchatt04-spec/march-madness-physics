"""Win-probability machinery for the tournament simulation.

The model is a Gaussian margin with a *game-dependent* scale::

    margin_ij  =  mu_ij  +  z_i - z_j  +  eps
    mu_ij      =  (Elo_i - Elo_j + HCA) / elo_per_point        [points]
    z_k        ~  N(0, phi * sigma_k^2)     drawn once per team per tournament
    eps        ~  N(0, (1 - phi) * (sigma_i^2 + sigma_j^2))   per game

and therefore, with `tau_ij = sqrt(sigma_i^2 + sigma_j^2)`::

    P(i beats j) = Phi( (Elo_i - Elo_j + HCA) / (elo_per_point * tau_ij) ).

The `z` terms are the latent strength offsets, and `phi` splits each team's
scale between a persistent component and per-game noise.  **Unconditionally**
this leaves the total variance at `sigma_i^2 + sigma_j^2`, which is exactly
what the dyadic fit calibrated; the split only redistributes it.  Conditioning
on a realised latent field, the remaining noise is
`sqrt(1 - phi) * tau_ij`.

Setting `phi = 1` (drawing the latent at full `sigma` while *also* keeping
the full `tau`) would inflate the predictive spread to `sqrt(2) * tau_ij`,
about 41% too wide, and would flatter underdogs - a double count, since the
fit already absorbed that scatter.

Elo is the numerator and volatility is the denominator.  The mean says who
*should* win; volatility says how much the result could deviate from that.

Two design points worth stating explicitly:

*   Volatility only bites through the denominator, so it generates upsets in
    *mismatched* games.  At an exactly equal Elo gap the probability is 0.5
    whatever the volatility, because ``Phi(0) = 0.5``.  A symmetric Gaussian
    cannot express "this team is a threat even when evenly matched".
*   The latent-strength perturbation is drawn **once per team per tournament**,
    not once per game (see :func:`draw_latent_field`).  This makes upsets
    correlated within a run, which is what a hot or cold team looks like.  It
    does not change calibration, only the dependence structure.

Units
-----
``mu`` and ``tau`` are both in **points**.  ``elo_per_point`` converts the
Elo-denominated strength into that scale; volatility is point-scale by
construction (see :class:`~src.models.VolatilityParams`), and latent draws are
point-scale offsets.  There is deliberately no second unit in play.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum

from .models import DEFAULT_VOLATILITY, ParticleTeam, VolatilityParams

__all__ = [
    "Venue",
    "MatchupParams",
    "DEFAULT_MATCHUP",
    "normal_cdf",
    "expected_margin_points",
    "margin_scale_points",
    "win_probability",
    "draw_latent_strength",
    "draw_latent_field",
]


class Venue(StrEnum):
    """Venue of a game, from the perspective of the team of interest."""

    HOME = "H"
    AWAY = "A"
    NEUTRAL = "N"


def normal_cdf(x: float) -> float:
    """Standard normal CDF, in closed form, without a SciPy dependency."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


@dataclass(frozen=True)
class MatchupParams:
    """Constants linking Elo to the point scale and to volatility.

    Attributes:
        elo_per_point: Elo units equivalent to one point of expected margin.
            Roughly 30 in D-I.  This is the *only* Elo-to-points conversion in
            the pipeline, and it is supplied rather than estimated: a margin
            likelihood identifies the product ``elo_per_point * sigma_0`` and
            cannot separate the two.  Empirical sensitivity is low, because the
            ridge anchor is weak next to several thousand games.
        home_advantage_points: Home advantage in points.  Zero for tournament
            play, since the bracket is played at neutral sites.  This is the
            **total** site effect, re-estimated by the dyadic fit; the 65 Elo
            inside the sequential pass is a transient term and does not add to
            it.
        volatility: Margin volatility coefficients.  Always point-scale, and
            carrying their own feature standardisation.
        latent_fraction: `phi` in `[0, 1)` - the share of each team's
            `sigma` variance that is *persistent* across a tournament
            rather than fresh noise each game.  `0` puts all of it in
            per-game noise (no latent field); `1` puts all of it in the
            persistent component and leaves zero game noise.  Fitted from the
            ratio of between-season to within-season residual variance; see
            :func:~src.calibration.estimate_latent_fraction.  Values at or
            above `1` are rejected, since they would leave the game noise
            degenerate.
    """

    elo_per_point: float = 30.0
    home_advantage_points: float = 1.8
    volatility: VolatilityParams = DEFAULT_VOLATILITY
    latent_fraction: float = 0.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.elo_per_point) or self.elo_per_point <= 0.0:
            raise ValueError(
                f"elo_per_point must be positive and finite, got {self.elo_per_point!r}"
            )
        if not math.isfinite(self.home_advantage_points):
            raise ValueError(
                "home_advantage_points must be finite, got "
                f"{self.home_advantage_points!r}"
            )
        if not isinstance(self.volatility, VolatilityParams):
            raise TypeError(
                f"volatility must be a VolatilityParams, got {type(self.volatility).__name__}"
            )
        if (
            not math.isfinite(self.latent_fraction)
            or not 0.0 <= self.latent_fraction < 1.0
        ):
            raise ValueError(
                "latent_fraction must be in [0, 1), got "
                f"{self.latent_fraction!r}"
            )

    def hca_for(self, venue: Venue) -> float:
        """Home-advantage term in points for a team playing at ``venue``."""
        if not isinstance(venue, Venue):
            raise TypeError(f"venue must be a Venue, got {type(venue).__name__}")
        if venue is Venue.HOME:
            return self.home_advantage_points
        if venue is Venue.AWAY:
            return -self.home_advantage_points
        return 0.0


DEFAULT_MATCHUP = MatchupParams()


def expected_margin_points(
    team: ParticleTeam,
    opponent: ParticleTeam,
    *,
    params: MatchupParams = DEFAULT_MATCHUP,
    venue: Venue = Venue.NEUTRAL,
    latent: tuple[float, float] | None = None,
) -> float:
    """Expected margin in points for ``team`` over ``opponent``.

    Positive means ``team`` is favoured.

    Args:
        team: The team whose margin is being computed.
        opponent: The other team.
        params: Elo/point conversion, home advantage and volatility.
        venue: Where ``team`` is playing.
        latent: Optional ``(team_offset, opponent_offset)`` point-scale latent
            draws from :func:`draw_latent_strength`, added to the mean.  Kept
            in points so that the perturbation composes with the Elo conversion
            instead of being added to a rating.
    """
    for name, value in (("team", team), ("opponent", opponent)):
        if not isinstance(value, ParticleTeam):
            raise TypeError(f"{name} must be a ParticleTeam, got {type(value).__name__}")
    if latent is not None:
        if len(latent) != 2:
            raise ValueError("latent must be a (team, opponent) pair of offsets")
        for name, offset in zip(("team", "opponent"), latent, strict=True):
            if not math.isfinite(float(offset)):
                raise ValueError(f"{name} latent offset must be finite, got {offset!r}")
    elo_gap = float(team.elo_rating) - float(opponent.elo_rating)
    drift = (latent[0] - latent[1]) if latent is not None else 0.0
    return elo_gap / params.elo_per_point + drift + params.hca_for(venue)


def margin_scale_points(
    team: ParticleTeam,
    opponent: ParticleTeam,
    *,
    params: MatchupParams = DEFAULT_MATCHUP,
    conditional: bool = False,
) -> float:
    """Scale of the margin distribution in points, ``sqrt(sigma_i^2 + sigma_j^2)``.

    Args:
        team: The team whose scale is being computed.
        opponent: The other team.
        params: Elo/point conversion, home advantage, volatility and
            ``latent_fraction``.
        conditional: When ``True``, return the scale remaining *after*
            conditioning on a realised latent field, namely
            ``sqrt(1 - latent_fraction) * sqrt(sigma_i^2 + sigma_j^2)``.  The
            default ``False`` returns the unconditional scale, which is what
            the dyadic fit calibrated and what an unconditional win
            probability must use.
    """
    for name, value in (("team", team), ("opponent", opponent)):
        if not isinstance(value, ParticleTeam):
            raise TypeError(f"{name} must be a ParticleTeam, got {type(value).__name__}")
    total = math.hypot(
        team.volatility(params.volatility), opponent.volatility(params.volatility)
    )
    if not conditional:
        return total
    return total * math.sqrt(1.0 - params.latent_fraction)


def win_probability(
    team: ParticleTeam,
    opponent: ParticleTeam,
    *,
    params: MatchupParams = DEFAULT_MATCHUP,
    venue: Venue = Venue.NEUTRAL,
    latent: tuple[float, float] | None = None,
) -> float:
    """Probability that ``team`` beats ``opponent``, in [0, 1].

    Args:
        team: The team whose win probability is being computed.
        opponent: The other team.
        params: Elo/point conversion, home advantage and volatility.
        venue: Where ``team`` is playing.  Use :attr:`Venue.NEUTRAL` for
            tournament games.
        latent: Optional ``(team_offset, opponent_offset)`` point-scale latent
            draws from :func:`draw_latent_strength`.

    Returns:
        ``Phi(mu / tau)``, the standard normal CDF of the standardised margin.
    """
    mu = expected_margin_points(
        team, opponent, params=params, venue=venue, latent=latent
    )
    # Conditioned on a realised latent field, only the per-game share of the
    # variance is left.  Using the unconditional scale here would add the
    # latent spread a second time.
    tau = margin_scale_points(
        team, opponent, params=params, conditional=latent is not None
    )
    if tau <= 0.0:
        raise ValueError("margin scale must be positive; volatility collapsed to zero")
    return min(1.0, max(0.0, normal_cdf(mu / tau)))


def draw_latent_strength(
    team: ParticleTeam,
    rng: object,
    *,
    params: MatchupParams = DEFAULT_MATCHUP,
) -> float:
    """Draw one latent **point-scale** strength offset for ``team``.

    Call this **once per team per tournament**.  Re-drawing per game would make
    upsets independent and destroy the correlated "hot team" structure.

    The draw is an offset in *points*, not a replacement Elo rating.  Since
    :func:`expected_margin_points` already divides the Elo gap by
    ``elo_per_point``, adding a point-scale perturbation there composes
    correctly for ratings sourced from either Elo or the dyadic fit.  Returning
    a rating instead would silently add points to Elo units.
    """
    # sqrt(phi) * sigma, not sigma.  Drawing at the full sigma while leaving
    # game noise at full strength double counts the same scatter and inflates
    # the predictive spread by sqrt(2).
    sigma = team.volatility(params.volatility) * math.sqrt(
        params.latent_fraction
    )
    normal = getattr(rng, "normal", None)
    if not callable(normal):
        raise TypeError("rng must expose a normal() method, e.g. numpy.random.Generator")
    return float(normal(0.0, sigma))


def draw_latent_field(
    teams: Mapping[int, ParticleTeam],
    rng: object,
    *,
    params: MatchupParams = DEFAULT_MATCHUP,
) -> dict[int, float]:
    """Draw one persistent latent point-scale offset per team, keyed by ``team_id``."""
    return {tid: draw_latent_strength(team, rng, params=params) for tid, team in teams.items()}
