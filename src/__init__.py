"""Data pipeline, state-variable models and matchup machinery.

Layering, bottom up:

* :mod:`src.models` - the :class:`ParticleTeam` state container and volatility
  coefficients.
* :mod:`src.data_loader` - reads the Kaggle CSV exports into those states.
* :mod:`src.matchup` - turns two states into a win probability.
* :mod:`src.calibration` - fits strengths, Elo scale, home advantage and
  volatility from historical game margins, anchored on the sequential Elo.
"""

from src.bracket import (
    TournamentBracket,
    load_bracket,
    load_seeds,
    load_slots,
    load_tourney_games,
)
from src.calibration import DyadicFit, fit_dyadic
from src.data_loader import (
    DataSchemaError,
    EloConfig,
    load_games,
    load_particle_teams,
    load_teams,
)
from src.matchup import (
    DEFAULT_MATCHUP,
    MatchupParams,
    Venue,
    draw_latent_field,
    draw_latent_strength,
    win_probability,
)
from src.models import (
    DEFAULT_VOLATILITY,
    IDENTITY_SCALES,
    FeatureScales,
    ParticleTeam,
    VolatilityParams,
)
from src.simulate import (
    SimulationResult,
    TournamentSimulator,
    simulate_bracket,
    simulate_seasons,
)

__all__ = [
    "ParticleTeam",
    "VolatilityParams",
    "DEFAULT_VOLATILITY",
    "FeatureScales",
    "IDENTITY_SCALES",
    "MatchupParams",
    "DEFAULT_MATCHUP",
    "Venue",
    "win_probability",
    "draw_latent_strength",
    "draw_latent_field",
    "DyadicFit",
    "fit_dyadic",
    "DataSchemaError",
    "EloConfig",
    "load_games",
    "load_teams",
    "load_particle_teams",
    "TournamentBracket",
    "load_bracket",
    "load_seeds",
    "load_slots",
    "load_tourney_games",
    "SimulationResult",
    "TournamentSimulator",
    "simulate_bracket",
    "simulate_seasons",
]
