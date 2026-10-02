"""Run a seeded Monte Carlo bracket simulation and print the leaderboard.

Usage:
    python tools/simulate_season.py --season 2025 --sims 20000
    python tools/simulate_season.py --season 2024 --seasons 2023 2024 2025
"""
from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.bracket import load_bracket  # noqa: E402
from src.calibration import (  # noqa: E402
    build_particle_teams_from_fit,
    estimate_latent_fraction,
    fit_dyadic,
    floored_team_fraction,
)
from src.data_loader import load_games, load_particle_teams, load_teams  # noqa: E402
from src.simulate import simulate_bracket  # noqa: E402

DATA_DIRNAME = "march-machine-learning-mania-2026"


def _default_data() -> Path:
    """Locate the Kaggle directory from either the root or a synced copy.

    `tools/` exists both at the repo root and inside `march-madness-physics`,
    so walking a fixed number of parents up and appending the data folder yields a
    doubled path in one of them.  Search upwards instead.
    """
    for parent in Path(__file__).resolve().parents:
        candidate = parent / DATA_DIRNAME
        if candidate.is_dir():
            return candidate
        nested = parent / "march-madness-physics" / DATA_DIRNAME
        if nested.is_dir():
            return nested
    raise FileNotFoundError(
        f"could not find {DATA_DIRNAME} above {Path(__file__).resolve()}"
    )


DEFAULT_DATA = _default_data()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=DEFAULT_DATA,
                   help="Directory holding the Kaggle M*.csv files.")
    p.add_argument("--season", type=int, default=2025,
                   help="Season whose bracket is simulated.")
    p.add_argument("--seasons", type=int, nargs="+", default=None,
                   help="Override the seasons used to fit the model.")
    p.add_argument("--sims", type=int, default=10_000, help="Brackets to play.")
    p.add_argument("--seed", type=int, default=0, help="RNG seed.")
    p.add_argument("--top", type=int, default=15, help="Leaderboard rows to show.")
    p.add_argument("--latent", action="store_true",
                   help="Draw per-team latent strength offsets, once per bracket.")
    p.add_argument("--latent-seed", type=int, default=None,
                   help="RNG seed for latent draws. Defaults to --seed.")
    p.add_argument("--quiet-fit", action="store_true",
                   help="Skip printing calibration details.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    fit_seasons = tuple(args.seasons) if args.seasons else (args.season,)

    games = load_games(args.data, seasons=fit_seasons)
    teams = load_particle_teams(args.data, seasons=fit_seasons)
    names = load_teams(args.data)
    if not args.quiet_fit:
        print(f"fitted on seasons {fit_seasons}: {len(games)} games, "
              f"{len(teams)} teams")

    fit = fit_dyadic(games, teams)
    phi = estimate_latent_fraction(games, teams, fit)
    if not args.quiet_fit:
        v = fit.volatility
        print(f"  converged={fit.converged} in {fit.iterations} iters")
        print(f"  sigma_0={v.sigma_0:.3f} alpha={v.alpha:.3f} beta={v.beta:.3f}")
        print(f"  home_advantage={fit.home_advantage_points:.3f} pts")
        floored = floored_team_fraction(teams, fit)
        if floored:
            print(f"  volatility floor hit for {floored:.1%} of teams")
        note = "" if phi else "  (undetermined - needs ~12+ seasons per team)"
        print(f"  latent_fraction phi={phi:.3f} (persistent share of sigma^2){note}")

    # Simulate the *fitted* strengths, not the sequential-Elo particles.  The
    # sequential Elo scale is a different gauge with a much narrower spread, and
    # feeding it to a volatility fitted in point-scale makes every matchup look
    # like a coin flip - see tools/seed_odds.py for the measured impact.
    sim_teams = build_particle_teams_from_fit(fit, teams)
    # phi splits each team's variance into a persistent part and per-game noise.
    # It has to reach the matchup params, otherwise the latent field is drawn at
    # full sigma while the game noise is also left at full sigma - a double
    # count that inflates the predictive spread by sqrt(2).
    matchup = dataclasses.replace(
        fit.matchup_params(),
        latent_fraction=phi if args.latent else 0.0,
    )

    bracket = load_bracket(args.data, args.season)
    result = simulate_bracket(
        bracket,
        sim_teams,
        matchup,
        n_sims=args.sims,
        seed=args.seed,
        use_latent=args.latent,
        latent_seed=args.latent_seed,
    )
    print()
    print(f"{args.season} bracket: {bracket.field_size} teams, "
          f"{args.sims:,} sims, seed={args.seed}, latent={args.latent}, "
          f"phi={matchup.latent_fraction:.3f}")
    print(result.summary(names, top=args.top))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
