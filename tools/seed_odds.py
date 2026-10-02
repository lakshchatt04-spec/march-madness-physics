"""Reproduce the strength-scale defect, and guard against its return.

Two ways of rating a team end up on wildly different scales:

* the sequential Elo pass, whose spread is tens of Elo; and
* the dyadic fit, whose spread is hundreds of Elo.

``win_probability`` divides the Elo gap by ``elo_per_point * sigma``, where sigma
is fitted in *points*.  Feeding the narrow sequential scale into that denominator
compresses every margin towards zero, so a 200-Elo lead looks like a coin flip
and the simulated field produces hundreds of co-favourites.

This script prints, for one fit window, the realised title spread under both
scales.  It is a diagnostic rather than a test, because the exact magnitudes
depend on the fit window: pooled multi-season fits put both paths near the same
Brier score, while a single-season fit exposes the defect sharply.  See the
module docstring in ``src/matchup.py`` for the unit argument.

Usage:
    python tools/seed_odds.py --seasons 2025
    python tools/seed_odds.py --seasons 2023 2024 2025
"""
from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.bracket import load_bracket  # noqa: E402
from src.calibration import build_particle_teams_from_fit, fit_dyadic  # noqa: E402
from src.data_loader import load_games, load_particle_teams  # noqa: E402
from src.simulate import simulate_bracket  # noqa: E402

DATA_DIRNAME = "march-machine-learning-mania-2026"


def _default_data() -> Path:
    for parent in Path(__file__).resolve().parents:
        for candidate in (parent / DATA_DIRNAME,
                          parent / "march-madness-physics" / DATA_DIRNAME):
            if candidate.is_dir():
                return candidate
    raise FileNotFoundError(f"could not find {DATA_DIRNAME}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=_default_data())
    p.add_argument("--seasons", type=int, nargs="+", default=[2025],
                   help="Seasons used to fit. A single season exposes the "
                        "defect most clearly.")
    p.add_argument("--bracket", type=int, default=2025,
                   help="Bracket season to simulate.")
    p.add_argument("--sims", type=int, default=2_000)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def _spread(probs: dict[int, float]) -> tuple[float, float]:
    values = sorted(probs.values(), reverse=True)
    return values[0], sum(values[:3])


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    fit_seasons = tuple(args.seasons)

    games = load_games(args.data, seasons=fit_seasons)
    teams = load_particle_teams(args.data, seasons=fit_seasons)
    fit = fit_dyadic(games, teams)

    elo = [t.elo_rating for t in teams.values()]
    fitted_pts = [fit.strength_of(tid) for tid in teams]
    sd_elo = statistics.pstdev(elo)
    sd_pts = statistics.pstdev(fitted_pts)
    print(f"fitted on {fit_seasons}: {len(games)} games, {len(teams)} teams")
    print(f"  sequential Elo spread (sd) = {sd_elo:.2f} Elo "
          f"= {sd_elo / fit.elo_per_point:.2f} pts")
    print(f"  fitted strength spread (sd) = {sd_pts:.2f} pts "
          f"= {sd_pts * fit.elo_per_point:.2f} Elo")
    print(f"  elo_per_point               = {fit.elo_per_point:.2f}")
    print()

    bracket = load_bracket(args.data, args.bracket)
    matchup = fit.matchup_params()
    fitted_teams = build_particle_teams_from_fit(fit, teams)

    print(f"{args.bracket} bracket, {args.sims:,} sims, fit window {fit_seasons}")
    for label, field in (("sequential Elo", teams), ("fitted strengths", fitted_teams)):
        result = simulate_bracket(
            bracket, field, matchup, n_sims=args.sims, seed=args.seed
        )
        top, top3 = _spread(result.title_probabilities())
        print(f"  {label:18s} max title = {top:6.2%}   top-3 = {top3:6.2%}")
    print()
    print("  A healthy fit concentrates: one clear favourite, a tail of longshots.")
    print("  The sequential-Elo row concentrating *worse* means the field is too flat.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
