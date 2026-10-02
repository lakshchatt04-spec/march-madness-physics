"""Walk-forward backtest of the bracket simulation.

For each evaluation season the model is fitted only on earlier seasons, then
asked to predict that season's tournament.  Nothing from the evaluation season
other than team features and bracket structure enters the fit.

Usage:
    python tools/backtest.py --seasons 2021 2022 2023 2024 2025
    python tools/backtest.py --seasons 2011 2025 --sims 4000 --latent
"""
from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.backtest import (  # noqa: E402
    TRIVIAL_ADVANCEMENT_BRIER,
    TRIVIAL_WIN_LOG_LOSS,
    UNIFORM_TITLE_LOG_LOSS,
    IncompleteTournamentError,
    actual_outcome,
    advancement_brier,
    brier,
    calibration_table,
    favourite_forecasts,
    reach_brier,
    reliability_forecasts,
    round_trivial_baselines,
    title_log_loss,
    win_log_loss,
    win_log_loss_by_round,
)
from src.bracket import load_bracket, load_tourney_games  # noqa: E402
from src.calibration import (  # noqa: E402
    build_particle_teams_from_fit,
    estimate_latent_fraction,
    fit_dyadic,
    unseen_strengths,
)
from src.data_loader import load_games, load_particle_teams, load_teams  # noqa: E402
from src.simulate import simulate_bracket  # noqa: E402

DATA_DIRNAME = "march-machine-learning-mania-2026"
MIN_HISTORY = 2003


def _default_data() -> Path:
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
    p.add_argument("--data", type=Path, default=DEFAULT_DATA)
    p.add_argument("--seasons", type=int, nargs="+", default=[2021, 2022, 2023, 2024, 2025],
                   help="Seasons to evaluate.  Each is predicted using only "
                        "earlier seasons.")
    p.add_argument("--first-fit-season", type=int, default=MIN_HISTORY,
                   help="Earliest season usable for fitting.")
    p.add_argument("--trailing", type=int, default=None,
                   help="Fit on this many seasons immediately before each "
                        "evaluation season instead of all history.")
    p.add_argument("--sims", type=int, default=4000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--latent", action="store_true")
    p.add_argument("--top", type=int, default=5,
                   help="Leaderboard rows to show per season.")
    p.add_argument("--no-latent", dest="latent", action="store_false",
                   help="Force phi=0 even when it is estimable.")
    p.set_defaults(latent=True)
    return p.parse_args(argv)


def _history(season: int, args: argparse.Namespace) -> tuple[int, ...]:
    if args.trailing:
        return tuple(range(season - args.trailing, season))
    return tuple(range(max(args.first_fit_season, MIN_HISTORY), season))


def evaluate_season(
    season: int, args: argparse.Namespace
) -> tuple[
    dict[str, float],
    list[str],
    list[tuple[int, float, int]],
    list[tuple[int, float, int]],
    dict[int, tuple[float, int]],
]:
    """Fit on prior seasons, predict ``season``, score against the result."""
    history = _history(season, args)
    if not history:
        raise ValueError(f"season {season} has no prior seasons to fit on")

    games = load_games(args.data, seasons=history)
    fit_teams = load_particle_teams(args.data, seasons=history)
    fit = fit_dyadic(games, fit_teams)

    # Features for the evaluation season, but strengths from the prior fit.
    season_teams = load_particle_teams(args.data, seasons=(season,))
    # An expansion team can reach the field before it has any D-I history in
    # the fit window, so fall back to the ridge prior and report it rather
    # than dropping the season.
    sim_teams = build_particle_teams_from_fit(fit, season_teams, unseen="prior")
    unseen = unseen_strengths(fit, sim_teams)

    bracket = load_bracket(args.data, season)
    missing = set(bracket.seeds.values()) - set(sim_teams)
    if missing:
        raise ValueError(
            f"season {season}: {len(missing)} bracket teams have no feature "
            f"row, e.g. {sorted(missing)[:5]}"
        )

    phi = estimate_latent_fraction(games, fit_teams, fit) if args.latent else 0.0
    matchup = dataclasses.replace(
        fit.matchup_params(), latent_fraction=phi
    )

    result = simulate_bracket(
        bracket,
        sim_teams,
        matchup,
        n_sims=args.sims,
        seed=args.seed,
        use_latent=args.latent and phi > 0.0,
    )

    outcome = actual_outcome(load_tourney_games(args.data, seasons=(season,)), bracket)
    names = load_teams(args.data)

    probs = result.title_probabilities()
    ranked = sorted(probs.items(), key=lambda kv: -kv[1])
    top = ranked[0] if ranked else (0, 0.0)

    win_ll, n_games = win_log_loss(result, outcome)

    metrics = {
        "season": float(season),
        "title_ll": title_log_loss(result, outcome.champion),
        "win_ll": win_ll,
        "adv_brier": advancement_brier(result, outcome),
        "reach_brier": reach_brier(result, outcome),
        "champ_p": probs.get(outcome.champion, 0.0),
        "top_p": top[1],
        "top_hit": 1.0 if top[0] == outcome.champion else 0.0,
        "phi": phi,
        "n_games": float(n_games),
        "n_unseen": float(len(unseen)),
        "n_history": float(len(history)),
    }

    lines = []
    if unseen:
        lines.append(
            f"  prior teams   {len(unseen)} on the ridge prior (no D-I "
            f"history in fit window)"
        )
    lines += [
        f"  champion      {names.get(outcome.champion, outcome.champion)} "
        f"(model {metrics['champ_p']:.1%}, top pick {names.get(top[0], top[0])} "
        f"at {top[1]:.1%}{' [HIT]' if top[0] == outcome.champion else ''})",
        "  field         " + ", ".join(
            f"{names.get(t, t)} {p:.1%}" for t, p in ranked[: args.top]
        ),
    ]
    # Every forecastable single-game event, tagged with its round so the
    # reliability table can be split - see main() for why late rounds have
    # to be reported separately.
    return (
        metrics,
        lines,
        reliability_forecasts(result, outcome),
        favourite_forecasts(result, outcome),
        win_log_loss_by_round(result, outcome),
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rows: list[dict[str, float]] = []
    all_forecasts: list[tuple[int, float, int]] = []
    all_favourites: list[tuple[int, float, int]] = []
    win_by_round: list[dict[int, tuple[float, int]]] = []

    skipped: list[str] = []
    for season in sorted(args.seasons):
        print(f"{season}: fitting on {args.first_fit_season}-{season - 1}")
        try:
            metrics, lines, forecasts, favourites, per_round = evaluate_season(
                season, args
            )
        except IncompleteTournamentError as exc:
            # A known defect in the source file, not a modelling failure.  Drop
            # the season and say so rather than silently scoring it as zero.
            print(f"  SKIPPED: {exc}")
            skipped.append(season)
            print()
            continue
        rows.append(metrics)
        all_forecasts.extend(forecasts)
        all_favourites.extend(favourites)
        win_by_round.append(per_round)
        print(
            f"  title log loss {metrics['title_ll']:.3f} | "
            f"win log loss {metrics['win_ll']:.3f} over {int(metrics['n_games'])} "
            f"games | adv Brier {metrics['adv_brier']:.4f} | "
            f"reach Brier {metrics['reach_brier']:.4f} | phi={metrics['phi']:.3f}"
        )
        for line in lines:
            print(line)
        print()

    if not rows:
        print("no evaluable seasons")
        return 1

    print("=" * 68)
    n = len(rows)
    mean_ll = sum(r["title_ll"] for r in rows) / n
    mean_win = sum(r["win_ll"] for r in rows) / n
    mean_adv = sum(r["adv_brier"] for r in rows) / n
    mean_reach = sum(r["reach_brier"] for r in rows) / n
    hits = sum(r["top_hit"] for r in rows) / n
    mean_top = sum(r["top_p"] for r in rows) / n

    if skipped:
        print(f"skipped (incomplete source data): {sorted(skipped)}")
    print(f"{n} seasons, {args.sims:,} sims each, seed={args.seed}")
    print(f"  mean title log loss     {mean_ll:.3f}  (vs {UNIFORM_TITLE_LOG_LOSS:.3f} uniform)")
    print(
        f"  mean win log loss       {mean_win:.3f}  "
        f"(vs {TRIVIAL_WIN_LOG_LOSS:.3f} coin flip)"
    )
    print(
        f"  mean advancement Brier  {mean_adv:.4f}  "
        f"(vs {TRIVIAL_ADVANCEMENT_BRIER:.4f} constant 0.5)"
    )
    print("  mean reach Brier        "
          f"{mean_reach:.4f}  (vs per-round base rate, below)")
    print(f"  top-pick hit rate       {hits:.1%}")
    print(f"  mean top-pick prob      {mean_top:.1%}")

    # The reach floor is not a single number: it depends on how deep the field
    # gets, so it is shown per round rather than quoted once.
    baseline_rows: dict[int, list[float]] = {}
    for season in sorted(args.seasons):
        if season in skipped:
            continue
        bracket = load_bracket(args.data, season)
        outcome = actual_outcome(load_tourney_games(args.data, seasons=(season,)), bracket)
        for round_no, base in round_trivial_baselines(outcome).items():
            baseline_rows.setdefault(round_no, []).append(base["reach"])

    if baseline_rows:
        print()
        print(f"  {'round':>5s} {'reach Brier floor':>18s}")
        for round_no in sorted(baseline_rows):
            mean_base = sum(baseline_rows[round_no]) / len(baseline_rows[round_no])
            print(f"  {round_no:>5d} {mean_base:>18.4f}")

    # Win log loss split the same way.  The pooled number is dominated by
    # whichever band the model handles worse, and the two bands point in
    # opposite directions, so the split is the whole message.
    bands: dict[str, tuple[float, int]] = {}
    for per_round in win_by_round:
        for label, keep in (
            ("rounds 1-2", lambda r: r <= 2),
            ("rounds 3-6", lambda r: r >= 3),
        ):
            subtotal = sum(m * n for r, (m, n) in per_round.items() if keep(r))
            games = sum(n for r, (_, n) in per_round.items() if keep(r))
            prior = bands.get(label, (0.0, 0))
            bands[label] = (prior[0] + subtotal, prior[1] + games)

    if bands:
        print()
        print(f"  {'band':>10s} {'win log loss':>13s} {'vs coin flip':>13s} "
              f"{'games':>7s}")
        for label in ("rounds 1-2", "rounds 3-6"):
            subtotal, games = bands[label]
            if not games:
                continue
            mean = subtotal / games
            verdict = "better" if mean < TRIVIAL_WIN_LOG_LOSS else "worse"
            print(f"  {label:>10s} {mean:>13.3f} {verdict:>13s} {games:>7d}")

    def report(label, rows_, floor=None):
        if not rows_:
            return
        pairs = [(p, y) for _, p, y in rows_]
        score = brier(pairs)
        print()
        header = f"Reliability, {label} ({len(pairs)} events, Brier {score:.4f}"
        if floor is not None:
            header += f", trivial {floor:.4f}"
        print(header + ")")
        print(f"  {'bucket':>7s} {'predicted':>10s} {'observed':>9s} "
              f"{'n':>7s} {'se':>6s}")
        for mid, mean_p, observed, count in calibration_table(pairs):
            se = (observed * (1.0 - observed) / count) ** 0.5
            print(
                f"  {mid:>7.2f} {mean_p:>9.1%} {observed:>8.1%} "
                f"{count:>7d} {se:>6.1%}"
            )

    # Advancement forecasts are split by round band because the aggregate
    # observed rate is forced to 0.5 in every round, so pooling tells you
    # nothing about calibration.  The trivial floor is quoted alongside.
    report("rounds 1-2", [r for r in all_forecasts if r[0] <= 2],
           TRIVIAL_ADVANCEMENT_BRIER)
    report("rounds 3-6", [r for r in all_forecasts if r[0] >= 3],
           TRIVIAL_ADVANCEMENT_BRIER)

    # Favourite forecasts are the diagnostic that is actually interpretable:
    # one calibrated forecast per game, base rate equal to the model's own hit
    # rate rather than pinned at 0.5 by the bracket.  They are split by round
    # band for the same reason, because the early and late games behave
    # completely differently.
    report("favourite, rounds 1-2", [r for r in all_favourites if r[0] <= 2])
    report("favourite, rounds 3-6", [r for r in all_favourites if r[0] >= 3])
    print()
    print(
        "  advancement: a calibrated model puts 'observed' next to "
        "'predicted'; observed pulled toward 50% is expected here, because "
        "each round splits its field evenly regardless of the model"
    )
    print(
        "  favourite:   observed near 'predicted' is real calibration, since "
        "the base rate here is the model's own hit rate, not a fixed 0.5"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
