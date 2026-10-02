"""Is home advantage being double counted between the Elo pass and the dyadic fit?

The Elo ratings are produced with a 65-Elo site adjustment, and the dyadic fit
then estimates its own hca on top of a prior built from those same ratings.  If
the 65 Elo were leaking into the strengths, the fitted hca would understate the
true value.

Test: refit with an Elo prior that was built with home_advantage = 0.  If the
fitted hca is genuinely a *total*, it should barely move.  If the 65 Elo were
double counted, hca should jump by roughly 65/30 = 2.17 points.
"""
import dataclasses

import numpy as np

from src.calibration import fit_dyadic
from src.data_loader import (
    EloConfig,
    ParticleTeam,
    _expected_score,
    load_games,
    load_particle_teams,
)

games = load_games("data")
teams = load_particle_teams("data")

print("EloConfig() =", EloConfig())
print()
print("frozen defaults used by the pipeline:")
c = EloConfig()
for f in ("initial_rating", "k_factor", "scale", "home_advantage",
          "reset_each_season", "mov_enabled", "mov_reference",
          "mov_exponent", "mov_max_multiplier"):
    print(f"   {f:20s} = {getattr(c, f)}")
print()
print("effective K per game with the shipped defaults:")
m = games.WTeamScore - games.LTeamScore
print(f"   mov_enabled={c.mov_enabled} -> multiplier is 1.0 for every game")
print(f"   so k_eff = {c.k_factor} on all {len(games)} games")
print(f"   margin actually ranges {m.min()}..{m.max()} but is ignored without MOV")
print()


def elo_ratings(cfg: EloConfig) -> dict[int, float]:
    r: dict[int, float] = {}
    for row in sorted(games.itertuples(index=False), key=lambda x: (x.Season, x.DayNum)):
        wr = r.get(row.WTeamID, cfg.initial_rating)
        lr = r.get(row.LTeamID, cfg.initial_rating)
        adv = cfg.home_advantage if row.WLoc == "H" else (
            -cfg.home_advantage if row.WLoc == "A" else 0.0)
        d = cfg.k_factor * (1.0 - _expected_score(wr, lr, adv, cfg.scale))
        r[row.WTeamID] = wr + d
        r[row.LTeamID] = lr - d
    return r


def with_elo(ratings: dict[int, float]) -> dict[int, ParticleTeam]:
    return {
        int(tid): dataclasses.replace(teams[tid], elo_rating=float(val))
        for tid, val in ratings.items()
    }


base = fit_dyadic(games, teams, ridge=1.0, elo_per_point=30.0)
no_hca_elo = elo_ratings(EloConfig(home_advantage=0.0))
alt = fit_dyadic(games, with_elo(no_hca_elo), ridge=1.0, elo_per_point=30.0)

print("DOUBLE-COUNTING TEST (fitted hca is a TOTAL, so it should be stable)")
print(f"  prior Elo built with HCA=65  -> fitted hca = {base.home_advantage_points:+.3f} pts")
print(f"  prior Elo built with HCA=0   -> fitted hca = {alt.home_advantage_points:+.3f} pts")
print(
    f"  difference                     = "
    f"{alt.home_advantage_points - base.home_advantage_points:+.3f} pts"
    f"   (65 Elo / 30 = 2.167 pts)"
)
print(
    f"  sigma_0                       = {base.volatility.sigma_0:.3f} "
    f"vs {alt.volatility.sigma_0:.3f}"
)
print()
print("  -> if the difference is near zero, the two HCA terms are NOT additive;")
print("     the fitted hca is the total, and the 65 Elo is only a transient")
print("     term inside the sequential pass.")
print()

# independent matched-pairs estimate on the clean data
cfg = EloConfig()
elo: dict[int, float] = {}
rows = []
for r in sorted(games.itertuples(index=False), key=lambda x: (x.Season, x.DayNum)):
    wr = elo.get(r.WTeamID, cfg.initial_rating)
    lr = elo.get(r.LTeamID, cfg.initial_rating)
    adv = cfg.home_advantage if r.WLoc == "H" else (-cfg.home_advantage if r.WLoc == "A" else 0.0)
    d = cfg.k_factor * (1.0 - _expected_score(wr, lr, adv, cfg.scale))
    elo[r.WTeamID] = wr + d
    elo[r.LTeamID] = lr - d
    rows.append((abs((wr + adv) - lr), adv, r.WTeamScore - r.LTeamScore))
a = np.array(rows)
print("INDEPENDENT MATCHED-PAIRS CHECK (roughly even games only)")
print(f"  {'elo-gap':<12s} {'n':>6s} {'meanMgnH':>10s} {'meanMgnA':>10s} {'diff':>10s}")
for lo, hi in ((0, 50), (50, 100), (100, 150)):
    m2 = (a[:, 0] >= lo) & (a[:, 0] < hi) & (a[:, 1] != 0)
    H = m2 & (a[:, 1] > 0)
    A = m2 & (a[:, 1] < 0)
    if H.sum() < 30 or A.sum() < 30:
        continue
    print(
        f"  {f'{lo}-{hi}':<12s} {m2.sum():>6d} {a[H, 2].mean():>+10.2f} "
        f"{a[A, 2].mean():>+10.2f} {a[H, 2].mean() - a[A, 2].mean():>+10.2f}"
    )
print()
print("  note: the 65-Elo HCA is already inside these ratings, so the 'diff'")
print("  column is the *residual* site effect, not the total.")
