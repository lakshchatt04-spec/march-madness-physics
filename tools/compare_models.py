"""Does the dyadic fit earn its keep over sequential Elo alone?

Both models predict the same 67 neutral-site tournament games.  Elo uses the
existing 65-Elo/400 logistic; the dyadic model uses fitted point-scale
strengths and fitted volatility.  Neither sees the tournament.
"""
import math

import numpy as np

from src.calibration import fit_dyadic
from src.data_loader import EloConfig, load_games, load_particle_teams

games = load_games("data")
teams = load_particle_teams("data")
tour = load_games("data/MTourneyDetailedResults.csv")
tour = tour[tour["Season"] == 2025]

fit = fit_dyadic(games, teams, ridge=1.0, elo_per_point=30.0)
v, sc = fit.volatility, fit.scales
cfg = EloConfig()


def ncdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def sigma_of(tid: int) -> float:
    t = teams[tid]
    return v.sigma_0 + v.alpha * sc.z_p3ar(t.p3ar) + v.beta * sc.z_effvar(t.efficiency_variance)


def logloss(ps: list[float]) -> float:
    return -float(np.mean([math.log(min(max(p, 1e-12), 1 - 1e-12)) for p in ps]))


def report(name: str, probs: list[float], labels: list[int]) -> None:
    ll = logloss(probs)
    brier = float(np.mean([(p - y) ** 2 for p, y in zip(probs, labels, strict=True)]))
    acc = float(np.mean([(p > 0.5) == bool(y) for p, y in zip(probs, labels, strict=True)]))
    print(f"  {name:28s} logloss={ll:.4f}  brier={brier:.4f}  fav-acc={acc:.4f}")


labels = [1] * len(tour)

# 1) Elo alone, neutral site.  A point-scale sigma becomes an Elo-scale sigma by
# multiplying by elo_per_point; the logistic's 400-point scale is the *rating*
# scale, not a points-to-Elo conversion.
elo_p, dy_p, flat_p = [], [], []
sigma_flat_pts = 10.0
sigma_flat_elo = sigma_flat_pts * fit.elo_per_point
for r in tour.itertuples(index=False):
    w_id, l_id = int(r.WTeamID), int(r.LTeamID)
    gap_elo = teams[w_id].elo_rating - teams[l_id].elo_rating
    elo_p.append(ncdf(gap_elo / sigma_flat_elo))
    mu = fit.strength_of(w_id) - fit.strength_of(l_id)
    dy_p.append(ncdf(mu / math.hypot(sigma_of(w_id), sigma_of(l_id))))
    flat_p.append(ncdf(mu / math.hypot(v.sigma_0, v.sigma_0)))

print(f"NCAA TOURNAMENT 2025  n={len(tour)}  (neutral site, true out-of-sample)")
report("coin flip", [0.5] * len(tour), labels)
report("Elo only (K=20,HCA=65)", elo_p, labels)
report("dyadic, flat sigma", flat_p, labels)
report("dyadic, feature sigma", dy_p, labels)
print()
print(f"fitted: sigma_0={v.sigma_0:.3f} alpha={v.alpha:+.3f} beta={v.beta:+.3f} hca={fit.home_advantage_points:+.3f}")
print(f"beta/alpha ratio = {v.beta / v.alpha:.2f}  (efficiency variance dominates p3ar)")
