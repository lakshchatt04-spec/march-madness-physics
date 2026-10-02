"""Empirical home-advantage check on the real 2025 season data.

Diagnostic only: it replays the sequential Elo pass and reports how often home
teams won, plus the margin difference between home and away games.  Nothing here
is asserted - the numbers are printed so a human can eyeball them.
"""
import numpy as np

from src.data_loader import EloConfig, _expected_score, load_games

games = load_games("data")
cfg = EloConfig()
elo: dict[int, float] = {}
rows = []
for r in sorted(games.itertuples(index=False), key=lambda r: (r.Season, r.DayNum)):
    wr = elo.get(r.WTeamID, cfg.initial_rating)
    lr = elo.get(r.LTeamID, cfg.initial_rating)
    adv = (
        cfg.home_advantage
        if r.WLoc == "H"
        else (-cfg.home_advantage if r.WLoc == "A" else 0.0)
    )
    d = cfg.k_factor * (1.0 - _expected_score(wr, lr, adv, cfg.scale))
    elo[r.WTeamID] = wr + d
    elo[r.LTeamID] = lr - d
    rows.append(
        (
            abs((wr + adv) - lr),
            adv,
            r.WTeamScore - r.LTeamScore,
            1.0 if r.WLoc == "H" else 0.0,
        )
    )

a = np.array(rows)
print(f"games replayed: {len(a)}")
print()
print("MATCHED-PAIRS HOME ADVANTAGE  (the sequential HCA is already inside these ratings)")
print(f"  {'elo-gap bin':<14s} {'n':>7s} {'homeW%':>8s} {'meanMgnH':>10s} {'meanMgnA':>10s}")
edges = [0, 50, 100, 150, 200, 300, 400, 10**9]
# Deliberately ragged: consecutive edges, so the tail is one shorter.
for lo, hi in zip(edges, edges[1:], strict=False):
    m = (a[:, 0] >= lo) & (a[:, 0] < hi) & (a[:, 1] != 0)
    if m.sum() < 40:
        continue
    H = m & (a[:, 1] > 0)
    A = m & (a[:, 1] < 0)
    if H.sum() < 15 or A.sum() < 15:
        continue
    label = f"{lo}-{hi}" if hi < 10**9 else f"{lo}+"
    print(
        f"  {label:<14s} {m.sum():>7d} {100 * a[m, 3].mean():>7.1f}%"
        f" {a[H, 2].mean():>+10.2f} {a[A, 2].mean():>+10.2f}"
    )

H = a[:, 1] > 0
A = a[:, 1] < 0
print()
print(f"  meanMgn(H) - meanMgnA        = {a[H, 2].mean() - a[A, 2].mean():+.2f} pts")
print(f"  home win rate, decided games    = {a[H | A, 3].mean():.4f}")
