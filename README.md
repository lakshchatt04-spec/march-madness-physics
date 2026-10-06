# march-madness-physics

Simulate the men's NCAA tournament and come out with *probabilities*, not just
picks.

**Status: work in progress.** The core pipeline (fit -> simulate -> backtest)
runs and is tested, but the modelling and the analysis are still being
developed.

## Goal

A bracket pool rewards picking winners; a sportsbook, a seeding study or any
serious "how good is this team" question needs `P(team X wins it all)`. The
goal here is a model whose probabilities are actually *calibrated* - when it
says 70%, the team should win about 70% of the time - and a backtest rigorous
enough to tell whether that is true.

The thermodynamics framing is the organising idea: each team is a particle with
a small vector of end-of-season state variables, and tournament outcomes are
sampled from that state rather than hand-tuned.

## How it works

1. **Ingest** (`src/data_loader.py`) - read Kaggle's regular-season results and
   build each team's state: a sequential Elo rating, three-point attempt rate
   (`p3ar`), and variance of per-game offensive efficiency.
2. **Fit** (`src/calibration.py`) - a dyadic ridge regression on game margins,
   anchored to the Elo ratings, that estimates relative team strengths, the
   Elo-to-points conversion, home-court advantage, and a per-team score
   volatility `sigma_i = sigma_0 + alpha*z_p3ar + beta*z_effvar`.
3. **Match up** (`src/matchup.py`) - a Gaussian margin model gives
   `P(i beats j) = Phi((Elo_i - Elo_j + HCA) / (elo_per_point * tau_ij))`,
   where `tau_ij` combines both teams' volatilities. A latent strength offset is
   drawn once per team per tournament, so a hot or cold run spans the whole
   bracket instead of resetting every game.
4. **Simulate** (`src/bracket.py`, `src/simulate.py`) - the bracket graph comes
   from Kaggle's seeds/slots files, then Monte Carlo plays it out many times and
   aggregates title and per-round probabilities.
5. **Score** (`src/backtest.py`, `tools/backtest.py`) - walk-forward backtest:
   each evaluation season is predicted using only earlier seasons, then scored
   with log loss and Brier against their trivial baselines, plus reliability
   tables.

## Getting started

```bash
pip install -e ".[dev]"
```

Download the data from the
[Kaggle competition](https://www.kaggle.com/competitions/march-machine-learning-mania-2026)
and put the `march-machine-learning-mania-2026/` folder somewhere above the
repo root (the tools search upwards for it).

```bash
# Simulate one season's bracket
python tools/simulate_season.py --season 2025 --sims 20000

# Walk-forward backtest over several seasons
python tools/backtest.py --seasons 2021 2022 2023 2024 2025
```

Checks:

```bash
pytest          # 291 tests
ruff check .
mypy
```

## Layout

| Path | What it is |
| --- | --- |
| `src/data_loader.py` | CSV ingestion and feature engineering |
| `src/models.py` | `ParticleTeam` state and volatility parameters |
| `src/calibration.py` | dyadic fit, latent-fraction estimate |
| `src/matchup.py` | win probability and latent strength draws |
| `src/bracket.py` | bracket graph from seeds/slots |
| `src/simulate.py` | Monte Carlo tournament runner |
| `src/backtest.py` | scoring metrics and ground truth |
| `tools/` | runnable CLIs: simulate, backtest, comparisons |
| `tests/` | pytest suite |

## Where it's going

- More seasons in the backtest and tighter calibration reporting
- Feature work on volatility and the latent-strength term
- A written-up results section once the metrics settle
