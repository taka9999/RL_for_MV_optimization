# POEMV (Regime-switching mean-variance) — from-scratch reference implementation

This repo is a small, self-contained Python implementation to **verify** the paper:
"Reinforcement learning for continuous-time mean-variance portfolio selection in a regime-switching market"
(Wu & Li, JEDC 2024).

What you get:
- RS-GBM simulator (hidden 2-state CTMC regime)
- Discretized Wonham filter using the **q-transform** to keep p in (0,1)
- POEMV actor-critic with polynomial approximation for ln f(t,p) and g(t,p)
- A training script that reproduces the simulation study setup (10y horizon, dt=1/252)

## Quick start
```bash
pip install numpy torch matplotlib
python -m poemv_rs.train --mode true_params --iters 20000
python -m poemv_rs.train --mode estimated_params --iters 20000
```

Outputs are written to `runs/`:
- `metrics.csv` (iter, omega, mean_terminal_wealth, loss_critic, loss_actor, phi*)
- `filter_demo.png` (belief tracking example)
- `learning_curves.png`

## Notes
- The PDE finite-difference benchmark is **not included** in this minimal version.
  Add it once training is stable; the paper uses it mainly as a reference surface.
- Learning rates in the paper are extremely small (e.g. alpha_theta=1e-13).
  This code defaults to the paper values, but you can override via CLI.
