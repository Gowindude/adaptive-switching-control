# Adaptive Switching Control: Model-Based ↔ Model-Free

A Python implementation of the optimal-control framework that **starts with a model-based controller
and switches to a learned composite controller** when the model proves too inaccurate — reproducing
the simulation studies from:

> S. Athalye, K. G. Vamvoudakis, P. J. Antsaklis, *"Synthesizing Interacting Model-Based Optimal
> Control and Model-Free Learning Approaches for Nonlinear Systems,"* International Journal of Robust
> and Nonlinear Control, 2026.

This is an independent study/reproduction project (the paper itself is not redistributed here).

## The framework in brief

For an unknown nonlinear system `ẋ = f(x) + g(x)u` where only an imperfect **linear** model
`(Aₘ, Bₘ)` is available:

1. **Model-based control** — start with the LQR `u_m = −Kx` synthesized from the linear model.
2. **Monitor the mismatch** — run `u_m` on the true system while simulating the model trajectory
   `x_m` under the same input, and track `η = x − x_m`.
3. **Switch** — the first instant an optimality-based threshold (Theorem 1) is violated, switch from
   the purely model-based controller to a **composite** controller `u = u_m + ũ`. A worse model
   switches sooner; an accurate model never switches.
4. **Learn the augmentation** — `ũ*` is obtained model-free via an **off-policy integral
   reinforcement-learning** policy iteration (Algorithm 1) from trajectory data.

## Contents

| File | What it does |
|------|--------------|
| `simple-harmonic.py` | §7.1.1 — 2-DOF mass–spring–damper with an **accurate** linear model: the switch never fires and the model-based controller regulates alone (reproduces Figure 2). |
| `inaccurate-simple-harmonic.py` | §7.1.2 — same system with an **inaccurate** (sign-flipped) model: the switch fires early, and the off-policy RL learns the optimal augmentation, validated against the analytic augmented-Riccati solution (reproduces Figures 3 & 4). |
| `perturbed-oscillator.py` | §7.2 — a **nonlinear** perturbed harmonic oscillator: mild nonlinearity (no switch) and stronger nonlinearity (switch + RL), with weights validated against the paper's closed-form optimum (reproduces Figures 5–7). |
| `rocket-pitch.py` | Work in progress — applying the full framework to a rocket pitch-attitude (thrust-vectoring) example with aerodynamic nonlinearity. |

## Requirements

- Python 3.x
- `numpy`, `scipy`, `matplotlib`

```bash
pip install numpy scipy matplotlib
```

## Running

Each script is self-contained and pops up its figures:

```bash
python simple-harmonic.py
python inaccurate-simple-harmonic.py
python perturbed-oscillator.py
```

## Notes

- Control design uses `scipy.linalg.solve_continuous_are` (standard LQR convention); the gain matches
  the paper's despite its different Riccati scaling.
- Integration is explicit forward Euler (`dt = 0.01 s`).
- The model-free augmentation is validated against an analytic ground truth in the linear/known-form
  cases (e.g. the RL gain converges to the augmented-Riccati solution to within a few percent).
