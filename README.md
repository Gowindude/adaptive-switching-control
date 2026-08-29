# Adaptive Switching Control: Model-Based ↔ Model-Free

A from-scratch reproduction of the switching control framework from:

> S. Athalye, K. G. Vamvoudakis, P. J. Antsaklis, *"Synthesizing Interacting Model-Based Optimal
> Control and Model-Free Learning Approaches for Nonlinear Systems,"* International Journal of
> Robust and Nonlinear Control, 2026.

This is an independent study project, not an official implementation, and the paper itself isn't
redistributed here.

## The idea

You're controlling a system you don't fully know. You have a linear model of it, `(Aₘ, Bₘ)`, but
the real plant is `ẋ = f(x) + g(x)u` and the model is at best an approximation. The framework:

1. **Start model-based.** Run the LQR `u_m = −Kx` synthesized from the model.
2. **Watch the mismatch.** Simulate the model's own trajectory `x_m` alongside the real one under
   the same input, and track the gap `η = x − x_m`.
3. **Switch when the model stops being trustworthy.** The first time an optimality bound (Theorem 1)
   is violated, hand off from the pure model-based controller to a composite one, `u = u_m + ũ`. A
   worse model switches sooner; an accurate model may never switch at all.
4. **Learn the correction.** `ũ` isn't designed — it's learned from trajectory data with an
   off-policy integral reinforcement-learning policy iteration (Algorithm 1 in the paper).

## What's here

**Paper reproduction (§7.1–§7.2):**

| File | Reproduces |
|------|------------|
| `simple-harmonic.py` | §7.1.1 — accurate linear model, switch never fires, model-based control alone regulates the system (Fig. 2). |
| `inaccurate-simple-harmonic.py` | §7.1.2 — deliberately wrong model, switch fires early, and the learned augmentation is checked against the closed-form augmented-Riccati solution (Figs. 3–4). |
| `perturbed-oscillator.py` | §7.2 — a nonlinear oscillator, run both under mild nonlinearity (no switch) and strong nonlinearity (switch + RL), validated against the paper's known closed-form optimum (Figs. 5–7). |

**Own example — rocket pitch-attitude control:** a thrust-vector-controlled rocket is
open-loop unstable in pitch, so it's a natural testbed for the same framework, extended in stages:

| File | What it adds |
|------|--------------|
| `rocket-pitch.py` | Tier 1 — cubic aerodynamic nonlinearity the linear model can't see. Switching + RL augmentation, and a stable-envelope comparison against model-based-only control. |
| `rocket-pitch-burn.py` | Tier 2 — propellant burn makes the plant time-varying (inertia drops, so the instability coefficient grows through flight) while the model stays frozen at ignition. Looks at whether a single learned correction goes stale as the mismatch drifts. |
| `rocket-pitch-slosh.py` | Tier 3 — adds fuel slosh as two extra, unmodeled states, which breaks the framework's assumption that model and plant share the same state dimension. Sweeps the coupling strength to find where the framework's usefulness (not stability) breaks down. |

Each `rocket-pitch*.py` has a matching `test_rocket_pitch*.py` with unit tests for its dynamics,
switching logic, and RL fit.

## Requirements

```bash
pip install numpy scipy matplotlib
```

## Running

Every script is self-contained and shows its own figures:

```bash
python simple-harmonic.py
python inaccurate-simple-harmonic.py
python perturbed-oscillator.py
python rocket-pitch.py
python rocket-pitch-burn.py
python rocket-pitch-slosh.py
```

Run a test suite with, e.g.:

```bash
python test_rocket_pitch.py
```

## Implementation notes

- LQR gains come from `scipy.linalg.solve_continuous_are` (standard convention). The paper's Riccati
  carries different scaling factors, but the resulting gain matches.
- Integration is explicit forward Euler, `dt = 0.01 s`.
- Where a closed-form answer exists (the linear cases, and the near-origin behavior of the rocket
  examples), the learned augmentation is checked against it directly rather than just eyeballed.
