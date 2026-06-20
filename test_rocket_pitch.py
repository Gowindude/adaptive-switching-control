"""Tests for rocket-pitch.py (Tier 1 = aero-only).

Run:  python test_rocket_pitch.py

Scope: structural properties, switching behavior, monotonicity of t_s in the
two knobs (xi, a3), the integral switching law, AND (section 6) the off-policy
RL augmentation (steps 3-5): basis dims/parity, recovery of -K_tilde_lin, the
negative cubic, usefulness (composite beats model-based) + stable-envelope
extension, the a3=0 degenerate collapse, and the OBS-11/OBS-12 guards (quartic
critic beats quadratic; large-theta data is poison). Assertions on the learned
weights are RANGE checks, not exact values (the fit has discretization noise).

No pytest dependency -- plain asserts + a PASS/FAIL summary so it runs anywhere.
"""

import importlib.util
import numpy as np

# rocket-pitch.py is hyphenated -> load it by path as module `rp`.
_spec = importlib.util.spec_from_file_location("rp", "rocket-pitch.py")
rp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rp)

dt = rp.dt
x0 = rp.x0
xi = rp.xi
x_min = rp.x_min
K = rp.K

_results = []


def check(name, ok, detail=""):
    _results.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


def t_s_seconds(a3, xi_val, T=8.0, x0_=None, xmin_=None):
    """Run the switching sim and return the switch time in seconds (or None)."""
    x0_ = x0 if x0_ is None else x0_
    xmin_ = x_min if xmin_ is None else xmin_
    *_, t_s = rp.sim_switching(x0_, a3, T, dt, K, xi_val, xmin_)
    return None if t_s is None else t_s * dt


# --- 1. Structural: the model must be the deliberately-imperfect-but-usable one ----
# Assumption 2 (u_m stabilizes the true system locally) + controllability of (A_m,B_m),
# and the whole point of the example: the airframe is open-loop UNSTABLE in pitch.
eig_open = np.linalg.eigvals(rp.Am)
check("A_m open-loop unstable (finless rocket: a positive real eigenvalue)",
      np.any(eig_open.real > 1e-9), f"eig(A_m)={eig_open}")

ctrb = np.hstack([rp.Bm, rp.Am @ rp.Bm])
check("(A_m, B_m) controllable (rank = 2)",
      np.linalg.matrix_rank(ctrb) == 2, f"rank={np.linalg.matrix_rank(ctrb)}")

eig_closed = np.linalg.eigvals(rp.Am - rp.Bm @ K)
check("model-based gain K stabilizes the model (A_m - B_m K Hurwitz)",
      np.all(eig_closed.real < 0), f"eig(A_m-B_mK)={eig_closed}")

# K_tilde_lin (small-signal validation target) should itself be a sane stabilizing
# augmentation: the composite -(K + K_tilde_lin) must keep the linearized plant stable.
eig_comp = np.linalg.eigvals(rp.Am - rp.Bm @ (K + rp.K_tilde_lin))
check("composite -(K+K_tilde_lin) stabilizes linearized plant",
      np.all(eig_comp.real < 0), f"eig={eig_comp}")


# --- 2. Switching: the two scenarios must separate cleanly (mirrors 7.2 rho=0.05 vs 1) --
t_mild = t_s_seconds(rp.a3_mild, xi)
check("mild cubic (a3_mild) -> NO switch (model-based regulates alone)",
      t_mild is None, f"t_s={t_mild}")

t_strong = t_s_seconds(rp.a3_strong, xi)
check("strong cubic (a3_strong) -> switch fires",
      t_strong is not None, f"t_s={t_strong}")

# Locked-in result: strong scenario switches early in the transient (~0.54 s).
check("strong-scenario t_s is early-transient (0.3 s < t_s < 0.8 s)",
      t_strong is not None and 0.3 < t_strong < 0.8, f"t_s={t_strong}")

# Sanity: eta(0) = 0 (true and model share x0), so no spurious t=0 switch.
arr, arr_m, eta, eta_norm, *_ = rp.sim_switching(x0, rp.a3_strong, 8.0, dt, K, xi, x_min)
check("model error eta(0) == 0 (no spurious switch at t0)",
      np.allclose(eta[0], 0.0), f"||eta(0)||={np.linalg.norm(eta[0]):.2e}")


# --- 3. Degenerate sanity: a perfect model never switches ---------------------------
# a3 = 0 -> true plant == linear model -> eta stays ~0 -> switch must never fire.
t_perfect = t_s_seconds(0.0, xi)
check("a3 = 0 (model exact) -> never switches",
      t_perfect is None, f"t_s={t_perfect}")


# --- 4. Monotonicity of t_s in the two knobs (the knobs must behave sensibly) -------
# (a) larger xi (less trust in u_m) -> earlier (smaller) switch time, monotonically.
xis = [1.0, 1.35, 2.0, 3.0, 5.0]
ts_xi = [t_s_seconds(rp.a3_strong, q) for q in xis]
mono_xi = all(a is not None and b is not None and b <= a + 1e-9
              for a, b in zip(ts_xi, ts_xi[1:]))
check("t_s decreases monotonically as xi increases (strong scenario)",
      mono_xi, f"xi={xis} -> t_s={[round(t,3) if t else None for t in ts_xi]}")

# (b) larger a3 (worse mismatch) -> earlier switch, monotonically (where it switches).
a3s = [5.0, 10.0, 20.0, 40.0, 60.0]
ts_a3 = [t_s_seconds(a, xi) for a in a3s]
mono_a3 = all(a is not None and b is not None and b <= a + 1e-9
              for a, b in zip(ts_a3, ts_a3[1:]))
check("t_s decreases monotonically as a3 increases (fixed xi)",
      mono_a3, f"a3={a3s} -> t_s={[round(t,3) if t else None for t in ts_a3]}")


# --- 5b. Integral / dwell-time switching law (OBS-10 extension) ----------------------
# tau -> 0 must recover the paper's first-crossing rule (Eq. 21).
def t_s_int(a3, xi_val, tau, T=14.0):
    t = rp.sim_switching_integral(x0, a3, T, dt, K, xi_val, tau, x_min)
    return None if t is None else t * dt

t_int0 = t_s_int(rp.a3_strong, xi, 0.0)
check("integral law with tau=0 recovers first-crossing t_s",
      t_int0 is not None and abs(t_int0 - t_strong) < 1e-9,
      f"first-crossing={t_strong}, integral(tau=0)={t_int0}")

# tau must be a monotone lever: larger budget -> later (or equal) switch.
taus = [0.0, 0.01, 0.1, 0.5, 2.0]
ts_tau = [t_s_int(rp.a3_strong, xi, tt) for tt in taus]
mono_tau = all(a is not None and b is not None and b >= a - 1e-9
               for a, b in zip(ts_tau, ts_tau[1:]))
check("t_s increases monotonically as tau (tolerated violation) increases",
      mono_tau, f"tau={taus} -> t_s={[round(t,3) if t else None for t in ts_tau]}")


# --- 5. CHARACTERIZATION (informational, not pass/fail): xi's limited pull on t_s ----
# Documents the finding in PLAN.md / RESEARCH-NOTES: xi only rescales the threshold
# (enters Eq.17 as 1/xi^2), so in the strong-mismatch regime it slides t_s within a
# short window and saturates. The mismatch (a3) is the real lever. Printed so the
# owner can eyeball it; no assertion (it's a property of the framework, not a bug).
print("\n--- characterization: t_s vs xi (strong scenario), and vs a3 (fixed xi) ---")
print("  xi   :", "  ".join(f"{q:>5}" for q in xis))
print("  t_s  :", "  ".join(f"{(t if t else float('nan')):>5.2f}" for t in ts_xi))
print(f"  -> over xi in [{xis[0]}, {xis[-1]}], t_s moves "
      f"{ts_xi[0]:.2f} -> {ts_xi[-1]:.2f} s (rescale-only, saturating).")
print("  a3   :", "  ".join(f"{a:>5}" for a in a3s))
print("  t_s  :", "  ".join(f"{(t if t else float('nan')):>5.2f}" for t in ts_a3))
print(f"  -> at fixed xi={xi}, varying a3 is the dominant lever on t_s.")


# --- 6. RL augmentation (steps 3-5): off-policy IRL learns a USEFUL u~ --------------
# Structural: basis dimensions + parity (rocket is odd-symmetric => V even, u~ odd).
xp = np.array([0.7, -0.4])
check("phi_v has 5 terms (quartic critic)", len(rp.phi_v(xp)) == 5, f"n_v={len(rp.phi_v(xp))}")
check("phi_u has 3 terms (cubic actor)", len(rp.phi_u(xp)) == 3, f"n_u={len(rp.phi_u(xp))}")
check("phi_u is ODD  (phi_u(-x) == -phi_u(x))", np.allclose(rp.phi_u(-xp), -rp.phi_u(xp)))
check("phi_v is EVEN (phi_v(-x) ==  phi_v(x))", np.allclose(rp.phi_v(-xp),  rp.phi_v(xp)))

# Deployed learning: moderate multi-IC, model-based behavior (OBS-12 recipe).
w_u, w_v, hist, pdiag, max_th = rp.learn_augmentation(rp.a3_strong)
Ktl = -rp.K_tilde_lin.flatten()
lin_err = np.linalg.norm(w_u[:2] - Ktl) / np.linalg.norm(Ktl)
check("collection stays in stabilizable region (max|theta| < 0.6)", max_th < 0.6, f"max|theta|={max_th:.3f}")
check("PI well-conditioned (cond < 1e4)", pdiag[-1]["cond"] < 1e4, f"cond={pdiag[-1]['cond']:.1e}")
check("learned linear part ~ -K_tilde_lin (<15%)", lin_err < 0.15, f"err={lin_err*100:.1f}%")
check("learned cubic NEGATIVE and substantial (-5 < w_u[2] < -2)",
      -5.0 < w_u[2] < -2.0, f"cubic={w_u[2]:.3f}")

# Usefulness: composite beats model-based in true-plant cost where the cubic bites.
for th0 in [0.4, 0.5]:
    Jmb = rp.rollout_cost(None, np.array([th0, 0.0]), rp.a3_strong)
    Jcp = rp.rollout_cost(w_u, np.array([th0, 0.0]), rp.a3_strong)
    check(f"usefulness: composite < model-based at theta0={th0}", Jcp < Jmb,
          f"J_mb={Jmb:.3f}  J_cmp={Jcp:.3f}")

# Stable-envelope extension. Three numbers: model-based, always-on composite, DEPLOYED switched.
env_mb = rp.stable_envelope(None, rp.a3_strong)
env_cp = rp.stable_envelope(w_u, rp.a3_strong)
env_sw = rp.switched_envelope(w_u, rp.a3_strong)
check("always-on composite extends the stable envelope (> +0.2 rad)", env_cp - env_mb > 0.2,
      f"mb={env_mb:.2f} -> composite={env_cp:.2f}")
check("DEPLOYED switched system extends the envelope (>= +0.15 rad vs model-based)",
      env_sw - env_mb >= 0.15, f"mb={env_mb:.2f} -> switched={env_sw:.2f}")
check("deployed switched envelope <= always-on (switch timing erodes it, not exceeds it)",
      env_sw <= env_cp + 1e-9, f"switched={env_sw:.2f}  always-on={env_cp:.2f}")

# Near origin the model is EXACT (cubic vanishes) => composite must NOT beat -Kx (OBS-1).
Jmb0 = rp.rollout_cost(None, np.array([0.2, 0.0]), rp.a3_strong)
Jcp0 = rp.rollout_cost(w_u,  np.array([0.2, 0.0]), rp.a3_strong)
check("near origin (theta0=0.2) model-based <= composite (model exact, OBS-1)",
      Jmb0 <= Jcp0 + 1e-9, f"J_mb={Jmb0:.4f}  J_cmp={Jcp0:.4f}")

# Degenerate: a3=0 => augmentation collapses (cubic ~ 0, linear ~ -K_tilde_lin).
w_u0, *_ = rp.learn_augmentation(0.0)
check("a3=0: cubic ~ 0 (|w_u[2]| < 0.5)", abs(w_u0[2]) < 0.5, f"cubic={w_u0[2]:.3f}")
check("a3=0: linear ~ -K_tilde_lin (<10%)",
      np.linalg.norm(w_u0[:2] - Ktl) / np.linalg.norm(Ktl) < 0.10,
      f"err={np.linalg.norm(w_u0[:2]-Ktl)/np.linalg.norm(Ktl)*100:.1f}%")

# OBS-12 guard: a single FAT large-theta(1.5) rollout is poison -> misses the negative cubic.
Xp, Up, _ = rp.collect_data(np.array([1.5, 0.0]), rp.a3_strong, rp.LEARN_TPE, rp.dt, rp.K, 0.3, stabilize=True)
w_u_p, _, _, _ = rp.policy_iteration([(Xp, Up)], rp.LEARN_W, rp.dt)
check("OBS-12 poison: large-theta fit misses the cubic (much less negative than deployed)",
      w_u_p[2] > w_u[2] + 1.0, f"poison cubic={w_u_p[2]:.3f} vs deployed {w_u[2]:.3f}")

# OBS-11 guard: hold the actor fixed (cubic), vary only the critic -> quartic fits V, quadratic can't.
segs = [rp.collect_data(np.array([xs, 0.0]), rp.a3_strong, rp.LEARN_TPE, rp.dt, rp.K, rp.LEARN_AMP)[:2]
        for xs in rp.LEARN_ICS]
_, _, _, dq4 = rp.policy_iteration(segs, rp.LEARN_W, rp.dt); rq4 = dq4[-1]["resid"]
_orig_phi_v = rp.phi_v; rp.phi_v = rp.phi_v_quad
_, _, _, dq2 = rp.policy_iteration(segs, rp.LEARN_W, rp.dt); rq2 = dq2[-1]["resid"]
rp.phi_v = _orig_phi_v
check("OBS-11: quartic critic residual < quadratic (same data, actor fixed)",
      rq4 < rq2, f"quartic={rq4:.2e}  quad={rq2:.2e}")


# --- summary ------------------------------------------------------------------------
n_pass = sum(ok for _, ok, _ in _results)
n_tot = len(_results)
print(f"\n==== {n_pass}/{n_tot} assertions passed ====")
if n_pass != n_tot:
    raise SystemExit(1)
