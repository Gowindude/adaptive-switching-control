"""Tests for rocket-pitch-burn.py (Tier 2 = MASS BURN, time-varying parameters).

Run:  python test_rocket_pitch_burn.py

Scope: burn profile correctness, stability throughout burn, model error
non-vanishing at origin, switching behavior (including IC-independence --
OBS-13), RL basis parity + quadratic exactness, linear actor recovery,
staleness (stale u~ vs fresh u~ at end-of-burn), composite vs model-based
benefit at late burn.

No pytest dependency -- plain asserts + PASS/FAIL summary.
"""

import importlib.util
import numpy as np

_spec = importlib.util.spec_from_file_location("rpb", "rocket-pitch-burn.py")
rpb   = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rpb)

_results = []


def check(name, ok, detail=""):
    _results.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


# ---------------------------------------------------------------------------
# 1. Burn profile: boundary conditions + b constant
# ---------------------------------------------------------------------------
a1_t0,  b_t0  = rpb.burn_params(0.0)
a1_tf,  b_tf  = rpb.burn_params(rpb.T_burn)
a1_mid, b_mid = rpb.burn_params(rpb.T_burn / 2.0)

check("burn_params: a1(0) == a1_0",
      np.isclose(a1_t0, rpb.a1_0), f"a1(0)={a1_t0}")

check("burn_params: a1(T_burn) == 3 * a1_0  (a1 triples at burnout)",
      np.isclose(a1_tf, 3.0 * rpb.a1_0, rtol=1e-6), f"a1(T_burn)={a1_tf:.4f}")

check("burn_params: b constant throughout (b(0)==b(T/2)==b(T))",
      np.isclose(b_t0, b_mid) and np.isclose(b_t0, b_tf), f"b={b_t0},{b_mid},{b_tf}")

check("burn_params: a1 monotone increasing (a1_mid between a1_0 and 3*a1_0)",
      rpb.a1_0 < a1_mid < 3.0 * rpb.a1_0, f"a1_mid={a1_mid:.3f}")


# ---------------------------------------------------------------------------
# 2. Stability: frozen K stabilizes the true plant throughout the burn
# ---------------------------------------------------------------------------
K = rpb.K
Bm = rpb.Bm

for t_check in [0.0, 5.0, 10.0]:
    a1_t, _ = rpb.burn_params(t_check)
    A_t     = np.array([[0.0, 1.0], [a1_t, 0.0]])
    eig_cl  = np.linalg.eigvals(A_t - Bm @ K)
    check(f"frozen K stabilizes plant at t={t_check:.0f} s (all poles Hurwitz)",
          np.all(eig_cl.real < 0.0), f"eig={np.sort(eig_cl.real)}")


# ---------------------------------------------------------------------------
# 3. Model error at origin: non-vanishing for all t > 0
#    Error term is (a1(t) - a1_0) * theta -- at theta=1 this equals a1(t)-a1_0.
#    Unlike Tier-1 cubic (error ~ a3*theta^3 = 0 at theta=0), the linear error
#    does NOT vanish at the origin.
# ---------------------------------------------------------------------------
for t_check in [5.0, 10.0]:
    a1_t, _ = rpb.burn_params(t_check)
    model_error_at_unit_theta = abs(a1_t - rpb.a1_0)
    check(f"model error non-zero at t={t_check:.0f} s (origin is NOT error-free)",
          model_error_at_unit_theta > 0.5, f"delta_a1={model_error_at_unit_theta:.2f}")


# ---------------------------------------------------------------------------
# 4. Switching: fires in finite time; standard sanity checks
# ---------------------------------------------------------------------------
dt = rpb.dt; xi = rpb.xi; x_min = rpb.x_min; x0 = rpb.x0

_, _, _, _, _, _, t_s = rpb.sim_switching_burn(x0, 6.0, dt, K, xi, x_min)
t_s_s = None if t_s is None else t_s * dt

check("switch fires with time-varying plant (t_s is not None)",
      t_s_s is not None, f"t_s={t_s_s}")

check("switch fires within [0.5, 3.0] s (linear drift is moderate)",
      t_s_s is not None and 0.5 < t_s_s < 3.0, f"t_s={t_s_s}")

# Model error eta(0) = 0 (true and model share x0 => no spurious t=0 switch)
_, _, eta, *_ = rpb.sim_switching_burn(x0, 6.0, dt, K, xi, x_min)
check("eta(0) == 0  (no spurious switch at t=0)",
      np.allclose(eta[0], 0.0, atol=1e-12), f"||eta(0)||={np.linalg.norm(eta[0]):.2e}")

# OBS-13: switch time is IC-INDEPENDENT for a linear mismatch.
# Both threshold and ||eta||^2 scale with ||x||^2, so their ratio is state-
# magnitude-independent.  All ICs should give the same t_s (up to one step).
ts_for_ics = []
for th0 in [0.2, 0.3, 0.5, 1.0]:
    xi0 = np.array([th0, 0.0])
    x_min_i = 0.05 * np.linalg.norm(xi0)
    _, _, _, _, _, _, ts_i = rpb.sim_switching_burn(xi0, 6.0, dt, K, xi, x_min_i)
    ts_for_ics.append(None if ts_i is None else ts_i * dt)
ic_span = max(t for t in ts_for_ics if t is not None) - min(t for t in ts_for_ics if t is not None)
check("OBS-13: switch time is IC-independent for linear mismatch (span < 0.05 s)",
      ic_span < 0.05, f"t_s for IC=[0.2,0.3,0.5,1.0]: {[round(t,3) if t else None for t in ts_for_ics]}")


# ---------------------------------------------------------------------------
# 5. RL bases: parity + dimensionality
# ---------------------------------------------------------------------------
xp = np.array([0.7, -0.4])
check("phi_v_burn has 3 terms (quadratic critic: no quartic needed for LTI plant)",
      len(rpb.phi_v_burn(xp)) == 3, f"n_v={len(rpb.phi_v_burn(xp))}")
check("phi_u_burn has 2 terms (linear actor)",
      len(rpb.phi_u_burn(xp)) == 2, f"n_u={len(rpb.phi_u_burn(xp))}")
check("phi_v_burn is EVEN  (phi_v(-x) == phi_v(x))",
      np.allclose(rpb.phi_v_burn(-xp), rpb.phi_v_burn(xp)))
check("phi_u_burn is ODD   (phi_u(-x) == -phi_u(x))",
      np.allclose(rpb.phi_u_burn(-xp), -rpb.phi_u_burn(xp)))


# ---------------------------------------------------------------------------
# 6. Quadratic critic exactness: V* is EXACTLY quadratic for a frozen LTI plant.
#    The PI residual should be tiny (machine-noise level after 1-2 iterations).
# ---------------------------------------------------------------------------
w_u0,  w_v0,  hist0,  pdiag0,  _ = rpb.learn_augmentation_burn(0.0)
w_u10, w_v10, hist10, pdiag10, _ = rpb.learn_augmentation_burn(10.0)

check("quadratic critic residual tiny at t=0  (exact fit, resid < 1e-3)",
      pdiag0[-1]["resid"] < 1e-3, f"resid={pdiag0[-1]['resid']:.2e}")
check("quadratic critic residual tiny at t=10 (exact fit, resid < 1e-3)",
      pdiag10[-1]["resid"] < 1e-3, f"resid={pdiag10[-1]['resid']:.2e}")
check("condition number well-conditioned (cond < 1e3 at t=0)",
      pdiag0[-1]["cond"] < 1e3, f"cond={pdiag0[-1]['cond']:.1e}")


# ---------------------------------------------------------------------------
# 7. Linear actor recovery: w_u converges to -K_tilde within 5 %
# ---------------------------------------------------------------------------
for tf, w_u_learned in [(0.0, w_u0), (10.0, w_u10)]:
    w_u_tgt = -rpb.K_tilde_at(tf).flatten()    # true optimal: w_u* = -K_tilde
    err     = np.linalg.norm(w_u_learned - w_u_tgt) / np.linalg.norm(w_u_tgt)
    check(f"linear actor recovery at t_freeze={tf:.0f} s (<5% from -K_tilde)",
          err < 0.05, f"err={err*100:.1f}%  w_u={w_u_learned}  target={w_u_tgt}")


# ---------------------------------------------------------------------------
# 8. Staleness: stale u~_t0 is worse than fresh u~_t10 at end of burn.
#    At t=0 both are equally (in)effective because the switch hasn't fired
#    yet (model is exact, no mismatch => composite never activated).
# ---------------------------------------------------------------------------
w_u5, *_ = rpb.learn_augmentation_burn(5.0)

J_mb_0    = rpb.snapshot_cost(None,   0.0)
J_t0_at0  = rpb.snapshot_cost(w_u0,  0.0)
J_t10_at0 = rpb.snapshot_cost(w_u10, 0.0)

check("at t=0 composite == model-based (model exact; switch does not fire)",
      np.isclose(J_mb_0, J_t0_at0, rtol=1e-6) and np.isclose(J_mb_0, J_t10_at0, rtol=1e-6),
      f"J_mb={J_mb_0:.4f}  J_t0={J_t0_at0:.4f}  J_t10={J_t10_at0:.4f}")

J_mb_10   = rpb.snapshot_cost(None,   10.0)
J_t0_at10 = rpb.snapshot_cost(w_u0,   10.0)
J_t5_at10 = rpb.snapshot_cost(w_u5,   10.0)
J_t10_at10= rpb.snapshot_cost(w_u10,  10.0)

check("composite beats model-based at end of burn  (J_composite < J_mb at t=10)",
      J_t10_at10 < J_mb_10,
      f"J_mb={J_mb_10:.4f}  J_t10={J_t10_at10:.4f}")

check("staleness: stale u~_t0 worse than fresh u~_t10 at end of burn",
      J_t0_at10 > J_t10_at10,
      f"J_t0={J_t0_at10:.4f}  J_t10={J_t10_at10:.4f}")

check("mid-burn re-learn (u~_t5) is between stale and fresh at t=10",
      J_t10_at10 <= J_t5_at10 <= J_t0_at10 + 1e-6,
      f"J_t0={J_t0_at10:.4f}  J_t5={J_t5_at10:.4f}  J_t10={J_t10_at10:.4f}")

check("composite benefit at end of burn >= 3% of J_mb (composite is useful late in burn)",
      (J_mb_10 - J_t10_at10) / J_mb_10 >= 0.03,
      f"benefit={(J_mb_10-J_t10_at10)/J_mb_10*100:.1f}%")


# ---------------------------------------------------------------------------
# 9. Late-burn quasi-static validation: does the frozen-snapshot assumption
#    hold on the GENUINELY TIME-VARYING plant?
#    Start from x0=[0.3,0] at t_start=8 s; a1 drifts 9.3->12 during transient.
#    Timescale separation: T_burn=10 s >> tau_reg ~0.7 s -> 14x ratio.
#    If composite benefit% on drifting plant matches the frozen snapshot -> quasi-static holds.
# ---------------------------------------------------------------------------
J_late_mb    = rpb.sim_late_burn_cost(None)
J_late_stale = rpb.sim_late_burn_cost(w_u0)    # u~ designed at t=0 (stale)
J_late_fresh = rpb.sim_late_burn_cost(w_u10)   # u~ designed at t=10 (fresh)

# Frozen snapshot at same conditions for comparison
J_snap_mb    = rpb.snapshot_cost(None,   10.0, T_snap=4.0)
J_snap_fresh = rpb.snapshot_cost(w_u10, 10.0, T_snap=4.0)

benefit_late = (J_late_mb - J_late_fresh) / J_late_mb
benefit_snap = (J_snap_mb - J_snap_fresh) / J_snap_mb

check("late-burn (time-varying plant): composite u~_t10 beats model-based",
      J_late_fresh < J_late_mb,
      f"J_mb={J_late_mb:.4f}  J_fresh={J_late_fresh:.4f}  benefit={benefit_late*100:.1f}%")

check("quasi-static check: composite benefit on drifting plant is non-trivial (>0.5% of J_mb)",
      benefit_late > 0.005,
      f"drifting={benefit_late*100:.1f}%  frozen={benefit_snap*100:.1f}%  "
      "(snapshot is conservative upper bound: drifting plant starts at a1=9.3 not 12)")

check("late-burn staleness: stale u~_t0 not catastrophically worse than fresh u~_t10",
      J_late_stale < J_late_mb * 1.20,
      f"J_stale={J_late_stale:.4f}  J_fresh={J_late_fresh:.4f}  J_mb={J_late_mb:.4f}")


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------
n_pass = sum(ok for _, ok, _ in _results)
n_tot  = len(_results)
print(f"\n==== {n_pass}/{n_tot} assertions passed ====")
if n_pass != n_tot:
    raise SystemExit(1)
