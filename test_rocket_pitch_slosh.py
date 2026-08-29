"""Tests for rocket-pitch-slosh.py (Tier 3 = SLOSH, Assumption-1 violation).

Run:  python test_rocket_pitch_slosh.py

Scope:
  - 4-D dynamics correctness (decoupling at eps=0, coupling structure)
  - 2-D model structural checks (open-loop unstable, controllable)
  - K_emb stability on 4-D plant
  - K_oracle / K_fullcomp analytic gain validity
  - Switching detection: fires at moderate eps, blind at small eps
  - Oracle benefit grows with eps (knows slosh state)
  - Pitch-only composite degrades with eps (Assumption-1 violation)
  - Bellman residual grows with eps (measurable signature)
  - OBS-1 in 4-D: full-state composite can be WORSE than model-based
    (surrogate cost ignores cross-term 2*u_m*u~*R)

No pytest dependency -- plain asserts + PASS/FAIL summary.
"""

import importlib.util
import numpy as np

_spec = importlib.util.spec_from_file_location("rps", "rocket-pitch-slosh.py")
rps   = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rps)

_results = []


def check(name, ok, detail=""):
    _results.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


# 1. 2-D model structural checks (same plant as Tier-1/2)
check("Am open-loop unstable (positive real eig)",
      np.any(np.linalg.eigvals(rps.Am).real > 0),
      f"eig={np.round(np.linalg.eigvals(rps.Am).real, 3)}")

check("(Am, Bm) controllable (rank 2)",
      np.linalg.matrix_rank(np.hstack([rps.Bm, rps.Am @ rps.Bm])) == 2)

check("Am - Bm @ K Hurwitz (all eig negative)",
      np.all(np.linalg.eigvals(rps.Am - rps.Bm @ rps.K).real < 0),
      f"eig={np.round(np.linalg.eigvals(rps.Am - rps.Bm @ rps.K).real, 4)}")


# 2. 4-D dynamics structure

# At eps=0, A_4d should decouple: pitch block = [[0,1],[a1,0]], slosh independent
A0 = rps.A_4d(0.0, 3.5)
check("A_4d(eps=0): pitch-slosh decoupled -- [0,0] entry == 0 (no slosh-to-pitch)",
      np.isclose(A0[1, 2], 0.0), f"A[1,2]={A0[1,2]}")

check("A_4d(eps=0): A[1,0] == a1 (linear aero, no coupling subtraction)",
      np.isclose(A0[1, 0], rps.a1), f"A[1,0]={A0[1,0]}")

# At eps=5, coupling terms present
A5 = rps.A_4d(5.0, 3.5)
check("A_4d(eps=5): A[1,0] == a1-eps (pitch row, coupling reduces self-restoring)",
      np.isclose(A5[1, 0], rps.a1 - 5.0), f"A[1,0]={A5[1,0]:.4f}, expect {rps.a1-5.0:.4f}")

check("A_4d(eps=5): A[1,2] == eps (slosh-to-pitch coupling)",
      np.isclose(A5[1, 2], 5.0), f"A[1,2]={A5[1,2]:.4f}")

check("A_4d: slosh row A[3,0] == omega_s^2 (pitch-to-slosh coupling, always present)",
      np.isclose(A5[3, 0], 3.5**2), f"A[3,0]={A5[3,0]:.4f}, expect {3.5**2:.4f}")

check("B_4d: control enters only pitch rate row (B[1]=b, others 0)",
      np.isclose(rps.B_4d.flatten(), [0, rps.b, 0, 0]).all(),
      f"B_4d={rps.B_4d.flatten()}")

check("A_4d: slosh damping A[3,3] == -2*zeta_s*omega_s (negative, stable)",
      np.isclose(A5[3, 3], -2.0 * rps.zeta_s * 3.5), f"A[3,3]={A5[3,3]:.4f}")


# 3. K_emb structure and 4-D stability
Ke = rps.K_emb()

check("K_emb shape is (1,4)",
      Ke.shape == (1, 4), f"shape={Ke.shape}")

check("K_emb: pitch components match 2-D K",
      np.allclose(Ke.flatten()[:2], rps.K.flatten()), f"K_emb[:2]={Ke.flatten()[:2]}")

check("K_emb: slosh components are zero",
      np.allclose(Ke.flatten()[2:], 0.0), f"K_emb[2:]={Ke.flatten()[2:]}")

# K_emb stabilises 4-D plant throughout (finding: strong pitch-LQR is robust to slosh coupling)
for eps_test, omega_test in [(0.0, 3.5), (2.0, 3.5), (5.0, 3.5), (5.0, 7.0), (10.0, 1.0)]:
    A_cl = rps.A_4d(eps_test, omega_test) - rps.B_4d @ Ke
    eigs = np.linalg.eigvals(A_cl)
    check(f"K_emb stabilises 4-D plant at eps={eps_test}, omega_s={omega_test}",
          np.all(eigs.real < 0),
          f"max_Re={np.max(eigs.real):.4f}")

# Spot-check: 2-D grid confirms K_emb stability (faster than full grid in tests)
_ke = rps.K_emb()
_grid_stable = True
for _eps_g in [0.0, 3.0, 6.0, 10.0, 12.0]:
    for _oms_g in [0.5, 3.5, 7.0, 12.0, 15.0]:
        _A_g  = rps.A_4d(_eps_g, _oms_g)
        _egs  = np.linalg.eigvals(_A_g - rps.B_4d @ _ke)
        if np.max(_egs.real) > 0:
            _grid_stable = False
            break
check("K_emb stability: 2-D grid spot-check (25 eps x omega_s combos, all stable)",
      _grid_stable, "K_emb robust to slosh coupling in tested range")


# 4. K_oracle validity (should improve on K_emb for eps > 0)
Ko3 = rps.K_oracle(3.0, 3.5)
check("K_oracle(eps=3) not None (ARE solves)",
      Ko3 is not None)

if Ko3 is not None:
    A_or = rps.A_4d(3.0, 3.5)
    eigs_or = np.linalg.eigvals(A_or - rps.B_4d @ Ko3)
    check("K_oracle(eps=3) stabilises 4-D plant",
          np.all(eigs_or.real < 0), f"max_Re={np.max(eigs_or.real):.4f}")


# 5. Switching behaviour: detection threshold
# At eps=0: model exact for pitch -> switch should NOT fire within 8 s
_arr0, _u0, _eta0, _thr0, _ts0 = rps.sim_slosh_framework(
    rps.x0_4, 0.0, 3.5, w_u=np.zeros(2))
check("Switch does NOT fire at eps=0 (model exact for pitch, no slosh coupling)",
      _ts0 is None, f"t_s={_ts0}")

# At eps=0.1: small coupling -> switch should NOT fire (pitch mismatch below threshold)
_arr01, _u01, _eta01, _thr01, _ts01 = rps.sim_slosh_framework(
    rps.x0_4, 0.1, 3.5, w_u=np.zeros(2))
check("Switch does NOT fire at eps=0.1 (coupling too small for pitch eta to cross threshold)",
      _ts01 is None, f"t_s={_ts01}")

# At eps=2.0: moderate coupling -> switch SHOULD fire
_arr2, _u2, _eta2, _thr2, _ts2 = rps.sim_slosh_framework(
    rps.x0_4, 2.0, 3.5, w_u=np.zeros(2))
check("Switch FIRES at eps=2.0 (slosh coupling creates pitch mismatch)",
      _ts2 is not None, f"t_s={None if _ts2 is None else _ts2*rps.dt:.2f} s")

# Switch time should decrease as eps grows (bigger mismatch -> earlier switch)
_ts_1 = rps.sim_slosh_framework(rps.x0_4, 1.0, 3.5, w_u=np.zeros(2))[4]
_ts_5 = rps.sim_slosh_framework(rps.x0_4, 5.0, 3.5, w_u=np.zeros(2))[4]
check("Switch fires earlier at eps=5 than at eps=1 (larger mismatch -> earlier switch)",
      _ts_1 is not None and _ts_5 is not None and _ts_5 < _ts_1,
      f"t_s(1)={None if _ts_1 is None else _ts_1*rps.dt:.2f},"
      f" t_s(5)={None if _ts_5 is None else _ts_5*rps.dt:.2f}")


# 6. Oracle always improves on model-based at moderate/large eps
for eps_test in [1.0, 3.0, 7.0]:
    Ke  = rps.K_emb()
    Ko  = rps.K_oracle(eps_test, 3.5)
    if Ko is not None:
        arr_mb, u_mb = rps.sim_4d_fixed(rps.x0_4, eps_test, 3.5, Ke)
        arr_or, u_or = rps.sim_4d_fixed(rps.x0_4, eps_test, 3.5, Ko)
        J_mb = rps.pitch_cost(arr_mb, u_mb)
        J_or = rps.pitch_cost(arr_or, u_or)
        check(f"Oracle beats model-based at eps={eps_test}",
              J_or < J_mb, f"J_mb={J_mb:.4f}, J_or={J_or:.4f}")


# 7. Pitch-only composite degrades with eps (Assumption-1 violation)
# Learn pitch composite at two eps values and compare benefit%
def _pitch_comp_benefit(eps_val, omega_s=3.5):
    wu, diag = rps.learn_augmentation_slosh(eps_val, omega_s)
    if wu is None:
        return float("nan"), None
    Ke = rps.K_emb()
    arr_mb, u_mb = rps.sim_4d_fixed(rps.x0_4, eps_val, omega_s, Ke)
    arr_pc, u_pc, *_ = rps.sim_slosh_framework(rps.x0_4, eps_val, omega_s, w_u=wu)
    J_mb = rps.pitch_cost(arr_mb, u_mb)
    J_pc = rps.pitch_cost(arr_pc, u_pc)
    if J_mb > 1e5 or J_pc > 1e5:
        return float("nan"), diag
    return (J_mb - J_pc) / J_mb * 100.0, diag

ben_lo, diag_lo = _pitch_comp_benefit(1.0)
ben_hi, diag_hi = _pitch_comp_benefit(7.0)

check("Pitch-only composite benefit DEGRADES as eps grows (core Assumption-1 finding)",
      (not np.isnan(ben_lo) and not np.isnan(ben_hi) and ben_lo > ben_hi),
      f"ben(eps=1)={ben_lo:.2f}%, ben(eps=7)={ben_hi:.2f}%")

check("Pitch-only composite benefit is NEGATIVE at large eps (actively harmful)",
      not np.isnan(ben_hi) and ben_hi < 0.0,
      f"ben(eps=7)={ben_hi:.2f}%")


# 8. Bellman residual grows with eps (Assumption-1 signature)
resids = []
for eps_test in [0.0, 1.0, 5.0]:
    _, diag = rps.learn_augmentation_slosh(eps_test, 3.5)
    resids.append((eps_test, diag[-1]["resid"] if diag else None))

r0, r1, r5 = resids[0][1], resids[1][1], resids[2][1]
check("Bellman residual at eps=0 < eps=1 (residual grows with coupling)",
      r0 is not None and r1 is not None and r1 > r0,
      f"resid(0)={r0:.2e}, resid(1)={r1:.2e}")

check("Bellman residual at eps=1 < eps=5 (residual keeps growing)",
      r1 is not None and r5 is not None and r5 > r1,
      f"resid(1)={r1:.2e}, resid(5)={r5:.2e}")


# 9. OBS-1 in 4-D: full-state composite can be WORSE than model-based
#    (surrogate cost ignores cross-term 2*u_m*u~*R)
Ke  = rps.K_emb()
Kfc = rps.K_fullcomp(0.0, 3.5)   # at eps=0 OBS-1 effect is most visible
if Kfc is not None:
    # Slosh components of K_tilde_4d should be ~0 at eps=0:
    # Q_4d has zero slosh penalty and slosh doesn't couple back to pitch at eps=0,
    # so controlling slosh has no pitch cost benefit.
    Kfc_slosh = Kfc.flatten()[2:]
    check("K_fullcomp slosh components ~= 0 at eps=0 (no slosh penalty in Q_4d, decoupled)",
          np.allclose(Kfc_slosh, 0.0, atol=1e-10),
          f"K_fc[2:]={Kfc_slosh}")

    arr_mb0, u_mb0 = rps.sim_4d_fixed(rps.x0_4, 0.0, 3.5, Ke)
    arr_fc0, u_fc0 = rps.sim_4d_fixed(rps.x0_4, 0.0, 3.5, Kfc)
    J_mb0 = rps.pitch_cost(arr_mb0, u_mb0)
    J_fc0 = rps.pitch_cost(arr_fc0, u_fc0)
    check("Full-state composite is WORSE than model-based at eps=0 (OBS-1: surrogate cost)",
          J_fc0 > J_mb0, f"J_mb={J_mb0:.4f}, J_fc={J_fc0:.4f}")


# 10. pitch_cost: diverged trajectory returns 1e6 sentinel
nan_arr = np.full((100, 4), np.nan)
nan_u   = np.zeros(100)
check("pitch_cost returns 1e6 for diverged (NaN) trajectory",
      rps.pitch_cost(nan_arr, nan_u) == 1e6)

check("pitch_cost = 0 for zero trajectory and zero control",
      rps.pitch_cost(np.zeros((50, 4)), np.zeros(50)) == 0.0)


# 11. phi_v_slosh and phi_u_slosh: parity checks
x_p = np.array([0.4, -0.1])
check("phi_v_slosh is EVEN: phi_v(-x) == phi_v(x)",
      np.allclose(rps.phi_v_slosh(-x_p), rps.phi_v_slosh(x_p)))

check("phi_u_slosh is ODD: phi_u(-x) == -phi_u(x)",
      np.allclose(rps.phi_u_slosh(-x_p), -rps.phi_u_slosh(x_p)))


# 12. Model-based pitch cost grows with eps (slosh coupling raises the cost)
Ke = rps.K_emb()
arr_lo, u_lo = rps.sim_4d_fixed(rps.x0_4, 0.0, 3.5, Ke)
arr_hi, u_hi = rps.sim_4d_fixed(rps.x0_4, 5.0, 3.5, Ke)
J_lo = rps.pitch_cost(arr_lo, u_lo)
J_hi = rps.pitch_cost(arr_hi, u_hi)
check("Model-based pitch cost grows with eps (slosh makes regulation harder)",
      J_hi > J_lo,
      f"J_mb: eps=0 -> {J_lo:.4f}, eps=5 -> {J_hi:.4f}")


# Summary
n_pass = sum(1 for _, ok, _ in _results if ok)
n_fail = sum(1 for _, ok, _ in _results if not ok)
print(f"\n{'='*60}")
print(f"Results: {n_pass}/{len(_results)} PASSED, {n_fail} FAILED")
if n_fail > 0:
    print("FAILED tests:")
    for name, ok, detail in _results:
        if not ok:
            print(f"  - {name}  ({detail})")
else:
    print("ALL TESTS PASSED")
print("="*60)
