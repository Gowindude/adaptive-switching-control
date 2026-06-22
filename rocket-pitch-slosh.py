"""rocket-pitch-slosh.py -- Tier 3: SLOSH (breaks Assumption 1)

Assumption 1 (paper §2, exact verbatim):
    "The dimensions of the linear model and the system are the same,
    i.e., A_m ∈ R^{n×n}, B_m ∈ R^{n×m}."

THIS TIER VIOLATES ASSUMPTION 1 BY DESIGN.
True plant: 4-D (pitch + slosh).  Model: 2-D (pitch only).
A_m ∈ R^{2×2}  but  A_true ∈ R^{4×4}.

Goal: NOT to make it work -- to MAP THE VALIDITY BOUNDARY of the framework
experimentally via a coupling-strength (eps) sweep.

Physics
-------
Fuel slosh is modelled as a harmonic pendulum coupled to pitch (standard
liquid-propellant slosh model; see Sidi "Spacecraft Dynamics and Control"):

  True plant (4-D):
    theta_ddot = a1*theta + b*u + eps*(psi - theta)   [pitch + coupling]
    psi_ddot   = omega_s^2*(theta - psi)              [slosh restoring]
               - 2*zeta_s*omega_s*psi_dot             [small damping, keeps ARE well-posed]
    states: x4 = [theta, theta_dot, psi, psi_dot]

  Model (2-D, omits slosh entirely):
    theta_ddot = a1*theta + b*u
    states: x_m = [theta, theta_dot]

Compromise (Assumption-1 violation workaround):
  eta_pitch = [theta, theta_dot] - x_m   (project onto shared pitch states)
  Switching threshold uses eta_pitch and x_pitch only.
  When slosh is active, eta_pitch UNDERESTIMATES total mismatch.

Four controllers (baseline comparison set):
  1. Model-based: u = -K @ x_pitch  (embedded 2-D gain K_emb into 4-D)
  2. Pitch-only composite (RL): u = -K @ x_pitch + w_u . phi_u(x_pitch)
     applied after switch fires.  Basis: linear [theta, thetadot].
     RL learns from 4-D plant data but uses only pitch features.
  3. Full-state composite (analytical): u = -(K_emb + K_tilde_4d) @ x4
     Solve augmented ARE in 4-D -- needs full slosh state measurable.
     Upper bound on what augmentation can achieve.
  4. Oracle 4-D LQR: u = -K_oracle @ x4
     Optimal full-state controller knowing true 4-D plant.
     Ceiling: maximum achievable pitch cost reduction.

Cost metric: PITCH cost = integral(x_pitch' Q x_pitch + u^2 R) dt.
Slosh amplitude = |psi - theta| shown separately (diagnostic only).

zeta_s = 0.02 (small damping) is physically standard for liquid-propellant
slosh.  Needed to make A_4d have no imaginary-axis eigenvalues so that the
4-D ARE is well-posed across the entire eps sweep.

Key findings (OBS-14+):
  * Switch fires when eps is large enough for eta_pitch to cross threshold.
    For small eps (weak coupling), switch may NOT fire -- the pitch-projected
    mismatch stays below threshold even though total 4-D mismatch exists.
    This is the Assumption-1 signature: the monitoring metric is blind to slosh.
  * Benefit% transition: pitch-only composite helps at small eps (slosh acts as
    a small pitch disturbance, learnable indirectly); breaks at large eps or
    near resonance (omega_s near control BW) where slosh is uncontrollable
    from pitch state alone.
  * Bellman residual of pitch-only RL DEGRADES with eps: as slosh grows,
    the pitch-state value function is no longer well-defined (same pitch state,
    different slosh state => different cost-to-go). Residual degradation is a
    measurable signature of the broken assumption.
  * Eigenvalue crossing: compute eig(A_4d - B_4d @ K_emb) vs eps to find the
    analytical prediction of where K_emb loses 4-D stability. Overlaid on
    benefit% curve as a cross-check.
"""

import numpy as np
from scipy.linalg import solve_continuous_are
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Physical parameters (same pitch plant as Tier-1 and Tier-2)
# ---------------------------------------------------------------------------
a1     = 4.0    # pitch instability coefficient [1/s^2]
b      = 8.0    # TVC authority [1/s^2 per rad gimbal]
zeta_s = 0.02   # small slosh damping ratio (physically standard; keeps ARE well-posed)

# Aero cubic absent (a3=0): isolates slosh effect cleanly vs Tier-1.

# LQR weights (same as Tier-1/2)
Q = np.diag([1.0, 0.1])    # penalise pitch angle, modest rate
R = np.array([[0.2]])       # control effort

# Default sweep / sim settings
T         = 8.0    # simulation horizon [s]
dt        = 0.01   # forward-Euler step [s]
xi        = 1.35   # suboptimality slack in switching threshold (Eq. 17)

OMEGA_S_DEFAULT = 3.5   # slosh natural freq [rad/s] -- near control BW (~3.5-5 rad/s)
EPS_SWEEP = [0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0]

x0_4    = np.array([0.3, 0.0, 0.0, 0.0])   # pitch kick, no initial slosh
x_min_4 = 0.05 * float(np.linalg.norm(x0_4[:2]))   # switching guard on pitch norm


# ---------------------------------------------------------------------------
# 2-D model (same as Tier-1: linear aero only, no cubic, no slosh)
# ---------------------------------------------------------------------------
Am = np.array([[0.0, 1.0],
               [a1,  0.0]])
Bm = np.array([[0.0], [b]])

Pm = solve_continuous_are(Am, Bm, Q, R)
K  = np.linalg.inv(R) @ Bm.T @ Pm     # shape (1, 2) -- model-based gain

# Quick structural checks (run at import time)
_eig_open   = np.linalg.eigvals(Am)
_ctrb       = np.hstack([Bm, Am @ Bm])
_eig_closed = np.linalg.eigvals(Am - Bm @ K)
assert np.any(_eig_open.real > 0), "model should be open-loop unstable"
assert np.linalg.matrix_rank(_ctrb) == 2, "model should be controllable"
assert np.all(_eig_closed.real < 0), "model-based closed loop should be stable"


# ---------------------------------------------------------------------------
# 4-D true plant (pitch + slosh)
# ---------------------------------------------------------------------------
B_4d = np.array([[0.0], [b], [0.0], [0.0]])    # control only enters pitch


def A_4d(eps, omega_s):
    """4-D state matrix as a function of coupling strength eps and slosh freq omega_s."""
    return np.array([
        [0.0,          1.0,  0.0,          0.0                   ],
        [a1 - eps,     0.0,  eps,          0.0                   ],
        [0.0,          0.0,  0.0,          1.0                   ],
        [omega_s**2,   0.0, -omega_s**2,  -2.0 * zeta_s * omega_s]
    ])


def f_slosh(x4, eps, omega_s):
    """4-D drift (no control input). x4 = [theta, thetadot, psi, psidot]."""
    th, thd, ps, psd = float(x4[0]), float(x4[1]), float(x4[2]), float(x4[3])
    return np.array([
        thd,
        (a1 - eps) * th + eps * ps,
        psd,
        omega_s**2 * (th - ps) - 2.0 * zeta_s * omega_s * psd
    ])


# ---------------------------------------------------------------------------
# Analytical gains
# ---------------------------------------------------------------------------

# Q extended to 4-D: penalise pitch only (consistent with 2-D objective).
# Detectability holds because slosh modes are stable (zeta_s > 0).
Q_4d = np.block([[Q,               np.zeros((2, 2))],
                 [np.zeros((2, 2)), np.zeros((2, 2))]])


def K_emb():
    """Pitch gain K embedded into 4-D by padding with zeros.  Shape (1,4).
    u = -K_emb @ x4 = -K @ x_pitch.  Ignores slosh states entirely."""
    return np.concatenate([K.flatten(), [0.0, 0.0]]).reshape(1, 4)


def K_oracle(eps, omega_s):
    """Oracle: optimal 4-D LQR gain (knows true plant).  Returns (1,4) or None."""
    try:
        A = A_4d(eps, omega_s)
        P = solve_continuous_are(A, B_4d, Q_4d, R)
        return np.linalg.inv(R) @ B_4d.T @ P
    except Exception:
        return None


def K_fullcomp(eps, omega_s):
    """Full-state composite (analytical upper bound on augmentation potential).
    Solves the augmented ARE in 4-D with augmented plant A_4d - B_4d @ K_emb.
    Returns total gain (K_emb + K_tilde_4d) as (1,4), or None on failure."""
    try:
        A     = A_4d(eps, omega_s)
        Ke    = K_emb()
        A_aug = A - B_4d @ Ke
        P_t   = solve_continuous_are(A_aug, B_4d, Q_4d, R)
        K_t   = np.linalg.inv(R) @ B_4d.T @ P_t
        return Ke + K_t      # shape (1, 4)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Simulation functions
# ---------------------------------------------------------------------------

def sim_4d_fixed(x0_4, eps, omega_s, K_4d, T=T, dt=dt):
    """4-D plant under a fixed 1×4 gain throughout (no switching).
    Used for oracle and full-state composite baselines.
    Returns (arr4, u_arr) where arr4[:,2:] = slosh states."""
    N    = int(T / dt)
    arr4 = np.zeros((N, 4)); arr4[0] = x0_4
    u_arr = np.zeros(N)
    x4   = x0_4.astype(float).copy()

    for i in range(1, N):
        u = float(-(K_4d @ x4).item())
        u_arr[i] = u
        x4 = x4 + (f_slosh(x4, eps, omega_s) + B_4d.flatten() * u) * dt
        if not np.all(np.isfinite(x4)) or float(np.linalg.norm(x4)) > 1e3:
            arr4[i:] = np.nan
            u_arr[i:] = np.nan
            break
        arr4[i] = x4
    return arr4, u_arr


def sim_slosh_framework(x0_4, eps, omega_s, w_u=None, T=T, dt=dt,
                        xi=xi, x_min=x_min_4):
    """Full switching-framework simulation on the 4-D plant.

    Phase 1 (pre-switch): model-based u = -K @ x_pitch.
    Switch fires (Theorem 1 / Eq. 21) when ||eta_pitch||^2 >= threshold
    AND ||x_pitch|| > x_min.  eta_pitch = x_pitch - x_m (2-D monitor).

    Phase 2 (post-switch, only if w_u is not None):
    Pitch-only composite u = -K @ x_pitch + w_u . phi_u(x_pitch).

    w_u=None: model-based throughout (no composite; used to measure J_mb).

    Returns (arr4, u_arr, eta_psq, thr_arr, t_s_step)
    where t_s_step is the switch step index (None if no switch).
    """
    N       = int(T / dt)
    arr4    = np.zeros((N, 4)); arr4[0] = x0_4
    u_arr   = np.zeros(N)
    eta_psq = np.zeros(N)
    thr_arr = np.zeros(N)

    lam_max_Q = float(np.max(np.diag(Q)))
    lam_min_R = float(np.min(np.diag(R)))
    lam_max_R = float(np.max(np.diag(R)))

    x4  = x0_4.astype(float).copy()
    xm  = x0_4[:2].astype(float).copy()   # 2-D model monitor
    sw  = False
    t_s = None

    for i in range(1, N):
        x_pitch = x4[:2]
        u_m     = float(-(K @ x_pitch).item())    # model-based on pitch

        # Switching: pitch-only eta vs pitch-state threshold
        eta_p = x_pitch - xm
        thr   = (1.0 / (xi**2 * lam_max_R)) * (
                lam_max_Q * float(np.linalg.norm(x_pitch))**2
                + lam_min_R * u_m**2)

        if (not sw and w_u is not None
                and float(np.linalg.norm(eta_p))**2 >= thr
                and float(np.linalg.norm(x_pitch)) > x_min):
            sw  = True
            t_s = i

        eta_psq[i] = float(np.linalg.norm(eta_p))**2
        thr_arr[i] = thr

        if sw and w_u is not None:
            u = u_m + float(w_u @ phi_u_slosh(x_pitch))
        else:
            u = u_m
        u_arr[i] = u

        x4  = x4  + (f_slosh(x4, eps, omega_s) + B_4d.flatten() * u) * dt
        xm  = xm  + (Am @ xm + Bm.flatten() * u_m) * dt

        if not np.all(np.isfinite(x4)) or float(np.linalg.norm(x4)) > 1e3:
            arr4[i:] = np.nan
            u_arr[i:] = np.nan
            break
        arr4[i] = x4

    return arr4, u_arr, eta_psq, thr_arr, t_s


def pitch_cost(arr4, u_arr, dt=dt):
    """Pitch regulation cost: integral(x_pitch' Q x_pitch + u^2 R) dt.
    Returns 1e6 if the trajectory diverged (NaN in arr4)."""
    J      = 0.0
    Rs     = R.item()
    for i in range(len(arr4)):
        if not np.all(np.isfinite(arr4[i])):
            return 1e6
        x_p = arr4[i, :2]
        J  += (float(x_p @ Q @ x_p) + u_arr[i]**2 * Rs) * dt
    return J


# ---------------------------------------------------------------------------
# RL bases for pitch-only composite (quadratic critic, linear actor)
# No cubic: a3=0 here, so V* is quadratic for pitch-only LTI sub-problem.
# Linear actor: optimal u~* is linear in x_pitch for LTI mismatch.
# Bellman residual DEGRADES with eps -- this is the Assumption-1 signature.
# ---------------------------------------------------------------------------

def phi_v_slosh(x_pitch):
    """Quadratic critic: [theta^2, theta*thetadot, thetadot^2] (3 terms)."""
    th, thd = float(x_pitch[0]), float(x_pitch[1])
    return np.array([th**2, th * thd, thd**2])


def phi_u_slosh(x_pitch):
    """Linear actor: [theta, thetadot] (2 terms).
    Parity: phi_u(-x) = -phi_u(x) (odd; consistent with linear system)."""
    return np.array([float(x_pitch[0]), float(x_pitch[1])])


SLOSH_ICS   = [0.15, 0.25, 0.35, -0.2, -0.3]   # pitch-only ICs for RL collection
SLOSH_AMP   = 0.08
SLOSH_W     = 10
SLOSH_T_PE  = 8.0


def collect_data_slosh(x0_pitch, eps, omega_s, T_PE=SLOSH_T_PE, amp=SLOSH_AMP,
                       freqs=None, phase=2.3):
    """Collect pitch-state data from the 4-D plant.
    Behavior policy: u = -K @ x_pitch + probe (model-based + exploration).
    Only probe component stored in U (off-policy IRL convention).
    x0_pitch: 2-D pitch IC; psi = psi_dot = 0 initially (no slosh at start).
    Returns (X_pitch, U_probe, diagnostic_dict)."""
    if freqs is None:
        freqs = np.array([1.0, 2.3, 3.7, 5.1, 7.9, 11.3])

    def probe(tau):
        return amp * float(np.sum(np.sin(freqs * tau + phase)))

    N       = int(T_PE / dt)
    X_pitch = np.zeros((N, 2))
    U_probe = np.zeros(N)
    x4      = np.array([float(x0_pitch[0]), float(x0_pitch[1]), 0.0, 0.0])
    diverged = False

    for k in range(N):
        x_p  = x4[:2]
        u_m  = float(-(K @ x_p).item())
        ut   = probe(k * dt)
        X_pitch[k] = x_p
        U_probe[k] = ut
        u    = u_m + ut
        x4   = x4 + (f_slosh(x4, eps, omega_s) + B_4d.flatten() * u) * dt
        if not np.all(np.isfinite(x4)) or float(np.linalg.norm(x4)) > 1e3:
            diverged = True
            X_pitch = X_pitch[:k + 1]
            U_probe = U_probe[:k + 1]
            break

    return X_pitch, U_probe, {
        "abs_theta_max": float(np.abs(X_pitch[:, 0]).max()) if len(X_pitch) else 0.0,
        "diverged":      diverged
    }


def build_regression_slosh(segments, w_u_i, W, R_scalar, dt=dt):
    """Off-policy IRL regression matrix (Algorithm 1) on pitch-state features.
    segments: list of (X_pitch, U_probe) from collect_data_slosh.
    W: Bellman window length; windows never straddle IC boundaries."""
    n_v, n_u = 3, 2   # phi_v: 3 quadratic; phi_u: 2 linear
    rows, costs = [], []
    for X_pitch, U_probe in segments:
        k = 0
        while k + W < len(X_pitch):
            psi_v    = phi_v_slosh(X_pitch[k + W]) - phi_v_slosh(X_pitch[k])
            psi_u    = np.zeros(n_u)
            phi_cost = 0.0
            for j in range(k, k + W):
                mu_ij     = float(w_u_i @ phi_u_slosh(X_pitch[j]))
                diff      = float(U_probe[j]) - mu_ij
                psi_u    += 2.0 * diff * R_scalar * phi_u_slosh(X_pitch[j]) * dt
                phi_cost += (float(X_pitch[j] @ Q @ X_pitch[j])
                             + mu_ij**2 * R_scalar) * dt
            rows.append(np.concatenate([psi_v, psi_u]))
            costs.append(phi_cost)
            k += W
    if not rows:
        return np.zeros((0, n_v + n_u)), np.zeros(0)
    return np.array(rows), np.array(costs)


def policy_iteration_slosh(segments, W=SLOSH_W, dt=dt, eps_cv=1e-6, max_iter=80):
    """Algorithm 1 PI loop on pitch-state features.
    Returns (w_u, w_v, diag) where diag contains resid/cond per iteration.
    NOTE: Bellman residual may be large when slosh is active (Assumption-1 violation).
    """
    n_v, n_u = 3, 2
    w_u_i  = np.zeros(n_u)
    W_prev = np.zeros(n_v + n_u)
    diag   = []

    for _ in range(max_iter):
        Psi, Phi = build_regression_slosh(segments, w_u_i, W, R.item(), dt)
        if Psi.shape[0] == 0:
            break
        W_hat, *_ = np.linalg.lstsq(Psi, -Phi, rcond=None)
        w_u_i     = W_hat[n_v:]
        s         = np.linalg.svd(Psi, compute_uv=False)
        diag.append({
            "resid": float(np.linalg.norm(Psi @ W_hat + Phi)),
            "cond":  float(s[0] / (s[-1] + 1e-30)),
        })
        if float(np.linalg.norm(W_hat - W_prev)) < eps_cv:
            break
        W_prev = W_hat.copy()

    if not diag:
        return np.zeros(n_u), np.zeros(n_v), diag
    return W_hat[n_v:], W_hat[:n_v], diag


def learn_augmentation_slosh(eps, omega_s, ICs=None, amp=SLOSH_AMP,
                              W=SLOSH_W, T_PE=SLOSH_T_PE):
    """Learn pitch-only u~ from the 4-D plant at given (eps, omega_s).
    Returns (w_u, diag).  w_u=None if all collection segments diverged."""
    ICs = SLOSH_ICS if ICs is None else ICs
    segments = []
    for xs in ICs:
        X_p, U_pr, d = collect_data_slosh(np.array([xs, 0.0]), eps, omega_s, T_PE, amp)
        if not d["diverged"] and len(X_p) >= W + 1:
            segments.append((X_p, U_pr))
    if not segments:
        return None, []
    w_u, _, diag = policy_iteration_slosh(segments, W, dt)
    return w_u, diag


# ---------------------------------------------------------------------------
# eps-sweep: the core Tier-3 experiment
# ---------------------------------------------------------------------------

def eps_sweep(eps_vals=EPS_SWEEP, omega_s=OMEGA_S_DEFAULT,
              T=T, dt=dt, xi=xi, x0=x0_4):
    """Sweep coupling strength eps at fixed omega_s.
    For each eps, compute pitch costs for all 4 controllers.
    Also compute: 4-D closed-loop eigenvalues under K_emb (stability map),
    RL Bellman residual (Assumption-1 signature), switch-fire status.
    Returns list of result dicts.
    """
    x_min = 0.05 * float(np.linalg.norm(x0[:2]))
    results = []

    for eps in eps_vals:
        r = {"eps": float(eps)}

        # --- analytical: K_emb stability on 4-D plant ---
        Ke   = K_emb()
        A    = A_4d(eps, omega_s)
        eigs = np.linalg.eigvals(A - B_4d @ Ke)
        r["eig_real_max"] = float(np.max(eigs.real))
        r["emb_stable"]   = bool(np.all(eigs.real < 0))

        # --- model-based (K_emb) on 4-D plant, no switch ---
        arr_mb, u_mb = sim_4d_fixed(x0, eps, omega_s, Ke, T, dt)
        r["J_mb"] = pitch_cost(arr_mb, u_mb, dt)

        # --- oracle (4-D LQR) ---
        Ko = K_oracle(eps, omega_s)
        if Ko is not None:
            arr_or, u_or = sim_4d_fixed(x0, eps, omega_s, Ko, T, dt)
            r["J_oracle"] = pitch_cost(arr_or, u_or, dt)
        else:
            r["J_oracle"] = 1e6

        # --- full-state composite (augmented 4-D ARE) ---
        Kfc = K_fullcomp(eps, omega_s)
        if Kfc is not None:
            arr_fc, u_fc = sim_4d_fixed(x0, eps, omega_s, Kfc, T, dt)
            r["J_fullcomp"] = pitch_cost(arr_fc, u_fc, dt)
        else:
            r["J_fullcomp"] = 1e6

        # --- pitch-only composite (RL + switching framework) ---
        wu, diag = learn_augmentation_slosh(eps, omega_s)
        if diag:
            r["rl_resid"] = diag[-1]["resid"]
            r["rl_cond"]  = diag[-1]["cond"]
        else:
            r["rl_resid"] = None
            r["rl_cond"]  = None

        arr_pc, u_pc, eta_psq, thr_arr, ts_pc = sim_slosh_framework(
            x0, eps, omega_s, w_u=wu, T=T, dt=dt, xi=xi, x_min=x_min)
        r["J_pitchcomp"] = pitch_cost(arr_pc, u_pc, dt)
        r["t_s"] = None if ts_pc is None else ts_pc * dt
        # switch_fires: did eta_pitch cross the threshold? (ts_pc detects it since wu is not None)
        r["switch_fires"] = ts_pc is not None

        # --- benefit% (relative to model-based; 1e6 -> NaN for display) ---
        def ben(J):
            if r["J_mb"] >= 1e5 or J >= 1e5:
                return float("nan")
            return (r["J_mb"] - J) / r["J_mb"] * 100.0

        r["benefit_oracle"]    = ben(r["J_oracle"])
        r["benefit_fullcomp"]  = ben(r["J_fullcomp"])
        r["benefit_pitchcomp"] = ben(r["J_pitchcomp"])

        results.append(r)

    return results


# ---------------------------------------------------------------------------
# Main: diagnostics, sweep, and figures
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Rocket Pitch Tier 3: SLOSH (Assumption-1 violation) ===\n")

    # Assumption 1 callout
    print("Assumption 1 (paper Sec 2): 'The dimensions of the linear model and the")
    print("system are the same, i.e., Am in R^{nxn}, Bm in R^{nxm}.'")
    print("HERE VIOLATED: true plant is 4-D, model is 2-D.\n")

    # Structural checks
    print("--- 2-D model structural checks ---")
    print(f"  open-loop eig(Am): {np.round(_eig_open.real, 3)} (expect one positive)")
    print(f"  controllability rank: {np.linalg.matrix_rank(_ctrb)} (expect 2)")
    print(f"  closed-loop eig:  {np.round(_eig_closed.real, 4)} (expect all negative)")
    print(f"  K = {K.flatten()}\n")

    omega_s = OMEGA_S_DEFAULT

    # Eigenvalue crossing vs eps (stability map)
    # 2-D stability grid: K_emb on 4-D plant across (eps, omega_s)
    print("--- K_emb 2-D stability grid: eps in [0,12] x omega_s in [0.5,15] ---")
    eps_fine   = np.linspace(0.0, 12.0, 49)
    oms_fine   = np.linspace(0.5, 15.0, 30)
    Ke_g       = K_emb()
    max_re_2d  = np.zeros((len(eps_fine), len(oms_fine)))
    for _i, eps_g in enumerate(eps_fine):
        for _j, oms_g in enumerate(oms_fine):
            A_g  = A_4d(eps_g, oms_g)
            egs  = np.linalg.eigvals(A_g - B_4d @ Ke_g)
            max_re_2d[_i, _j] = float(np.max(egs.real))
    grid_max_re = float(max_re_2d.max())
    print(f"  Max Re across entire grid: {grid_max_re:.4f}")
    print(f"  Any unstable? {'YES' if grid_max_re > 0 else 'NO -- K_emb stable everywhere in sweep range'}")
    print("  'Breaks' = pitch-only composite loses USEFULNESS, not stability.\n")

    # omega_s slice for figure 4 (at omega_s=OMEGA_S_DEFAULT)
    _j_default = int(round((omega_s - 0.5) / (15.0 - 0.5) * (len(oms_fine) - 1)))
    max_re     = [float(max_re_2d[_i, _j_default]) for _i in range(len(eps_fine))]
    eps_crit_idx = next((_i for _i, v in enumerate(max_re) if v > 0), None)
    eps_crit     = float(eps_fine[eps_crit_idx]) if eps_crit_idx is not None else None

    # omega_s damping criterion at eps=2.0 (damping = min|Re(eig)| across 4 modes)
    print("--- omega_s damping scan (eps=2.0): most slowly-decaying mode ---")
    print(f"  {'omega_s':>8}  {'min|Re(eig)|':>14}  {'slowest tc [s]':>16}")
    for oms in np.linspace(1.0, 12.0, 12):
        A_s   = A_4d(2.0, oms)
        egs_s = np.linalg.eigvals(A_s - B_4d @ Ke_g)
        min_re = float(np.min(np.abs(egs_s.real)))
        tc    = 1.0 / min_re if min_re > 1e-8 else float("inf")
        print(f"  {oms:>8.2f}  {min_re:>14.4f}  {tc:>16.2f}")
    print(f"  (Smallest min|Re| = slowest-decaying mode; low omega_s -> slow slosh.)")
    print(f"  Using omega_s = {omega_s} rad/s for main sweep.\n")

    # Main eps-sweep
    print(f"--- eps-sweep (omega_s={omega_s} rad/s) ---")
    print(f"  {'eps':>6}  {'J_mb':>8}  {'J_or':>8}  {'J_fc':>8}  {'J_pc':>8}"
          f"  {'ben_or%':>8}  {'ben_fc%':>8}  {'ben_pc%':>8}"
          f"  {'switch':>7}  {'t_s':>6}  {'resid':>8}  {'emb_stab':>9}")

    sweep = eps_sweep(EPS_SWEEP, omega_s)

    for r in sweep:
        bo  = r["benefit_oracle"]
        bfc = r["benefit_fullcomp"]
        bpc = r["benefit_pitchcomp"]
        ts  = r["t_s"]
        rsd = r["rl_resid"]
        J_or_s  = f"{r['J_oracle']:8.4f}" if r["J_oracle"] < 1e5 else "     nan"
        J_fc_s  = f"{r['J_fullcomp']:8.4f}" if r["J_fullcomp"] < 1e5 else "     nan"
        J_pc_s  = f"{r['J_pitchcomp']:8.4f}" if r["J_pitchcomp"] < 1e5 else "     nan"
        bo_s    = f"{bo:8.1f}" if not np.isnan(bo) else "     nan"
        bfc_s   = f"{bfc:8.1f}" if not np.isnan(bfc) else "     nan"
        bpc_s   = f"{bpc:8.1f}" if not np.isnan(bpc) else "     nan"
        ts_s    = f"{ts:.2f}" if ts is not None else "  None"
        rsd_s   = f"{rsd:.2e}" if rsd is not None else " -------"
        stab_s  = "YES" if r["emb_stable"] else "*** NO ***"
        sw_s    = "YES" if r["switch_fires"] else " no"
        print(f"  {r['eps']:>6.2f}  {r['J_mb']:>8.4f}  {J_or_s}  {J_fc_s}  {J_pc_s}"
              f"  {bo_s}  {bfc_s}  {bpc_s}  {sw_s:>7}  {ts_s:>6}  {rsd_s}  {stab_s:>9}")

    # -----------------------------------------------------------------------
    # Figures
    # -----------------------------------------------------------------------

    # Helper to extract finite eps values and corresponding benefits
    def _fin(field):
        xs, ys = [], []
        for r in sweep:
            v = r[field]
            if not np.isnan(v) and r["J_mb"] < 1e5:
                xs.append(r["eps"])
                ys.append(v)
        return xs, ys

    # Fig 1: benefit% vs eps for all controllers
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    for field, label, style in [
        ("benefit_oracle",    "Oracle 4-D LQR",          "C2-o"),
        ("benefit_fullcomp",  "Full-state composite",     "C1-s"),
        ("benefit_pitchcomp", "Pitch-only composite (RL)", "C0-^"),
    ]:
        xs, ys = _fin(field)
        if xs:
            ax1.plot(xs, ys, style, label=label, lw=1.5, ms=6)
    ax1.axhline(0, color="k", lw=0.8, ls="--")
    if eps_crit is not None:
        ax1.axvline(eps_crit, color="gray", ls=":", lw=1.2,
                    label=f"K_emb 4-D stability boundary\n(eps={eps_crit:.1f})")
    ax1.set_xlabel("Coupling strength eps  [1/s²]")
    ax1.set_ylabel("Benefit% = (J_mb - J_ctrl) / J_mb × 100")
    ax1.legend(fontsize=9, loc="upper right")
    ax1.set_title(f"Tier-3 slosh: validity boundary (omega_s = {omega_s} rad/s)\n"
                  "Pitch-only composite degrades as slosh coupling grows")
    plt.tight_layout()
    plt.show()

    # Fig 2: state trajectories -- "works" vs "breaks" regimes
    eps_works = 0.2   # weak coupling -> framework works
    eps_breaks = 5.0  # strong coupling -> framework breaks (or is near boundary)
    t_arr  = np.arange(int(T / dt)) * dt

    fig2, axes2 = plt.subplots(3, 2, figsize=(11, 9), sharex=True)
    fig2.suptitle(f"Tier-3 slosh: state trajectories (omega_s = {omega_s} rad/s)\n"
                  f"Left: eps={eps_works} (works)   Right: eps={eps_breaks} (breaks)")

    for col, eps_val in enumerate([eps_works, eps_breaks]):
        # model-based
        Ke = K_emb()
        arr_mb_p, u_mb_p = sim_4d_fixed(x0_4, eps_val, omega_s, Ke)
        # pitch-only composite (re-learn)
        wu_p, _ = learn_augmentation_slosh(eps_val, omega_s)
        arr_pc_p, u_pc_p, *_ = sim_slosh_framework(x0_4, eps_val, omega_s, w_u=wu_p)
        # oracle
        Ko = K_oracle(eps_val, omega_s)
        if Ko is not None:
            arr_or_p, u_or_p = sim_4d_fixed(x0_4, eps_val, omega_s, Ko)
        else:
            arr_or_p = arr_mb_p.copy(); arr_or_p[:] = np.nan

        lbl_mb = "model-based" if col == 0 else None
        lbl_pc = "pitch composite" if col == 0 else None
        lbl_or = "oracle" if col == 0 else None

        for arr, lbl, sty in [(arr_mb_p, lbl_mb, "k-"),
                               (arr_pc_p, lbl_pc, "C0-"),
                               (arr_or_p, lbl_or, "C2--")]:
            if np.any(np.isfinite(arr[:, 0])):
                axes2[0, col].plot(t_arr, arr[:, 0], sty, lw=1.5, label=lbl, alpha=0.8)
                axes2[1, col].plot(t_arr, arr[:, 1], sty, lw=1.5, alpha=0.8)
                axes2[2, col].plot(t_arr, arr[:, 2], sty, lw=1.5, alpha=0.8)

        axes2[0, col].set_title(f"eps = {eps_val}")
        axes2[0, col].set_ylabel(r"$\theta$ [rad]")
        axes2[1, col].set_ylabel(r"$\dot\theta$ [rad/s]")
        axes2[2, col].set_ylabel(r"$\psi$ [rad]  (slosh)")
        axes2[2, col].set_xlabel("t [s]")

    axes2[0, 0].legend(fontsize=8)
    plt.tight_layout()
    plt.show()

    # Fig 3: slosh amplitude |psi - theta| vs time for several eps values
    fig3, ax3 = plt.subplots(figsize=(8, 4))
    Ke = K_emb()
    for eps_val, style in [(0.05, "C0"), (0.2, "C1"), (1.0, "C3"), (3.0, "C2"), (7.0, "k")]:
        arr_s, _ = sim_4d_fixed(x0_4, eps_val, omega_s, Ke)
        slosh_amp = arr_s[:, 2] - arr_s[:, 0]   # psi - theta
        label = f"eps={eps_val}"
        if np.any(np.isfinite(slosh_amp)):
            ax3.plot(t_arr, np.where(np.isfinite(slosh_amp), np.abs(slosh_amp), np.nan),
                     style, lw=1.5, label=label, alpha=0.8)
    ax3.set_xlabel("t [s]")
    ax3.set_ylabel(r"$|\psi - \theta|$ [rad]  (slosh relative amplitude)")
    ax3.legend(fontsize=9)
    ax3.set_title(f"Tier-3 slosh: relative displacement |psi-theta| (omega_s={omega_s} rad/s)\n"
                  "Strong coupling: psi tracks theta (small gap); weak coupling: psi lags (large gap)")
    ax3.set_ylim(bottom=0)
    plt.tight_layout()
    plt.show()

    # Fig 4: 2-D stability map -- max Re(eig) over (eps, omega_s) grid
    fig4, ax4 = plt.subplots(figsize=(8, 5))
    # max_re_2d shape: (len(eps_fine), len(oms_fine)); transpose for (omega_s on y, eps on x)
    pcm = ax4.pcolormesh(eps_fine, oms_fine, max_re_2d.T,
                         cmap="RdBu_r", vmin=-1.0, vmax=0.5, shading="auto")
    fig4.colorbar(pcm, ax=ax4, label="Max Re(eig)  [1/s]")
    cs = ax4.contour(eps_fine, oms_fine, max_re_2d.T, levels=[0.0],
                     colors="k", linewidths=1.5, linestyles="--")
    ax4.clabel(cs, fmt="Re=0 (stability boundary)", fontsize=8)
    ax4.axhline(omega_s, color="C2", ls=":", lw=1.5,
                label=f"sweep omega_s = {omega_s} rad/s")
    ax4.set_xlabel("Coupling strength eps  [1/s²]")
    ax4.set_ylabel("Slosh freq omega_s  [rad/s]")
    ax4.legend(fontsize=9)
    ax4.set_title("Tier-3 slosh: K_emb 4-D stability map (max Re < 0 = stable)\n"
                  "'Breaks' = pitch composite loses usefulness, not K_emb stability")
    plt.tight_layout()
    plt.show()

    # Fig 5: RL Bellman residual vs eps (Assumption-1 signature)
    eps_rl  = [r["eps"] for r in sweep if r["rl_resid"] is not None]
    res_rl  = [r["rl_resid"] for r in sweep if r["rl_resid"] is not None]
    if eps_rl:
        fig5, ax5 = plt.subplots(figsize=(7, 4))
        ax5.semilogy(eps_rl, res_rl, "C0-o", lw=1.5, ms=6)
        ax5.set_xlabel("Coupling strength eps  [1/s²]")
        ax5.set_ylabel("Pitch-only RL Bellman residual")
        ax5.set_title("Tier-3 slosh: RL residual grows with eps\n"
                      "(same pitch state, different slosh => ill-defined value fn: Assumption-1 signature)")
        ax5.grid(True, which="both", alpha=0.3)
        if eps_crit is not None:
            ax5.axvline(eps_crit, color="C3", ls=":", lw=1.2,
                        label=f"K_emb stability boundary eps={eps_crit:.1f}")
            ax5.legend(fontsize=9)
        plt.tight_layout()
        plt.show()
