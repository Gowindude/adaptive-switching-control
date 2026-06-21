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
