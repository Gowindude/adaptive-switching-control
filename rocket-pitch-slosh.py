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
