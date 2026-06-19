from scipy.linalg import solve_continuous_are
import numpy as np
import matplotlib.pyplot as plt


# Owner's own example: single-axis rocket PITCH attitude control via thrust
# vector control (TVC).  Control-affine, same state dimension for model and
# plant (Assumption 1).  Tier 1 = AERO NONLINEARITY ONLY (cubic Duffing-like
# pitching moment).  Fuel slosh and mass burn are LATER tiers -- design is built
# to extend to them, but they are not implemented here.
#
# This is the paper's Example 1 (perturbed oscillator) generalized to a rocket:
# a finless / aerodynamically-unstable airframe stabilized by gimballing.



# State / control
#   x = [theta, theta_dot]   theta = pitch angle [rad], theta_dot = pitch rate
#   u = delta                gimbal deflection [rad] (scalar control)
#   xdot = f(x) + g(x)*u     control-affine


# Physical parameter mapping (documented; we use clean rounded values below):
#   a1 = q*S*d*Cm_alpha / I     linear aero pitching-moment slope / inertia
#   a3 = q*S*d*Cm_3    / I      CUBIC aero moment (Duffing-like) / inertia
#   b  = T * ell / I            TVC control authority: thrust * moment arm / I
# Finless rocket => Cm_alpha > 0 => a1 > 0 => statically UNSTABLE in pitch
# (the airframe pitches away from the wind), stabilized only by TVC.
#
# Clean Tier-1 values (rad, s):
a1 = 4.0    # unstable: open-loop poles at +/- sqrt(a1) = +/- 2 rad/s
b  = 8.0    # control authority (T*ell/I); g is CONSTANT here

# Cubic-aero strength is the SCENARIO KNOB (analogous to rho in section 7.2).
# Same A_m / same K in both scenarios -- only the perturbation a3 changes.
a3_mild   = 0.5     # weak cubic  -> mismatch stays under threshold, NO switch
a3_strong = 60.0    # strong cubic -> mismatch grows -> switch fires (~0.54 s), then RL


def f(x, a3):
    # true nonlinear drift: linear aero + cubic Duffing-like aero moment
    theta, theta_dot = x
    return np.array([theta_dot,
                     a1 * theta + a3 * theta**3])


def g(x):
    # control vector field: constant, gimbal torque enters the rate equation
    return np.array([0.0, b])



# Available (deliberately-imperfect) model: LINEAR AERO ONLY.
# Same state dimension (Assumption 1); the only thing the model is missing is
# the cubic a3*theta^3.  Because g(x) is constant, B_m = g EXACTLY -- there is
# ZERO input-channel mismatch; ALL model error is the omitted cubic in f.

Am = np.array([[0.0, 1.0],
               [a1,  0.0]])
Bm = np.array([[0.0],
               [b]])

# LQR weights.  Penalize pitch angle (regulate attitude); modest rate / control.
Q = np.diag([1.0, 0.1])
R = np.array([[0.2]])

# Model-based LQR gain (Eq. 11): u_m = -K x, from the inaccurate (linear) model.
Pm = solve_continuous_are(Am, Bm, Q, R)
K  = np.linalg.inv(R) @ Bm.T @ Pm     # shape (1, 2)


# Near-origin ground-truth augmentation gain K_tilde_lin (for later validation).
# The ONLY nonlinearity is a3*theta^3, whose Jacobian at the origin is zero, so
# the linearized true plant EQUALS A_m exactly.  Hence K_tilde_lin is just the
# section-7.1 augmented-Riccati gain with A -> A_m:
#   K_tilde = inv(R) B^T ARE(A_m - B_m K, B_m, Q, R)
# The RL augmentation ũ(x) should match -K_tilde_lin x for SMALL states (where
# the cubic is negligible).  Computed now; used in step 5.
P_tilde = solve_continuous_are(Am - Bm @ K, Bm, Q, R)
K_tilde_lin = np.linalg.inv(R) @ Bm.T @ P_tilde   # shape (1, 2)


# STEP-1 CHECKS: open-loop instability + controllability of (A_m, B_m).
# These must actually be verified, not just asserted.

eig_open = np.linalg.eigvals(Am)
ctrb = np.hstack([Bm, Am @ Bm])           # [B_m, A_m B_m]
ctrb_rank = np.linalg.matrix_rank(ctrb)
ctrb_det = np.linalg.det(ctrb)

# closed-loop poles under the model-based gain (sanity: should be stable)
eig_closed = np.linalg.eigvals(Am - Bm @ K)


# =============================================================================
# STEP 2: nonlinear simulation + switching detection.
#   - True plant integrated with the FULL nonlinear f (incl. cubic), under the
#     model-based control u = -K x.
#   - Model trajectory x_m integrated with the LINEAR model (A_m, B_m) under the
#     SAME applied input -> mismatch eta = x - x_m (Eq. 14).
#   - Switch fires (Theorem 1 / Eq. 17 & 40) the first instant
#       ||eta||^2 >= threshold   AND   ||x|| > x_min.
# No augmentation is applied yet (that's steps 3-4); here we only LOCATE t_s and
# confirm: mild a3 -> never fires, strong a3 -> fires early.
#
# Threshold uses lam_MAX(Q) per OBS-5 (paper's figures); here Q=diag(1,0.1) is
# NOT degenerate, so lam_max vs lam_min is the section-7.1 modeling choice.
# =============================================================================

x0 = np.array([0.3, 0.0])   # ~17 deg pitch disturbance (e.g. a wind-gust kick), zero rate
dt = 0.01
xi = 1.35                   # suboptimality slack in the threshold (Eq. 17); tuned below
x_min = 0.05 * np.linalg.norm(x0)   # switching guard: don't switch once essentially regulated


def sim_switching(x0, a3, T, dt, K, xi, x_min, allow_switch=True):
    lenT = int(T / dt)
    n = x0.shape[0]
    arr   = np.zeros((lenT, n)); arr[0]   = x0
    arr_m = np.zeros((lenT, n)); arr_m[0] = x0
    eta        = np.zeros((lenT, n))
    thresh_arr = np.zeros(lenT)
    u_arr      = np.zeros(lenT)

    lam_max_Q = np.max(np.diag(Q))
    lam_min_R = np.min(np.diag(R))
    lam_max_R = np.max(np.diag(R))

    x  = x0.astype(float).copy()
    xm = x0.astype(float).copy()
    t_s = None

    for i in range(1, lenT):
        u_m = (-(K @ x)).item()        # scalar model-based control on the REAL state

        thr = (1 / (xi**2 * lam_max_R)) * (lam_max_Q * np.linalg.norm(x)**2
                                           + lam_min_R * u_m**2)
        if (t_s is None and allow_switch
                and np.linalg.norm(x - xm)**2 >= thr
                and np.linalg.norm(x) > x_min):
            t_s = i

        thresh_arr[i] = thr
        u_arr[i] = u_m

        # integrate: true nonlinear plant vs linear model, SAME input u_m
        x  = x  + (f(x, a3) + g(x) * u_m) * dt
        xm = xm + (Am @ xm + Bm.flatten() * u_m) * dt
        arr[i] = x; arr_m[i] = xm; eta[i] = x - xm

    eta_norm = np.linalg.norm(eta, axis=1)
    return arr, arr_m, eta, eta_norm, thresh_arr, u_arr, t_s


def sim_switching_integral(x0, a3, T, dt, K, xi, tau, x_min, allow_switch=True):
    """Integral / dwell-time switching (OBS-10 extension). Instead of switching at the FIRST
    instant ||eta||^2 >= threshold (paper Eq. 21), switch when the ACCUMULATED certificate
    violation exceeds a tolerance tau:
        t_s = inf{ t : integral_0^t  max(0, ||eta||^2 - thr) ds  >= tau }
    tau -> 0+ recovers the paper's first-crossing rule. tau = "how much cumulative
    suboptimality I tolerate before switching" -- a constant with real pull on t_s where
    xi alone saturates (xi only rescales thr; see OBS-10). Returns the switch index or None.

    CAVEAT (OBS-4 / OBS-10): once the true plant regulates (here ~0.6 s), ||eta|| is dominated
    by the UNSTABLE model monitor x_m diverging, not genuine plant-model mismatch. So switches
    with t_s beyond regulation integrate that artifact. A principled version freezes/resets x_m
    at regulation (the OBS-4 fix). Left explicit here as a meeting talking point.
    """
    lenT = int(T / dt)
    lam_max_Q = np.max(np.diag(Q))
    lam_min_R = np.min(np.diag(R))
    lam_max_R = np.max(np.diag(R))

    x  = x0.astype(float).copy()
    xm = x0.astype(float).copy()
    acc = 0.0          # accumulated certificate violation, integral of (||eta||^2 - thr)_+
    t_s = None

    for i in range(1, lenT):
        u_m = (-(K @ x)).item()
        thr = (1 / (xi**2 * lam_max_R)) * (lam_max_Q * np.linalg.norm(x)**2
                                           + lam_min_R * u_m**2)
        acc += max(0.0, np.linalg.norm(x - xm)**2 - thr) * dt

        # acc > 0 guard makes tau=0 fire at the first genuine violation (== Eq. 21),
        # not spuriously at t0 where acc is trivially 0 >= 0.
        if (t_s is None and allow_switch and acc >= tau and acc > 0.0
                and np.linalg.norm(x) > x_min):
            t_s = i

        x  = x  + (f(x, a3) + g(x) * u_m) * dt
        xm = xm + (Am @ xm + Bm.flatten() * u_m) * dt

    return t_s


if __name__ == "__main__":
    print("=== Step 1: dynamics, model, LQR gain, structural checks ===")
    print(f"a1 = {a1},  b = {b}")
    print(f"open-loop eig(A_m)        = {eig_open}   (unstable iff any Re > 0)")
    print(f"controllability [B, A B]  =\n{ctrb}")
    print(f"  rank = {ctrb_rank} (need 2),  det = {ctrb_det:.3f} (need != 0)")
    print(f"model-based gain K        = {K}")
    print(f"closed-loop eig(A_m-B_mK) = {eig_closed}   (should be stable)")
    print(f"K_tilde_lin (val target)  = {K_tilde_lin}")

    print("\n=== Step 2: switching behavior (mild vs strong cubic) ===")
    T2 = 8.0
    *_, t_s_mild   = sim_switching(x0, a3_mild,   T2, dt, K, xi, x_min)
    *_, t_s_strong = sim_switching(x0, a3_strong, T2, dt, K, xi, x_min)
    print(f"mild   a3={a3_mild:<5}: t_s = "
          + ("None (no switch) " if t_s_mild is None else f"{t_s_mild*dt:.3f} s"))
    print(f"strong a3={a3_strong:<5}: t_s = "
          + ("None (no switch) " if t_s_strong is None else f"{t_s_strong*dt:.3f} s"))

    print("\n=== Step 2b (OBS-10): integral/dwell-time switching -- giving the constant 'pull' ===")
    print("  switch when  integral( (||eta||^2 - thr)_+ ) dt >= tau   (tau->0+ == paper Eq. 21)")
    print(f"  strong a3={a3_strong}, xi={xi}  (compare: xi alone gives t_s=0.62->0.15 s over xi in [1,5]):")
    for tau in [0.0, 0.01, 0.1, 0.5, 2.0]:
        t_int = sim_switching_integral(x0, a3_strong, 14.0, dt, K, xi, tau, x_min)
        print(f"    tau={tau:<5}: t_s = " + ("None (no switch)" if t_int is None else f"{t_int*dt:.3f} s"))
    print("  -> tau is a strong, near-monotone lever on t_s where xi saturates (see OBS-10).")
    print("     CAVEAT: t_s beyond ~0.6 s integrates the OBS-4 monitor divergence, not real mismatch.")
