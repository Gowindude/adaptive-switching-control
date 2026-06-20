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


# =============================================================================
# STEPS 3-4: off-policy integral-RL augmentation (Algorithm 1).
#
# FAIL-FIRST experiment (owner's call): pair a QUADRATIC critic with an odd,
# cubic-capable actor and watch the critic fail, then diagnose. Only the critic
# is "wrong" here, so any failure is attributable to it (controlled experiment).
#
# Parity (the rocket is odd-symmetric (th,thd)->(-th,-thd), cost even):
#   V is EVEN -> phi_v has only even monomials.
#   u~ is ODD -> phi_u has only odd monomials.
# =============================================================================

def phi_v_quad(x):
    # Quadratic critic -- the FAIL-FIRST diagnostic (kept for the quad-vs-quartic figure).
    # V* here is NOT quadratic: the cubic drift a3*th^3 seeds quartic terms in the augmented
    # HJB (Vx2*a3*th^3 -> th^4, th^3*thd), so a quadratic V leaves an unfittable residual at
    # large theta. Demonstrated: resid 3.5e-2, garbage actor [0.09,-0.05,0.02]. See OBS-11.
    th, thd = x
    return np.array([th**2, th*thd, thd**2])


def phi_v(x):
    # DEPLOYED CRITIC: quartic, even parity. Quadratic core + the two quartic terms the cubic
    # drift sources (th^4, th^3*thd). th^3*thd is also the term whose d/d(thd) yields th^3 -- the
    # critic term that generates the actor's cubic. Recovers resid 9.1e-3 (vs 3.5e-2 quadratic).
    th, thd = x
    return np.array([th**2, th*thd, thd**2, th**4, th**3 * thd])


def phi_u(x):
    # ACTOR basis for u~(x). Odd parity, cubic-capable so it is NOT the
    # bottleneck: linear part recovers K_tilde_lin near origin, th^3 counters
    # the cubic aero moment.
    th, thd = x
    return np.array([th, thd, th**3])


def collect_data(x_start, a3, T_PE, dt, K, amp, stabilize=False, freqs=None, phase=2.3):
    """Separate exploratory rollout (OBS-3 architecture). Run the TRUE plant under a behavior
    policy, recording the state and the behavior AUGMENTATION (the part on top of u_m=-Kx).
    Off-policy IRL lets the learning data come from this dedicated run.

    behavior augmentation:
      stabilize=False : ut = probe                 (model-based behavior u = -Kx + probe)
      stabilize=True  : ut = -(a3/b)*theta^3 + probe  (rough cubic FEEDFORWARD + probe)

    CAVEAT (control-authority tension, OBS-11): with stabilize=False the model-based -Kx cannot
    hold large theta (cliff ~|theta|=0.48 at a3=60) -- the very instability the augmentation
    exists to fix -- so the cubic is barely excitable. stabilize=True cancels the cubic so the
    plant is ~linear+stable and large-theta data is reachable (DEBT: the feedforward 'knows' a3)."""
    if freqs is None:
        freqs = np.array([1.0, 2.3, 3.7, 5.1, 7.9, 11.3])

    def probe(tau):
        return amp * np.sum(np.sin(freqs * tau + phase))    # scalar

    N = int(T_PE / dt)
    X = np.zeros((N, 2))
    U = np.zeros(N)
    x = x_start.astype(float).copy()
    diverged = False
    for k in range(N):
        u_m = (-(K @ x)).item()
        ff  = -(a3 / b) * x[0]**3 if stabilize else 0.0   # rough cubic feedforward (stabilizing part)
        ut  = ff + probe(k * dt)
        X[k] = x                      # state at step k
        U[k] = ut                     # behavior augmentation applied at X[k]
        u = u_m + ut
        x = x + (f(x, a3) + g(x) * u) * dt
        if not np.all(np.isfinite(x)) or np.linalg.norm(x) > 1e3:
            diverged = True
            X, U = X[:k+1], U[:k+1]   # truncate at blow-up
            break

    diag = {
        "theta_min": X[:, 0].min(), "theta_max": X[:, 0].max(),
        "abs_theta_max": np.abs(X[:, 0]).max(),
        "cubic_to_linear": np.abs(X[:, 0]).max()**2,   # th^3/th ratio at the extreme = th^2
        "diverged": diverged, "N_kept": len(X),
    }
    return X, U, diag


def build_regression(X_data, U_data, w_u_i, W, R_scalar, dt):
    # Ported from perturbed-oscillator.py, parameterized: n_v/n_u replace the
    # hardcoded 3s so the bases can change with a one-line edit.
    n_u = len(phi_u(X_data[0]))
    rows, costs = [], []
    k = 0
    while k + W < len(X_data):
        psi_v = phi_v(X_data[k+W]) - phi_v(X_data[k])   # value difference V(x_{k+W}) - V(x_k)
        psi_u = np.zeros(n_u)
        phi_cost = 0.0
        for j in range(k, k+W):
            mu_i_j = w_u_i @ phi_u(X_data[j])           # current-iteration target policy mu_i(x_j)
            diff = U_data[j] - mu_i_j                    # behavior - target (off-policy gap)
            psi_u += 2 * diff * R_scalar * phi_u(X_data[j]) * dt
            phi_cost += (X_data[j] @ Q @ X_data[j] + mu_i_j**2 * R_scalar) * dt
        rows.append(np.concatenate([psi_v, psi_u]))
        costs.append(phi_cost)
        k += W
    return np.array(rows), np.array(costs)


def policy_iteration(X_data, U_data, W, dt, eps=1e-6, max_iter=50):
    n_v = len(phi_v(X_data[0]))
    n_u = len(phi_u(X_data[0]))
    w_u_i = np.zeros(n_u)
    W_prev = np.zeros(n_v + n_u)
    history, diag = [], []
    for _ in range(max_iter):
        Psi, Phi = build_regression(X_data, U_data, w_u_i, W, R.item(), dt)
        W_hat, *_ = np.linalg.lstsq(Psi, -Phi, rcond=None)
        w_v   = W_hat[:n_v]
        w_u_i = W_hat[n_v:]
        history.append(W_hat)

        # --- DIAGNOSTICS (how to "see" the basis fail) ---
        resid = np.linalg.norm(Psi @ W_hat + Phi)       # Bellman LS residual (~0 => basis can fit V)
        s = np.linalg.svd(Psi, compute_uv=False)
        cond = s[0] / s[-1]                              # huge => a regressor is under-excited (data problem)
        diag.append({"resid": resid, "cond": cond})

        if np.linalg.norm(W_hat - W_prev) < eps:
            break
        W_prev = W_hat
    return w_u_i, w_v, history, diag


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

    print("\n=== Steps 3-4: off-policy IRL augmentation ===")
    print(f"  deployed bases: phi_v = {len(phi_v(x0))} terms (QUARTIC critic), "
          f"phi_u = {len(phi_u(x0))} terms (odd cubic actor)")
    T_PE = 8.0; W = 10

    # OBS-11 contrast: model-based behavior -Kx cannot reach large theta (the cubic's regime).
    Xmb, _, dmb = collect_data(np.array([0.4, 0.0]), a3_strong, T_PE, dt, K, 0.2, stabilize=False)
    print(f"  [OBS-11] model-based behavior (-Kx + probe): |theta|_max = {dmb['abs_theta_max']:.3f}"
          f"  (stuck near x_start; cubic barely excited)")

    # Stabilizing behavior policy (-(a3/b)theta^3 feedforward + probe) reaches large theta bounded,
    # where the cubic IS identifiable. DEBT: this feedforward 'knows' a3 (PLAN tech-debt #1).
    x_start = np.array([1.5, 0.0]); amp = 0.3
    X, U, cdiag = collect_data(x_start, a3_strong, T_PE, dt, K, amp, stabilize=True)
    print(f"  stabilized behavior: x_start={x_start}, amp={amp}  -> "
          f"|theta|_max = {cdiag['abs_theta_max']:.3f}  diverged={cdiag['diverged']}  N={cdiag['N_kept']}")

    if cdiag['N_kept'] > W + 1:
        w_u, w_v, history, pidiag = policy_iteration(X, U, W, dt)
        print(f"  policy iteration: {len(history)} iters")
        print(f"    Bellman residual = {pidiag[-1]['resid']:.4e}   "
              "(quartic critic; quadratic gives 3.5e-2 -- OBS-11)")
        print(f"    cond(Psi)        = {pidiag[-1]['cond']:.4e}")
        print(f"    learned w_v (critic) = {w_v}")
        print(f"    learned w_u (actor)  = {w_u}")
        print(f"    [validation step 5 will check: linear part ~ -K_tilde_lin = "
              f"{-K_tilde_lin.flatten()} on SMALL-theta data; usefulness on LARGE-theta]")
    else:
        print("  collection diverged too early to learn -- back off amp / x_start.")
