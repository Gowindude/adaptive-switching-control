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


def build_regression(segments, w_u_i, W, R_scalar, dt):
    # Off-policy IRL regression (ported from perturbed-oscillator.py, parameterized n_v/n_u).
    # `segments` is a list of (X, U) rollouts; rows are built WITHIN each segment so a Bellman
    # window never straddles two trajectories. This is what lets us learn from MULTIPLE
    # moderate-theta initial conditions at once -- the unbiased-fit fix (OBS-12): one fat
    # large-theta rollout is poison (target policy can't stabilize there -> V undefined), while
    # several moderate, stabilizable rollouts pin both the linear gain and the cubic cleanly.
    n_u = len(phi_u(segments[0][0][0]))
    rows, costs = [], []
    for X_data, U_data in segments:
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


def policy_iteration(segments, W, dt, eps=1e-6, max_iter=80):
    n_v = len(phi_v(segments[0][0][0]))
    n_u = len(phi_u(segments[0][0][0]))
    w_u_i = np.zeros(n_u)
    W_prev = np.zeros(n_v + n_u)
    history, diag = [], []
    Psi = None
    for _ in range(max_iter):
        Psi, Phi = build_regression(segments, w_u_i, W, R.item(), dt)
        W_hat, *_ = np.linalg.lstsq(Psi, -Phi, rcond=None)
        w_v   = W_hat[:n_v]
        w_u_i = W_hat[n_v:]
        history.append(W_hat)

        # --- DIAGNOSTICS: resid = basis adequacy (can it fit V?); cond = data excitation.
        resid = np.linalg.norm(Psi @ W_hat + Phi)       # Bellman LS residual (~0 => basis fits V)
        s = np.linalg.svd(Psi, compute_uv=False)
        cond = s[0] / s[-1]                              # huge => a regressor under-excited (data problem)
        diag.append({"resid": resid, "cond": cond})

        if np.linalg.norm(W_hat - W_prev) < eps:
            break
        W_prev = W_hat
    return w_u_i, w_v, history, diag


# Deployed learning recipe (the robust, well-conditioned config from the OBS-12 robustness
# sweep): several MODERATE-theta initial conditions under plain model-based behavior (-Kx +
# probe, NO a3-feedforward). max|theta| ~ 0.46 stays in the stabilizable region, so the target
# policy's value function exists and the fit is clean: linear part ~5-7% from -K_tilde_lin,
# cubic ~ -3.7 (counters the aero cubic). cond flags bad draws (probe/IC choices -> ~5e6).
LEARN_ICS = [0.2, 0.3, 0.4, 0.45, -0.35]
LEARN_AMP = 0.15
LEARN_W   = 10
LEARN_TPE = 8.0


def learn_augmentation(a3, ICs=None, amp=LEARN_AMP, W=LEARN_W, T_PE=LEARN_TPE, stabilize=False):
    """Collect one rollout per initial condition and learn u~ via off-policy PI (Algorithm 1).
    Returns (w_u, w_v, history, diag, max_abs_theta). stabilize=False keeps data in the
    stabilizable region automatically (model-based -Kx regulates moderate theta)."""
    ICs = LEARN_ICS if ICs is None else ICs
    segments, max_abs_theta = [], 0.0
    for xs in ICs:
        X, U, d = collect_data(np.array([xs, 0.0]), a3, T_PE, dt, K, amp, stabilize=stabilize)
        segments.append((X, U))
        max_abs_theta = max(max_abs_theta, d["abs_theta_max"])
    w_u, w_v, history, diag = policy_iteration(segments, W, dt)
    return w_u, w_v, history, diag, max_abs_theta


def u_composite(w_u, x):
    # applied composite control: model-based -Kx plus learned augmentation u~ = w_u . phi_u(x)
    return (-(K @ x)).item() + float(w_u @ phi_u(x))


def rollout_cost(w_u, x0, a3, T=8.0):
    """True-plant cost J = integral (x'Qx + u^2 R) dt. w_u=None -> model-based -Kx only.
    Returns 1e6 on divergence (so it reads as 'failed to regulate')."""
    x = x0.astype(float).copy(); J = 0.0; Rs = R.item()
    for _ in range(int(T / dt)):
        u = u_composite(w_u, x) if w_u is not None else (-(K @ x)).item()
        J += (x @ Q @ x + u * u * Rs) * dt
        x = x + (f(x, a3) + g(x) * u) * dt
        if not np.all(np.isfinite(x)) or np.linalg.norm(x) > 1e3:
            return 1e6
    return J


def stable_envelope(w_u, a3, lo=0.2, hi=1.5, tol=1e-3):
    """Largest theta0 from which the ALWAYS-ON controller still regulates (bisection on divergence).
    w_u=None -> model-based -Kx; else the composite -Kx+u~ applied from t=0."""
    while hi - lo > tol:
        mid = (lo + hi) / 2.0
        if rollout_cost(w_u, np.array([mid, 0.0]), a3) < 1e5:
            lo = mid
        else:
            hi = mid
    return lo


def switched_envelope(w_u, a3, T=8.0, lo=0.2, hi=1.5, tol=1e-3):
    """Largest theta0 the DEPLOYED switched system regulates (model-based until t_s, then composite).
    This is tighter than the always-on composite envelope: during the pre-switch model-based phase the
    state grows, so the composite must still catch it -- at large theta0 it can't, and the switch
    timing erodes the achievable envelope. This is the operationally honest number."""
    def regulates(th0):
        x0 = np.array([th0, 0.0]); xmin = 0.05 * np.linalg.norm(x0)
        arr, *_rest = deploy_switched(w_u, x0, a3, T, xi, xmin)
        return np.all(np.isfinite(arr[-1])) and np.linalg.norm(arr[-1]) < 0.05
    while hi - lo > tol:
        mid = (lo + hi) / 2.0
        if regulates(mid):
            lo = mid
        else:
            hi = mid
    return lo


def deploy_switched(w_u, x0, a3, T, xi, x_min):
    """The actual deployed run: model-based -Kx until the switch (Eq.17 eta-test), then the
    composite -Kx + u~. Returns trajectories + t_s for the figures."""
    lenT = int(T / dt)
    arr = np.zeros((lenT, 2)); arr[0] = x0
    u_arr = np.zeros(lenT); eta = np.zeros((lenT, 2)); thr_arr = np.zeros(lenT)
    lam_max_Q = np.max(np.diag(Q)); lam_min_R = np.min(np.diag(R)); lam_max_R = np.max(np.diag(R))
    x = x0.astype(float).copy(); xm = x0.astype(float).copy()
    t_s = None; switched = False
    for i in range(1, lenT):
        if not np.all(np.isfinite(x)) or np.linalg.norm(x) > 1e3:
            arr[i:] = 1e3; break          # diverged (state blew up before the switch could catch it)
        u_m = (-(K @ x)).item()
        thr = (1 / (xi**2 * lam_max_R)) * (lam_max_Q * np.linalg.norm(x)**2 + lam_min_R * u_m**2)
        if not switched and np.linalg.norm(x - xm)**2 >= thr and np.linalg.norm(x) > x_min:
            switched = True; t_s = i
        u = u_composite(w_u, x) if switched else u_m
        thr_arr[i] = thr; u_arr[i] = u
        x  = x  + (f(x, a3) + g(x) * u) * dt
        xm = xm + (Am @ xm + Bm.flatten() * u_m) * dt
        arr[i] = x; eta[i] = x - xm
    return arr, u_arr, eta, np.linalg.norm(eta, axis=1), thr_arr, t_s


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

    print("\n=== Steps 3-4: off-policy IRL augmentation (deployed recipe) ===")
    print(f"  bases: phi_v = {len(phi_v(x0))} terms (QUARTIC critic), "
          f"phi_u = {len(phi_u(x0))} terms (odd cubic actor)")
    print(f"  learning from moderate-theta ICs {LEARN_ICS} (model-based behavior, no feedforward)")
    w_u, w_v, history, pidiag, max_th = learn_augmentation(a3_strong)
    Ktl = -K_tilde_lin.flatten()
    lin_err = np.linalg.norm(w_u[:2] - Ktl) / np.linalg.norm(Ktl) * 100
    print(f"  collection max|theta| = {max_th:.3f} (in the stabilizable region)")
    print(f"  PI: {len(history)} iters   resid = {pidiag[-1]['resid']:.3e}   cond = {pidiag[-1]['cond']:.2e}")
    print(f"  learned w_u (actor)  = {w_u}")
    print(f"  learned w_v (critic) = {w_v}")

    print("\n=== Step 5: validation ===")
    print(f"  (a) small-signal: w_u linear part {w_u[:2]} vs -K_tilde_lin {Ktl}  -> {lin_err:.1f}% error")
    print(f"  (b) cubic weight w_u[2] = {w_u[2]:+.3f}  (NEGATIVE => counters the +a3*theta^3 drift)")

    # (c) usefulness: composite vs model-based true-plant cost, swept over theta0.
    print("  (c) usefulness (true-plant cost J, composite vs model-based):")
    print(f"      {'theta0':>7} {'J_mb':>10} {'J_comp':>10}   winner")
    n_win = 0
    for th0 in [0.2, 0.3, 0.4, 0.5, 0.55]:
        Jmb = rollout_cost(None, np.array([th0, 0.0]), a3_strong)
        Jcp = rollout_cost(w_u, np.array([th0, 0.0]), a3_strong)
        win = Jcp < Jmb; n_win += win
        tag = "composite" if win else "model-based (origin: model exact, OBS-1)"
        print(f"      {th0:>7} {Jmb:>10.4f} {Jcp:>10.4f}   {tag}")
    env_mb  = stable_envelope(None, a3_strong)
    env_cmp = stable_envelope(w_u, a3_strong)
    env_sw  = switched_envelope(w_u, a3_strong)
    print(f"      -> composite wins {n_win}/5 (loses only near origin)")
    print(f"      STABLE ENVELOPE theta0:  model-based {env_mb:.2f}  |  composite(always-on) {env_cmp:.2f}"
          f"  |  DEPLOYED switched {env_sw:.2f} rad")
    print(f"      (deployed {env_sw:.2f} < always-on {env_cmp:.2f}: the pre-switch model-based phase lets"
          f" the state grow, so the switch timing erodes the envelope -- still +{env_sw-env_mb:.2f} rad vs model-based)")

    # (d) degenerate: a3=0 (perfect model) => augmentation should collapse to ~ -K_tilde_lin
    #     (the cubic ~0), confirming there is "nothing extra to learn" when the model is exact.
    w_u0, *_ = learn_augmentation(0.0)
    print(f"  (d) a3=0 (model exact): w_u = {w_u0}  (linear ~ -K_tilde_lin, cubic ~ 0)")

    # (e) OBS-12 regression guard: a single FAT large-theta rollout is POISON (target policy
    #     can't stabilize 60*theta^3 there -> V undefined -> garbage fit). Contrast vs deployed.
    Xp, Up, _ = collect_data(np.array([1.5, 0.0]), a3_strong, LEARN_TPE, dt, K, 0.3, stabilize=True)
    w_u_poison, _, _, dpoison = policy_iteration([(Xp, Up)], LEARN_W, dt)
    print(f"  (e) OBS-12 poison check: large-theta(1.5) fit w_u = {w_u_poison}  "
          f"(cubic sign wrong/weak) vs deployed cubic {w_u[2]:+.2f}")

    # ---- Figures (saved as PNG; gitignored) -------------------------------------------
    print("\n=== Figures ===")

    # Fig 1: the deployed SWITCHED run at the nominal kick x0=[0.3,0] (mechanism view).
    arr_sw, u_sw, eta_sw, etan_sw, thr_sw, t_s_idx = deploy_switched(w_u, x0, a3_strong, 8.0, xi, x_min)
    t = np.arange(len(arr_sw)) * dt
    ts = t_s_idx * dt if t_s_idx is not None else None
    fig1, ax = plt.subplots(3, 1, figsize=(7, 9))
    ax[0].plot(t, arr_sw[:, 0], label=r'$\theta$'); ax[0].plot(t, arr_sw[:, 1], label=r'$\dot\theta$')
    ax[0].set_ylabel('states [rad, rad/s]'); ax[0].legend(fontsize=8); ax[0].set_xlim(0, 8)
    ax[1].plot(t[1:], u_sw[1:], label='u (gimbal)'); ax[1].set_ylabel('control'); ax[1].legend(fontsize=8); ax[1].set_xlim(0, 8)
    ax[2].plot(t[1:], etan_sw[1:]**2, label=r'$\|\eta\|^2$'); ax[2].plot(t[1:], thr_sw[1:], label='threshold')
    ax[2].set_ylabel('model error'); ax[2].set_xlabel('t [s]'); ax[2].legend(fontsize=8)
    # zoom to the switching window: past ~0.6 s eta^2 is the OBS-4 monitor divergence, not mismatch
    _we = int(0.8 / dt)
    ax[2].set_xlim(0, 0.8); ax[2].set_ylim(0, max(thr_sw[1:_we].max(), (etan_sw[1:_we]**2).max()) * 1.2)
    for a in ax:
        if ts: a.axvline(ts, color='gray', ls=':')
    fig1.suptitle(f'Rocket pitch: deployed switched run (x0={x0}, a3={a3_strong}), t_s={ts:.2f}s' if ts
                  else 'Rocket pitch: deployed run')
    plt.tight_layout(); fig1.savefig('rocket_switched.png', dpi=110); plt.close(fig1)

    # Fig 2: weight convergence (critic & actor) vs PI iteration -> their final values.
    wv_f, wu_f = history[-1][:len(w_v)], history[-1][len(w_v):]
    ev = [np.linalg.norm(h[:len(w_v)] - wv_f) for h in history]
    eu = [np.linalg.norm(h[len(w_v):] - wu_f) for h in history]
    fig2, a2 = plt.subplots(figsize=(7, 4))
    a2.semilogy(range(len(ev)), np.array(ev) + 1e-16, 'o-', label=r'$\|\hat w_v^i - \hat w_v^\infty\|$')
    a2.semilogy(range(len(eu)), np.array(eu) + 1e-16, 's-', label=r'$\|\hat w_u^i - \hat w_u^\infty\|$')
    a2.set_xlabel('PI iteration'); a2.set_ylabel('weight error'); a2.legend(); a2.grid(True, which='both', alpha=0.3)
    a2.set_title('Rocket pitch: Algorithm-1 weight convergence')
    plt.tight_layout(); fig2.savefig('rocket_weights.png', dpi=110); plt.close(fig2)

    # Fig 3: usefulness -- true-plant cost vs theta0, composite vs model-based, with envelopes.
    th0s = np.linspace(0.2, 1.0, 33)
    Jmbs = [rollout_cost(None, np.array([th, 0.0]), a3_strong) for th in th0s]
    Jcps = [rollout_cost(w_u, np.array([th, 0.0]), a3_strong) for th in th0s]
    cap = lambda L: [min(j, 5.0) for j in L]   # cap diverged (1e6) for plotting
    fig3, a3p = plt.subplots(figsize=(7, 4.5))
    a3p.plot(th0s, cap(Jmbs), 'o-', label='model-based $-Kx$')
    a3p.plot(th0s, cap(Jcps), 's-', label='composite $-Kx+\\tilde u$')
    a3p.axvline(env_mb, color='C0', ls=':', label=f'mb envelope {env_mb:.2f}')
    a3p.axvline(env_sw, color='C2', ls='--', label=f'deployed switched env {env_sw:.2f}')
    a3p.axvline(env_cmp, color='C1', ls=':', label=f'composite(always-on) env {env_cmp:.2f}')
    a3p.set_xlabel(r'$\theta_0$ [rad]'); a3p.set_ylabel('true-plant cost J (capped at 5)')
    a3p.legend(fontsize=8); a3p.set_title('Rocket pitch: augmentation usefulness & stable-envelope extension')
    plt.tight_layout(); fig3.savefig('rocket_usefulness.png', dpi=110); plt.close(fig3)

    # Fig 4: OBS-11 critic-basis lesson -- hold actor (cubic), vary critic (quad vs quartic).
    import sys as _sys
    segs_fig = [collect_data(np.array([xs, 0.0]), a3_strong, LEARN_TPE, dt, K, LEARN_AMP)[:2]
                for xs in LEARN_ICS]
    _, _, _, d_q4 = policy_iteration(segs_fig, LEARN_W, dt); resid_q4 = d_q4[-1]['resid']
    _orig = phi_v; _sys.modules[__name__].phi_v = phi_v_quad
    _, _, _, d_q2 = policy_iteration(segs_fig, LEARN_W, dt); resid_q2 = d_q2[-1]['resid']
    _sys.modules[__name__].phi_v = _orig
    fig4, a4 = plt.subplots(figsize=(5, 4))
    a4.bar(['quadratic\ncritic', 'quartic\ncritic'], [resid_q2, resid_q4], color=['C3', 'C2'])
    a4.set_ylabel('final Bellman residual'); a4.set_yscale('log')
    a4.set_title('Rocket pitch (OBS-11): quartic critic fits V,\nquadratic cannot (actor held fixed)')
    plt.tight_layout(); fig4.savefig('rocket_critic_basis.png', dpi=110); plt.close(fig4)
    print(f"  saved: rocket_switched.png, rocket_weights.png, rocket_usefulness.png, rocket_critic_basis.png")
    print(f"  (critic-basis: quadratic resid {resid_q2:.2e} vs quartic {resid_q4:.2e})")
