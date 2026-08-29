from scipy.linalg import solve_continuous_are
import numpy as np
import matplotlib.pyplot as plt


# Tier 1: single-axis rocket pitch attitude control via thrust-vector control (TVC).
# Same setup as the paper's perturbed-oscillator example, generalized to a finless,
# aerodynamically-unstable airframe stabilized by gimballing. Model and plant share
# the same state dimension (Assumption 1); the model is missing a cubic aero term.
#
# x = [theta, theta_dot] (pitch angle, pitch rate), u = gimbal deflection [rad]
# xdot = f(x) + g(x)*u
#
# a1 = linear aero pitching-moment slope / inertia (open-loop unstable: a1 > 0)
# a3 = cubic aero moment (Duffing-like) / inertia -- the part the model doesn't see
# b  = TVC control authority (thrust * moment arm / inertia)
a1 = 4.0    # poles at +/- sqrt(a1) = +/- 2 rad/s
b  = 8.0

# a3 is the scenario knob: same A_m/K in both cases, only the perturbation changes.
a3_mild   = 0.5     # mismatch stays under threshold, no switch
a3_strong = 60.0    # mismatch grows, switch fires (~0.54 s), then RL kicks in


def f(x, a3):
    # true nonlinear drift: linear aero + cubic Duffing-like aero moment
    theta, theta_dot = x
    return np.array([theta_dot,
                     a1 * theta + a3 * theta**3])


def g(x):
    # control vector field: constant, gimbal torque enters the rate equation
    return np.array([0.0, b])



# Available model: linear aero only. g(x) is constant so Bm = g exactly -- all
# model error is the omitted cubic in f.
Am = np.array([[0.0, 1.0],
               [a1,  0.0]])
Bm = np.array([[0.0],
               [b]])

Q = np.diag([1.0, 0.1])   # penalize pitch angle; modest rate/control
R = np.array([[0.2]])

# Model-based LQR gain (Eq. 11): u_m = -Kx
Pm = solve_continuous_are(Am, Bm, Q, R)
K  = np.linalg.inv(R) @ Bm.T @ Pm     # shape (1, 2)

# Near-origin ground truth for later validation: a3*theta^3 has zero Jacobian at
# the origin, so the linearized true plant equals Am exactly there, and the
# optimal augmentation gain is just the section-7.1 augmented-Riccati solution
# with A -> Am. ũ(x) should match -K_tilde_lin x for small theta.
P_tilde = solve_continuous_are(Am - Bm @ K, Bm, Q, R)
K_tilde_lin = np.linalg.inv(R) @ Bm.T @ P_tilde   # shape (1, 2)

# Structural checks: open-loop instability + controllability of (Am, Bm).
eig_open = np.linalg.eigvals(Am)
ctrb = np.hstack([Bm, Am @ Bm])           # [B_m, A_m B_m]
ctrb_rank = np.linalg.matrix_rank(ctrb)
ctrb_det = np.linalg.det(ctrb)

# closed-loop poles under the model-based gain (sanity: should be stable)
eig_closed = np.linalg.eigvals(Am - Bm @ K)


# Switching: true plant (full nonlinear f) and model trajectory x_m (linear Am, Bm)
# are integrated under the same input; mismatch eta = x - x_m (Eq. 14). The switch
# fires (Theorem 1 / Eq. 17 & 40) the first instant ||eta||^2 >= threshold and
# ||x|| > x_min. No augmentation is applied here -- this just locates t_s.

x0 = np.array([0.3, 0.0])   # ~17 deg pitch disturbance, zero rate
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
    """Integral/dwell-time switching: instead of switching at the first instant
    ||eta||^2 >= threshold (Eq. 21), switch once the accumulated violation
        integral(max(0, ||eta||^2 - thr) dt
    exceeds tolerance tau. tau -> 0+ recovers the paper's first-crossing rule; tau
    gives a second, more direct lever on t_s than xi alone. Returns switch index or None.

    Caveat: once the plant regulates (~0.6 s here), ||eta|| is mostly the unstable
    model monitor x_m diverging rather than genuine mismatch, so large tau integrates
    that artifact rather than real suboptimality.
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


# Off-policy integral-RL augmentation (Algorithm 1). The system is odd-symmetric
# ((th,thd) -> (-th,-thd), cost even), so V is even (phi_v has only even monomials)
# and u~ is odd (phi_u has only odd monomials).

def phi_v_quad(x):
    # Quadratic critic. V* is actually NOT quadratic here -- the cubic drift seeds
    # quartic terms in the augmented HJB -- so this basis leaves a residual at large
    # theta. Kept only for the quad-vs-quartic comparison figure.
    th, thd = x
    return np.array([th**2, th*thd, thd**2])


def phi_v(x):
    # Deployed critic: quadratic core plus the two quartic terms the cubic drift
    # sources (th^4, th^3*thd), which fits V* cleanly (residual ~4x lower than quadratic).
    th, thd = x
    return np.array([th**2, th*thd, thd**2, th**4, th**3 * thd])


def phi_u(x):
    # Actor basis: odd, cubic-capable. Linear part recovers K_tilde_lin near the
    # origin; th^3 counters the cubic aero moment.
    th, thd = x
    return np.array([th, thd, th**3])


def collect_data(x_start, a3, T_PE, dt, K, amp, stabilize=False, freqs=None, phase=2.3):
    """Exploratory rollout on the true plant under a behavior policy, recording state
    and the behavior augmentation (the part on top of u_m = -Kx) for off-policy IRL.

    stabilize=False: ut = probe (model-based behavior u = -Kx + probe)
    stabilize=True:  ut = -(a3/b)*theta^3 + probe (rough cubic feedforward + probe)

    With stabilize=False, -Kx alone can't hold large theta (it diverges around
    |theta|~0.48 at a3=60), so the cubic is barely excitable there; stabilize=True
    cancels the cubic to reach large-theta data at the cost of the feedforward
    already "knowing" a3.
    """
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
    # Off-policy IRL regression. `segments` is a list of (X, U) rollouts; windows are
    # built within each segment so a Bellman window never straddles two trajectories.
    # This lets us learn from several moderate-theta ICs at once instead of one large
    # rollout, which is unstable to fit since the target policy can't stabilize there.
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


# Deployed learning recipe: several moderate-theta ICs under plain model-based
# behavior (-Kx + probe, no feedforward). max|theta| ~0.46 stays in the stabilizable
# region, so the fit is clean: linear part within ~5-7% of -K_tilde_lin, cubic ~-3.7.
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
    plt.tight_layout(); plt.show()

    # Fig 2: weight convergence (critic & actor) vs PI iteration -> their final values.
    wv_f, wu_f = history[-1][:len(w_v)], history[-1][len(w_v):]
    ev = [np.linalg.norm(h[:len(w_v)] - wv_f) for h in history]
    eu = [np.linalg.norm(h[len(w_v):] - wu_f) for h in history]
    fig2, a2 = plt.subplots(figsize=(7, 4))
    a2.semilogy(range(len(ev)), np.array(ev) + 1e-16, 'o-', label=r'$\|\hat w_v^i - \hat w_v^\infty\|$')
    a2.semilogy(range(len(eu)), np.array(eu) + 1e-16, 's-', label=r'$\|\hat w_u^i - \hat w_u^\infty\|$')
    a2.set_xlabel('PI iteration'); a2.set_ylabel('weight error'); a2.legend(); a2.grid(True, which='both', alpha=0.3)
    a2.set_title('Rocket pitch: Algorithm-1 weight convergence')
    plt.tight_layout(); plt.show()

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
    plt.tight_layout(); plt.show()

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
    plt.tight_layout(); plt.show()
    print(f"  (critic-basis: quadratic resid {resid_q2:.2e} vs quartic {resid_q4:.2e})")
