"""rocket-pitch-burn.py -- Tier 2: mass burn (time-varying pitch dynamics)

As propellant burns, inertia I(t) drops, so the aero-instability coefficient
a1(t) = C_aero / I(t) grows over the burn while control authority b is held
constant. The nominal model stays frozen at its t=0 value, so the mismatch is
a linear term (a1(t) - a1_0)*theta that does not vanish at the origin -- unlike
Tier 1's cubic term, which does.

Central question: is one switch plus one learned augmentation enough, or does
drift make the learned correction go stale over the burn? See
TIER2-EXPLANATION.md for the full writeup of what was found.
"""

from scipy.linalg import solve_continuous_are
import numpy as np
import matplotlib.pyplot as plt

# Burn profile
T_burn = 10.0       # total burn duration [s]
a1_0   = 4.0        # initial pitch instability coefficient
b0     = 8.0        # TVC authority -- constant through burn (see docstring)


def burn_params(t):
    """True plant parameters at burn time t.
    I(t) drops to I0/3 at burnout; a1(t) triples; b(t) = b0 (constant)."""
    frac   = np.clip(t, 0.0, T_burn) / T_burn
    I_norm = 1.0 - (2.0 / 3.0) * frac    # 1 at t=0, 1/3 at t=T_burn
    return a1_0 / I_norm, b0              # (a1_t, b_t)


# Nominal (frozen) model
Am = np.array([[0.0, 1.0],
               [a1_0, 0.0]])
Bm = np.array([[0.0], [b0]])

Q  = np.diag([1.0, 0.1])
R  = np.array([[0.2]])

Pm = solve_continuous_are(Am, Bm, Q, R)
K  = np.linalg.inv(R) @ Bm.T @ Pm        # shape (1, 2)


# True plant dynamics (time-varying during burn; LTI when frozen at t_freeze)

def f_burn(x, t):
    """True drift at burn time t (no cubic; purely a1(t) mismatch)."""
    a1_t, _ = burn_params(t)
    return np.array([x[1], a1_t * x[0]])


def g_burn():
    """Input vector (b = constant)."""
    return np.array([0.0, b0])


# Analytic target: optimal augmentation gain at a frozen burn time

def K_tilde_at(t_freeze):
    """Solve the augmented ARE for the LTI plant frozen at t_freeze.
    Augmented plant: A_aug = A_true(t_freeze) - Bm @ K
    Augmented cost:  integral(x'Qx + u~'Ru~) dt
    Returns K_tilde, shape (1,2).  True w_u* = -K_tilde.flatten()."""
    a1_t, _ = burn_params(t_freeze)
    A_aug    = np.array([[0.0, 1.0], [a1_t, 0.0]]) - Bm @ K
    P_tilde  = solve_continuous_are(A_aug, Bm, Q, R)
    return np.linalg.inv(R) @ Bm.T @ P_tilde    # shape (1, 2)


# Default sim parameters (same grid as Tier 1)
x0    = np.array([0.3, 0.0])
dt    = 0.01
xi    = 1.35
x_min = 0.05 * np.linalg.norm(x0)


# Step 2: switching with time-varying plant

def sim_switching_burn(x0, T, dt, K, xi, x_min):
    """Euler sim with time-varying plant.  First-crossing switch (Eq. 21).
    Model monitor sees u_m = -Kx only (paper convention).
    Returns (arr, xm_arr, eta, eta_norm, thr_arr, u_arr, t_s)
    where t_s is the switch step (None if no switch)."""
    lenT    = int(T / dt)
    arr     = np.zeros((lenT, 2)); arr[0]     = x0
    xm_arr  = np.zeros((lenT, 2)); xm_arr[0]  = x0
    eta     = np.zeros((lenT, 2))
    u_arr   = np.zeros(lenT)
    thr_arr = np.zeros(lenT)

    lam_max_Q = np.max(np.diag(Q))
    lam_min_R = np.min(np.diag(R))
    lam_max_R = np.max(np.diag(R))

    x   = x0.astype(float).copy()
    xm  = x0.astype(float).copy()
    t_s = None

    for i in range(1, lenT):
        t   = (i - 1) * dt
        u_m = float(-(K @ x).item())
        thr = (1.0 / (xi**2 * lam_max_R)) * (
              lam_max_Q * float(np.linalg.norm(x))**2
              + lam_min_R * u_m**2)

        if (t_s is None
                and float(np.linalg.norm(x - xm))**2 >= thr
                and float(np.linalg.norm(x)) > x_min):
            t_s = i

        thr_arr[i] = thr
        u_arr[i]   = u_m
        x   = x  + (f_burn(x, t) + g_burn() * u_m) * dt
        xm  = xm + (Am @ xm + Bm.flatten() * u_m) * dt
        arr[i]    = x
        xm_arr[i] = xm
        eta[i]    = x - xm

    return arr, xm_arr, eta, np.linalg.norm(eta, axis=1), thr_arr, u_arr, t_s


# RL bases: quadratic critic + linear actor. V* is exactly quadratic for any
# frozen-time LTI plant, so no quartic/cubic basis is needed here (contrast Tier 1).

def phi_v_burn(x):
    """Quadratic critic: [theta^2, theta*thetadot, thetadot^2] (3 terms)."""
    th, thd = float(x[0]), float(x[1])
    return np.array([th**2, th * thd, thd**2])


def phi_u_burn(x):
    """Linear actor: [theta, thetadot] (2 terms).
    phi_u(-x) = -phi_u(x): ODD parity preserved (linear system is odd-symmetric)."""
    return np.array([float(x[0]), float(x[1])])


# Off-policy IRL regression (Algorithm 1 adapted to Tier-2 bases)

def build_regression_burn(segments, w_u_i, W, R_scalar, dt):
    """Build the off-policy regression matrix.
    segments: list of (X_data, U_data) from collect_data_burn.
    W: window length; windows never straddle IC boundaries."""
    n_v  = len(phi_v_burn(segments[0][0][0]))
    n_u  = len(phi_u_burn(segments[0][0][0]))
    rows, costs = [], []
    for X_data, U_data in segments:
        k = 0
        while k + W < len(X_data):
            psi_v    = phi_v_burn(X_data[k + W]) - phi_v_burn(X_data[k])
            psi_u    = np.zeros(n_u)
            phi_cost = 0.0
            for j in range(k, k + W):
                mu_ij     = float(w_u_i @ phi_u_burn(X_data[j]))
                diff      = float(U_data[j]) - mu_ij
                psi_u    += 2.0 * diff * R_scalar * phi_u_burn(X_data[j]) * dt
                phi_cost += (X_data[j] @ Q @ X_data[j]
                             + mu_ij**2 * R_scalar) * dt
            rows.append(np.concatenate([psi_v, psi_u]))
            costs.append(phi_cost)
            k += W
    return np.array(rows), np.array(costs)


def policy_iteration_burn(segments, W, dt, eps=1e-6, max_iter=80):
    """Algorithm 1 PI loop.  Returns (w_u, w_v, history, diag).
    Convergence expected in 1-2 iterations (V* exactly quadratic => exact fit)."""
    n_v    = len(phi_v_burn(segments[0][0][0]))
    n_u    = len(phi_u_burn(segments[0][0][0]))
    w_u_i  = np.zeros(n_u)
    W_prev = np.zeros(n_v + n_u)
    history, diag = [], []
    for _ in range(max_iter):
        Psi, Phi = build_regression_burn(segments, w_u_i, W, R.item(), dt)
        W_hat, *_ = np.linalg.lstsq(Psi, -Phi, rcond=None)
        w_u_i = W_hat[n_v:]
        history.append(W_hat.copy())
        s = np.linalg.svd(Psi, compute_uv=False)
        diag.append({
            "resid": float(np.linalg.norm(Psi @ W_hat + Phi)),
            "cond":  float(s[0] / (s[-1] + 1e-30)),
        })
        if float(np.linalg.norm(W_hat - W_prev)) < eps:
            break
        W_prev = W_hat.copy()
    return w_u_i, W_hat[:n_v], history, diag


# Data collection from the FROZEN plant at t_freeze (LTI, no cubic cliff)

BURN_LEARN_ICS = [0.15, 0.25, 0.35, -0.2, -0.35]
BURN_LEARN_AMP = 0.1
BURN_LEARN_W   = 10
BURN_LEARN_TPE = 8.0


def collect_data_burn(x_start, t_freeze, T_PE, dt, K, amp, freqs=None, phase=2.3):
    """Collect rollout from the LTI plant FROZEN at t_freeze.
    Behavior policy: u = -Kx + probe (model-based + exploration).
    Only probe is stored in U (off-policy)."""
    if freqs is None:
        freqs = np.array([1.0, 2.3, 3.7, 5.1, 7.9, 11.3])

    def probe(tau):
        return amp * float(np.sum(np.sin(freqs * tau + phase)))

    N  = int(T_PE / dt)
    X  = np.zeros((N, 2))
    U  = np.zeros(N)
    x  = x_start.astype(float).copy()
    a1_t, b_t = burn_params(t_freeze)
    diverged   = False
    for k in range(N):
        u_m  = float(-(K @ x).item())
        ut   = probe(k * dt)
        X[k] = x
        U[k] = ut
        u    = u_m + ut
        x    = x + np.array([x[1], a1_t * x[0] + b_t * u]) * dt
        if not np.all(np.isfinite(x)) or float(np.linalg.norm(x)) > 1e3:
            diverged = True
            X, U = X[:k + 1], U[:k + 1]
            break
    return X, U, {"abs_theta_max": float(np.abs(X[:, 0]).max()), "diverged": diverged}


def learn_augmentation_burn(t_freeze,
                             ICs=None, amp=BURN_LEARN_AMP,
                             W=BURN_LEARN_W, T_PE=BURN_LEARN_TPE):
    """Learn u~ for the plant FROZEN at t_freeze.
    Returns (w_u, w_v, history, diag, max_abs_theta)."""
    ICs = BURN_LEARN_ICS if ICs is None else ICs
    segments, max_th = [], 0.0
    for xs in ICs:
        X, U, d = collect_data_burn(np.array([xs, 0.0]), t_freeze, T_PE, dt, K, amp)
        segments.append((X, U))
        max_th = max(max_th, d["abs_theta_max"])
    w_u, w_v, history, pdiag = policy_iteration_burn(segments, W, dt)
    return w_u, w_v, history, pdiag, max_th


# Composite control + deployed snapshot cost (the staleness diagnostic)

def u_composite_burn(w_u, x):
    """u = -Kx + w_u . phi_u(x)."""
    return float(-(K @ x).item()) + float(w_u @ phi_u_burn(x))


def snapshot_cost(w_u, t_freeze, x0_snap=None, T_snap=5.0):
    """Deployed regulator cost on the FROZEN-at-t_freeze plant starting from x0_snap.
    Applies model-based control until the switch fires, then composite.
    w_u=None: model-based only (no switch needed).
    This is the key staleness diagnostic: compare J(w_u_early, t_late) vs J(w_u_late, t_late)."""
    if x0_snap is None:
        x0_snap = x0
    x   = x0_snap.astype(float).copy()
    xm  = x0_snap.astype(float).copy()
    J   = 0.0
    Rs  = R.item()
    a1_t, b_t   = burn_params(t_freeze)
    switched     = False
    lam_max_Q    = float(np.max(np.diag(Q)))
    lam_min_R    = float(np.min(np.diag(R)))
    lam_max_R    = float(np.max(np.diag(R)))
    x_min_snap   = 0.05 * float(np.linalg.norm(x0_snap))

    for _ in range(int(T_snap / dt)):
        u_m  = float(-(K @ x).item())
        thr  = (1.0 / (xi**2 * lam_max_R)) * (
               lam_max_Q * float(np.linalg.norm(x))**2
               + lam_min_R * u_m**2)
        if (not switched
                and float(np.linalg.norm(x - xm))**2 >= thr
                and float(np.linalg.norm(x)) > x_min_snap):
            switched = True
        if switched and w_u is not None:
            u = u_composite_burn(w_u, x)
        else:
            u = u_m
        J  += (x @ Q @ x + u * u * Rs) * dt
        x   = x  + np.array([x[1], a1_t * x[0] + b_t * u]) * dt
        xm  = xm + (Am @ xm + Bm.flatten() * u_m) * dt
        if not np.all(np.isfinite(x)) or float(np.linalg.norm(x)) > 1e3:
            return 1e6
    return J


# Full burn simulation (time-varying plant, single switch + composite after)

def sim_full_burn(w_u_fn, x0, T, dt, K, xi, x_min):
    """Full burn sim with time-varying plant.
    w_u_fn: None -> model-based only
             callable(t) -> w_u weights (applied after switch fires)
    Returns (arr, u_arr, eta_norm, thr_arr, switch_step)."""
    lenT     = int(T / dt)
    arr      = np.zeros((lenT, 2)); arr[0] = x0
    u_arr    = np.zeros(lenT)
    eta      = np.zeros((lenT, 2))
    thr_arr  = np.zeros(lenT)

    lam_max_Q = float(np.max(np.diag(Q)))
    lam_min_R = float(np.min(np.diag(R)))
    lam_max_R = float(np.max(np.diag(R)))

    x    = x0.astype(float).copy()
    xm   = x0.astype(float).copy()
    sw   = False
    sw_i = None

    for i in range(1, lenT):
        t   = (i - 1) * dt
        u_m = float(-(K @ x).item())
        thr = (1.0 / (xi**2 * lam_max_R)) * (
              lam_max_Q * float(np.linalg.norm(x))**2
              + lam_min_R * u_m**2)

        if (not sw
                and float(np.linalg.norm(x - xm))**2 >= thr
                and float(np.linalg.norm(x)) > x_min):
            sw   = True
            sw_i = i

        if sw and w_u_fn is not None:
            u = u_composite_burn(w_u_fn(t), x)
        else:
            u = u_m

        thr_arr[i] = thr
        u_arr[i]   = u
        x   = x  + (f_burn(x, t) + g_burn() * u) * dt
        xm  = xm + (Am @ xm + Bm.flatten() * u_m) * dt
        arr[i]  = x
        eta[i]  = x - xm

    return arr, u_arr, np.linalg.norm(eta, axis=1), thr_arr, sw_i


# Validates the quasi-static (snapshot) assumption against the genuinely
# time-varying plant: start from x0 late in the burn and let the plant drift
# during the regulation transient, then compare to snapshot_cost at the same
# frozen time. If the benefits are similar, snapshot is a valid proxy.

def sim_late_burn_cost(w_u, t_start=8.0, T_sim=4.0):
    """Regulation cost starting at t_start on the genuinely time-varying plant
    (same logic as snapshot_cost, but f_burn uses the real drifting time).
    Compare to snapshot_cost(w_u, t_freeze=10.0, T_snap=T_sim) -- similar
    benefit % means the quasi-static approximation is sound."""
    x    = x0.astype(float).copy()
    xm   = x0.astype(float).copy()
    J    = 0.0
    Rs   = R.item()
    lam_max_Q = float(np.max(np.diag(Q)))
    lam_min_R = float(np.min(np.diag(R)))
    lam_max_R = float(np.max(np.diag(R)))
    x_min_l   = 0.05 * float(np.linalg.norm(x0))
    sw = False

    for k in range(int(T_sim / dt)):
        t_abs = t_start + k * dt
        u_m   = float(-(K @ x).item())
        thr   = (1.0 / (xi**2 * lam_max_R)) * (
                lam_max_Q * float(np.linalg.norm(x))**2
                + lam_min_R * u_m**2)

        if (not sw
                and float(np.linalg.norm(x - xm))**2 >= thr
                and float(np.linalg.norm(x)) > x_min_l):
            sw = True

        if sw and w_u is not None:
            u = float(-(K @ x).item()) + float(w_u @ phi_u_burn(x))
        else:
            u = u_m

        J  += (x @ Q @ x + u**2 * Rs) * dt
        x   = x  + (f_burn(x, t_abs) + g_burn() * u) * dt
        xm  = xm + (Am @ xm + Bm.flatten() * u_m) * dt
        if not np.all(np.isfinite(x)):
            return 1e6

    return J


# Main: diagnostics + staleness study + figures

if __name__ == "__main__":
    print("=== Rocket Pitch Tier 2: MASS BURN ===\n")

    # Burn profile + pole drift
    print("--- Burn profile (a1 triples; b constant; frozen K) ---")
    print(f"  {'t':>5}  {'a1':>6}  {'slow_pole':>10}  {'model_err/theta':>16}")
    for t in [0.0, 2.5, 5.0, 7.5, 10.0]:
        a1_t, _ = burn_params(t)
        A_t     = np.array([[0.0, 1.0], [a1_t, 0.0]])
        slow    = float(np.sort(np.linalg.eigvals(A_t - Bm @ K).real)[-1])
        print(f"  {t:>5.1f}  {a1_t:>6.2f}  {slow:>10.3f}  {a1_t - a1_0:>16.2f}")

    # Switching
    print("\n--- Switching with time-varying plant (T=6 s) ---")
    _, _, _, eta_nrm, thr, _, t_s = sim_switching_burn(x0, 6.0, dt, K, xi, x_min)
    t_s_s = None if t_s is None else t_s * dt
    print(f"  t_s = {t_s_s} s  (OBS-13: IC-independent for linear mismatch)")

    # Learn u~ at three freeze times
    freeze_times = [0.0, 5.0, 10.0]
    print("\n--- Learn u~ at t_freeze in {0, 5, 10} s ---")
    w_us = {}
    for tf in freeze_times:
        w_u, w_v, hist, pdiag, max_th = learn_augmentation_burn(tf)
        Ktl     = K_tilde_at(tf).flatten()
        w_u_tgt = -Ktl                    # true optimal: w_u* = -K_tilde
        err     = (float(np.linalg.norm(w_u - w_u_tgt))
                   / float(np.linalg.norm(w_u_tgt))) * 100.0
        print(f"  t_freeze={tf:.0f}s:  w_u={w_u}  target={w_u_tgt}"
              f"  err={err:.1f}%  resid={pdiag[-1]['resid']:.2e}"
              f"  cond={pdiag[-1]['cond']:.1e}")
        w_us[tf] = w_u

    # Staleness study: snapshot cost at 11 eval points
    print("\n--- Staleness: deployed snapshot cost J(w_u, t_freeze) ---")
    eval_ts = np.linspace(0.0, T_burn, 11)
    J_mb  = [snapshot_cost(None,       t) for t in eval_ts]
    J_t0  = [snapshot_cost(w_us[0.0],  t) for t in eval_ts]
    J_t5  = [snapshot_cost(w_us[5.0],  t) for t in eval_ts]
    J_t10 = [snapshot_cost(w_us[10.0], t) for t in eval_ts]
    print(f"  {'t':>5}  {'J_mb':>8}  {'J_t0':>8}  {'J_t5':>8}  {'J_t10':>8}")
    for i, t in enumerate(eval_ts):
        print(f"  {t:>5.1f}  {J_mb[i]:>8.4f}  {J_t0[i]:>8.4f}"
              f"  {J_t5[i]:>8.4f}  {J_t10[i]:>8.4f}")
    print(f"\n  At t=10: composite benefit (vs mb): {(J_mb[-1]-J_t10[-1])/J_mb[-1]*100:.1f}%"
          f"  staleness gap (t0 vs t10): {(J_t0[-1]-J_t10[-1])/J_mb[-1]*100:.1f}%")

    # Full burn sim
    arr_fb, u_fb, eta_fb, thr_fb, sw_fb = sim_full_burn(
        lambda t: w_us[0.0], x0, T_burn, dt, K, xi, x_min)
    sw_s = None if sw_fb is None else sw_fb * dt
    print(f"\n--- Full burn sim (u~_t0 applied after switch): t_s={sw_s} s,"
          f"  final state={arr_fb[-1]} ---")

    # ----------------------------------------------------------------
    # Figures
    # ----------------------------------------------------------------
    t_arr = np.arange(len(arr_fb)) * dt

    # Fig 1: Burn profile + pole drift
    ts_p  = np.linspace(0.0, T_burn, 201)
    a1s   = [burn_params(t)[0] for t in ts_p]
    poles_slow = [float(np.sort(
                  np.linalg.eigvals(np.array([[0,1],[a1,0]]) - Bm @ K).real)[-1])
                  for a1 in a1s]
    fig1, ax1 = plt.subplots(2, 1, figsize=(7, 5))
    ax1[0].plot(ts_p, a1s, lw=2, label=r'$a_1(t)$ true')
    ax1[0].axhline(a1_0, ls=':', color='gray', label=r'model $a_{1,0}$')
    ax1[0].set_ylabel(r'$a_1$ [1/s$^2$]'); ax1[0].legend(fontsize=9)
    ax1[0].text(0.05, 0.5, 'Linear error = $(a_1(t)-a_{1,0})\\theta$\nnon-zero at origin',
                transform=ax1[0].transAxes, fontsize=8)
    ax1[1].plot(ts_p, poles_slow, 'C1', lw=2)
    ax1[1].set_ylabel('slow cl-pole'); ax1[1].set_xlabel('burn time [s]')
    ax1[1].text(0.05, 0.15,
                'Slow pole drifts -3.46 -> -1.41\nregulation slows; cost grows',
                transform=ax1[1].transAxes, fontsize=8)
    fig1.suptitle('Rocket pitch Tier-2: burn profile (a1 triples, b constant)')
    plt.tight_layout(); plt.show()

    # Fig 2: Staleness cost curves
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.plot(eval_ts, J_mb,  'k-o',  label='model-based $-Kx$')
    ax2.plot(eval_ts, J_t0,  'C0-s', label=r'composite $\tilde u_{t=0}$ (early)')
    ax2.plot(eval_ts, J_t5,  'C1-^', label=r'composite $\tilde u_{t=5}$ (mid)')
    ax2.plot(eval_ts, J_t10, 'C2-D', label=r'composite $\tilde u_{t=10}$ (late)')
    ax2.set_xlabel('Burn time [s]'); ax2.set_ylabel('Snapshot cost J')
    ax2.legend(fontsize=9)
    ax2.set_title('Rocket pitch Tier-2: staleness\n'
                  r'(each $\tilde u$ learned from frozen LTI plant at its $t_\mathrm{freeze}$)')
    plt.tight_layout(); plt.show()

    # Fig 3: Full burn sim (states, control, eta pre-switch)
    fig3, axes3 = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
    axes3[0].plot(t_arr, arr_fb[:, 0], label=r'$\theta$')
    axes3[0].plot(t_arr, arr_fb[:, 1], label=r'$\dot\theta$')
    if sw_s:
        for a in axes3:
            a.axvline(sw_s, color='gray', ls=':', alpha=0.6)
    axes3[0].legend(fontsize=8); axes3[0].set_ylabel('states [rad, rad/s]')
    axes3[1].plot(t_arr[1:], u_fb[1:])
    axes3[1].set_ylabel('control [rad/s²]')
    zoom = min(int((sw_s + 1.5) / dt) if sw_s else len(t_arr), len(t_arr))
    axes3[2].plot(t_arr[1:zoom], eta_fb[1:zoom]**2, label=r'$\|\eta\|^2$')
    axes3[2].plot(t_arr[1:zoom], thr_fb[1:zoom], label='threshold', ls='--')
    axes3[2].legend(fontsize=8); axes3[2].set_ylabel('model error'); axes3[2].set_xlabel('t [s]')
    axes3[2].text(0.02, 0.85,
                  'eta shown pre-switch only (OBS-4: A_m unstable -> x_m diverges after switch)',
                  transform=axes3[2].transAxes, fontsize=7)
    sw_lbl = f'{sw_s:.2f}' if sw_s else 'None'
    fig3.suptitle(f'Rocket pitch Tier-2: full burn sim, t_s={sw_lbl} s\n'
                  r'(model-based before switch; composite $\tilde u_{t=0}$ after)')
    plt.tight_layout(); plt.show()

    # Fig 4: PI convergence at t_freeze=0 and t_freeze=10
    fig4, ax4 = plt.subplots(figsize=(7, 4))
    for tf, style in [(0.0, 'C0'), (10.0, 'C2')]:
        w_u, w_v, history, _, _ = learn_augmentation_burn(tf)
        n_v = len(phi_v_burn(np.zeros(2)))
        wf  = history[-1]
        eu  = [float(np.linalg.norm(h[n_v:] - wf[n_v:])) for h in history]
        ax4.semilogy(range(len(eu)), [e + 1e-16 for e in eu],
                     f'{style}-s', label=f't_freeze={tf:.0f} s')
    ax4.set_xlabel('PI iteration'); ax4.set_ylabel('actor weight change')
    ax4.legend(fontsize=9); ax4.grid(True, which='both', alpha=0.3)
    ax4.set_title('Rocket pitch Tier-2: Algorithm-1 convergence\n'
                  '(quadratic V* exact => 1-2 iterations)')
    plt.tight_layout(); plt.show()

    # ----------------------------------------------------------------
    # Late-burn scenario: validates quasi-static (snapshot) assumption
    # a1 drifts from 9.3 -> 12 during the regulation transient (t_start=8s)
    # ----------------------------------------------------------------
    print("\n--- Late-burn quasi-static validation (t_start=8 s, T_sim=4 s) ---")
    print("  Timescale sep: T_burn=10 s >> tau_reg ~0.7 s (slow pole -1.41) -> ratio ~14x")
    J_late_mb    = sim_late_burn_cost(None)
    J_late_stale = sim_late_burn_cost(w_us[0.0])    # u~ designed at t=0 (stale)
    J_late_fresh = sim_late_burn_cost(w_us[10.0])   # u~ designed at t=10 (fresh)
    J_snap_mb    = snapshot_cost(None,       10.0, T_snap=4.0)
    J_snap_fresh = snapshot_cost(w_us[10.0], 10.0, T_snap=4.0)
    ben_late   = (J_late_mb - J_late_fresh) / J_late_mb * 100.0
    ben_snap   = (J_snap_mb - J_snap_fresh) / J_snap_mb * 100.0
    print(f"  Drifting plant: J_mb={J_late_mb:.4f}  J_stale={J_late_stale:.4f}"
          f"  J_fresh={J_late_fresh:.4f}  benefit={ben_late:.1f}%")
    print(f"  Frozen  plant: J_mb={J_snap_mb:.4f}  J_fresh={J_snap_fresh:.4f}"
          f"  benefit={ben_snap:.1f}%")
    print(f"  Quasi-static check: drifting benefit {ben_late:.1f}% vs frozen {ben_snap:.1f}%"
          f"  (should be comparable)")

    # Fig 5: late-burn quasi-static validation
    fig5, ax5 = plt.subplots(figsize=(7, 4))
    labels  = ['MB\n(drifting)', 'Stale $u_{t0}$\n(drifting)', 'Fresh $u_{t10}$\n(drifting)',
               'MB\n(frozen t=10)', 'Fresh $u_{t10}$\n(frozen t=10)']
    vals    = [J_late_mb, J_late_stale, J_late_fresh, J_snap_mb, J_snap_fresh]
    colors  = ['k', 'C0', 'C2', 'gray', 'C3']
    bars    = ax5.bar(labels, vals, color=colors, alpha=0.75)
    ax5.set_ylabel('Regulation cost J (4 s transient)')
    ax5.set_title('Quasi-static validation: late-burn drifting vs frozen-t=10 snapshot\n'
                  r'(start $t_0=8\,s$; $a_1$ drifts $9.3\to12$ during transient)')
    for bar, v in zip(bars, vals):
        ax5.text(bar.get_x() + bar.get_width()/2, v, f'{v:.4f}',
                 ha='center', va='bottom', fontsize=8)
    plt.tight_layout(); plt.show()
