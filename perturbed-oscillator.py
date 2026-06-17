from scipy.linalg import solve_continuous_are
import numpy as np
import matplotlib.pyplot as plt
import math

def f(x, rho):
    arr = []
    arr.append(x[1])
    arr.append(-x[0] - rho * x[1] * (1/2 - 1/2*x[0]**2 - x[0]))
    return np.array(arr)

def g(x, rho):
    arr = []
    arr.append(0)
    arr.append(rho * x[0] + (1 - rho))
    return np.array(arr)

x0 = np.array([1.75, -1])
dt = 0.01
xi = 2.5
x_min = 0.1 * np.linalg.norm(np.array([1.75, -1]))
T_PE = 15.0

Am = np.array([[0, 1],
               [-1, 0]])

Bm = np.array([[0], [1]])

Q = np.diag([0, 1])
R = np.array([[1.0]])

Pm = solve_continuous_are(Am, Bm, Q, R)
K = np.linalg.inv(R) @ Bm.T @ Pm  # shape (1,2)


def phi_v(x):
    x1, x2 = x
    return np.array([x1**2, x1*x2, x2**2])


def phi_u(x):
    x1, x2 = x
    return np.array([x1**2, x1*x2, x2**2])


def run_experiment(x0, rho, T, dt, K, T_PE, W):
    lenT = int(T / dt)
    n = x0.shape[0]
    arr   = np.zeros((lenT, n)); arr[0]   = x0
    arr_m = np.zeros((lenT, n)); arr_m[0] = x0
    eta        = np.zeros((lenT, n))
    thresh_arr = np.zeros(lenT)
    u_arr      = np.zeros(lenT)   # scalar control

    # Threshold uses lam_max(Q): lam_min(Q)=0 would zero the state term (degenerate).
    # Consistent with OBS-5 (paper uses lam_max in implementation).
    lam_max_Q = np.max(np.diag(Q))
    lam_min_R = np.min(np.diag(R))
    lam_max_R = np.max(np.diag(R))

    # PE probe: sum of sinusoids (scalar, since u is scalar here)
    freqs = np.array([1.0, 2.3, 3.7, 5.1, 7.9, 11.3]); amp = 1.0; phase = 2.3
    def probe(tau):
        return amp * np.sum(np.sin(freqs * tau + phase))

    N_PE = int(T_PE / dt)
    X_data = np.zeros((N_PE, n))
    U_data = np.zeros(N_PE)   # scalar probe values

    x  = x0.astype(float).copy()
    xm = x0.astype(float).copy()
    t_s = None; collect_start = None
    w_u = None; w_v = None; history = None

    for i in range(1, lenT):
        u_m = (-(K @ x)).item()   # scalar

        # switching test (phase 1 only)
        if collect_start is None:
            thr = (1 / (xi**2 * lam_max_R)) * (lam_max_Q * np.linalg.norm(x)**2
                                                + lam_min_R * u_m**2)
            if np.linalg.norm(x - xm)**2 >= thr and np.linalg.norm(x) > x_min:
                t_s = i; collect_start = i

        # control for this step
        if collect_start is None:
            u = u_m
        elif (i - collect_start) < N_PE:
            k = i - collect_start
            ut = probe(k * dt)
            u = u_m + ut
            X_data[k] = x
            U_data[k] = ut
        else:
            # composite: u_m + learned augmentation (nonlinear in x via phi_u)
            u_tilde = float(w_u @ phi_u(x)) if w_u is not None else 0.0
            u = u_m + u_tilde

        thresh_arr[i] = (1 / (xi**2 * lam_max_R)) * (lam_max_Q * np.linalg.norm(x)**2
                                                      + lam_min_R * u_m**2)
        u_arr[i] = u

        # learn once PE window is full
        if collect_start is not None and (i - collect_start) == N_PE - 1 and w_u is None:
            w_u, w_v, history = policy_iteration(X_data, U_data, W, dt)

        x  = x  + (f(x, rho)  + g(x, rho)  * u)  * dt   # true nonlinear system
        xm = xm + (Am @ xm + Bm.flatten() * u_m) * dt   # linear model
        arr[i] = x; arr_m[i] = xm; eta[i] = x - xm

    eta_norm = np.linalg.norm(eta, axis=1)
    return arr, arr_m, eta, eta_norm, thresh_arr, u_arr, t_s, w_v, w_u, X_data, U_data, history

def build_regression(X_data, U_data, w_u_i, W, R_scalar, dt):
    rows = []
    costs = []
    k = 0
    while k+W < len(X_data):
        psi_v = phi_v(X_data[k+W]) - phi_v(X_data[k])
        phi_cost = 0.0
        psi_u = np.zeros(3)
        for j in range(k, k+W):
            mu_i_j = w_u_i @ phi_u(X_data[j])
            diff = U_data[j] - mu_i_j
            psi_u += 2 * diff * R_scalar * phi_u(X_data[j]) * dt
            phi_cost += (X_data[j] @ Q @ X_data[j]  +  mu_i_j**2 * R_scalar) * dt
        row = np.concatenate([psi_v, psi_u])
        rows.append(row)
        costs.append(phi_cost)

        k += W
    Psi = np.array(rows)
    Phi = np.array(costs)
    return Psi, Phi

def policy_iteration(X_data, U_data, W, dt, eps=1e-6):
    w_u_i = np.zeros((3,))
    W_prev = 0
    history = []
    while True:
        Psi, Phi = build_regression(X_data, U_data, w_u_i, W, R.item(), dt)
        W_hat = np.linalg.lstsq(Psi, -Phi, rcond =None)[0]
        w_v = W_hat[:3]
        w_u_i = W_hat[3:]
        history.append(W_hat)
        if np.linalg.norm(W_hat - W_prev) < eps:
            break
        W_prev = W_hat
    return w_u_i, w_v, history

# ── Case 1: rho=0.05, T=12s (switch should never fire) ──────────────────────
arr1, _, eta1, eta_norm1, thresh1, u_arr1, t_s1, _, _, _, _, _ = \
    run_experiment(x0, 0.05, 12, dt, K, T_PE, 10)

# ── Case 2: rho=1, T=30s (switch fires, RL learns augmentation) ──────────────
(arr2, _, eta2, eta_norm2, thresh2, u_arr2,
 t_s2, w_v2, w_u2, X_data, U_data, history) = run_experiment(x0, 1, 30, dt, K, T_PE, 10)

# model-based only run for case 2 dashed comparison (no switch allowed)
def run_model_based_only(x0, rho, T, dt, K):
    lenT = int(T / dt)
    arr = np.zeros((lenT, 2)); arr[0] = x0
    x = x0.astype(float).copy()
    lam_max_Q = np.max(np.diag(Q)); lam_min_R = np.min(np.diag(R)); lam_max_R = np.max(np.diag(R))
    for i in range(1, lenT):
        u = (-(K @ x)).item()
        x = x + (f(x, rho) + g(x, rho) * u) * dt
        arr[i] = x
    return arr

arr2_mb = run_model_based_only(x0, 1, 30, dt, K)

# optimal weights for validation
w_v_star = np.array([1.0, 0.0, 1.0])
w_u_star = np.array([0.0, -1.0, 0.0])

print(f"Case 2  t_s = {t_s2 * dt:.3f} s")
print(f"w_v = {w_v2}")
print(f"w_u = {w_u2}")

# ── Figure 5: Case 7.2.1 (rho=0.05) ─────────────────────────────────────────
t1 = np.arange(int(12/dt)) * dt
inset_lo, inset_hi = 4.2, 5.4   # zoom window where eta nearly crosses threshold

fig5, ax5 = plt.subplots(4, 1, figsize=(7, 10))

# pane 1: states x1, x2
ax5[0].plot(t1, arr1[:, 0], label='$x_1$')
ax5[0].plot(t1, arr1[:, 1], label='$x_2$')
ax5[0].set_ylabel('States')
ax5[0].legend(fontsize=8)
ax5[0].set_xlim(0, 12)

# pane 2: control
ax5[1].plot(t1, u_arr1, label='$u$')
ax5[1].set_ylabel('Control')
ax5[1].legend(fontsize=8)
ax5[1].set_xlim(0, 12)

# pane 3: model error vs threshold (with inset)
ax5[2].plot(t1, eta_norm1**2, label=r'$\|\eta\|^2$')
ax5[2].plot(t1, thresh1, label='Threshold')
ax5[2].set_ylabel('Model Error')
ax5[2].legend(fontsize=8)
ax5[2].set_xlim(0, 12)
axin3 = ax5[2].inset_axes([0.55, 0.30, 0.40, 0.55])
axin3.plot(t1, eta_norm1**2); axin3.plot(t1, thresh1)
axin3.set_xlim(inset_lo, inset_hi)
axin3.set_ylim(0, max(thresh1[int(inset_lo/dt):int(inset_hi/dt)].max(),
                      (eta_norm1**2)[int(inset_lo/dt):int(inset_hi/dt)].max()) * 1.15)

# pane 4: state norm vs x_min (with inset)
state_norm1 = np.linalg.norm(arr1, axis=1)
ax5[3].plot(t1, state_norm1, label=r'$\|x(t)\|$')
ax5[3].axhline(x_min, color='k', linestyle=':', linewidth=1.2, label='$x_{min}$')
ax5[3].set_ylabel('States Norm')
ax5[3].set_xlabel('$t$ [s]')
ax5[3].legend(fontsize=8)
ax5[3].set_xlim(0, 12)
axin4 = ax5[3].inset_axes([0.55, 0.30, 0.40, 0.55])
axin4.plot(t1, state_norm1); axin4.axhline(x_min, color='k', linestyle=':', linewidth=1.2)
axin4.set_xlim(inset_lo, inset_hi)

fig5.suptitle('Figure 5 (Section 7.2.1): rho=0.05, no switching')
plt.tight_layout()

# ── Figure 6: Case 7.2.2 (rho=1) ────────────────────────────────────────────
t2 = np.arange(int(30/dt)) * dt
ts_time = t_s2 * dt if t_s2 is not None else None

fig6, ax6 = plt.subplots(4, 1, figsize=(7, 10))

# pane 1: states — switched composite (solid) vs model-based only (dashed)
ax6[0].plot(t2, arr2[:, 0], label='$x_1$')
ax6[0].plot(t2, arr2[:, 1], label='$x_2$')
ax6[0].plot(t2, arr2_mb[:, 0], '--', label='$x_1$ (model-based)')
ax6[0].plot(t2, arr2_mb[:, 1], '--', label='$x_2$ (model-based)')
ax6[0].set_ylabel('States')
ax6[0].legend(fontsize=7)
ax6[0].set_xlim(0, 30)
if ts_time: ax6[0].axvline(ts_time, color='gray', linestyle=':')

# pane 2: control (oscillatory during PE window, settles after)
ax6[1].plot(t2, u_arr2, label='$u$', linewidth=0.8)
ax6[1].set_ylabel('Control')
ax6[1].legend(fontsize=8)
ax6[1].set_xlim(0, 30)
if ts_time: ax6[1].axvline(ts_time, color='gray', linestyle=':')

# pane 3: model error vs threshold with inset on switching window
ax6[2].plot(t2, eta_norm2**2, label=r'$\|\eta\|^2$')
ax6[2].plot(t2, thresh2, label='Threshold')
ax6[2].set_ylabel('Model Error')
ax6[2].legend(fontsize=8)
ax6[2].set_xlim(0, 30)
if ts_time: ax6[2].axvline(ts_time, color='gray', linestyle=':')
axin6 = ax6[2].inset_axes([0.55, 0.30, 0.40, 0.55])
axin6.plot(t2, eta_norm2**2); axin6.plot(t2, thresh2)
axin6.set_xlim(0, 1.0)
axin6.set_ylim(0, max(thresh2[:100].max(), (eta_norm2**2)[:100].max()) * 1.15)
if ts_time: axin6.axvline(ts_time, color='gray', linestyle=':')

# pane 4: state norm vs x_min
state_norm2 = np.linalg.norm(arr2, axis=1)
ax6[3].plot(t2, state_norm2, label=r'$\|x(t)\|$')
ax6[3].axhline(x_min, color='k', linestyle=':', linewidth=1.2, label='$x_{min}$')
ax6[3].set_ylabel('States Norm')
ax6[3].set_xlabel('$t$ [s]')
ax6[3].legend(fontsize=8)
ax6[3].set_xlim(0, 30)
if ts_time: ax6[3].axvline(ts_time, color='gray', linestyle=':')

fig6.suptitle('Figure 6 (Section 7.2.2): rho=1, switched composite vs model-based')
plt.tight_layout()

# ── Figure 7: weight convergence ─────────────────────────────────────────────
w_v_errs = [np.linalg.norm(h[:3] - w_v_star) for h in history]
w_u_errs = [np.linalg.norm(h[3:] - w_u_star) for h in history]

fig7, ax7 = plt.subplots(figsize=(7, 4))
ax7.semilogy(range(len(w_v_errs)), w_v_errs, 'o-', label=r'$\|\hat{w}_v^i - w_v^*\|$')
ax7.semilogy(range(len(w_u_errs)), w_u_errs, 's-', label=r'$\|\hat{w}_u^i - w_u^*\|$')
ax7.set_xlabel('Iteration Number')
ax7.set_ylabel('Weights Norm')
ax7.set_title('Figure 7 (Section 7.2.2): weight convergence')
ax7.legend(fontsize=9)
ax7.grid(True, which='both', alpha=0.3)
plt.tight_layout()

plt.show()