from scipy.linalg import solve_continuous_are
import numpy as np
import matplotlib.pyplot as plt
import math

# Section 7.1.2 -- inaccurate model case.
# True 2-DOF mass-spring-damper system.
A = np.array([[0, 1, 0, 0],
              [-8.5, -10.625, 4.25, 6.375],
              [0, 0, 0, 1],
              [4.25, 6.375, -4.25, -6.375]])
x0 = np.array([3, -2, 5, -1])
dt = 0.01  # seconds
T = 15     # seconds

lenT = int(T / dt)


def getArr(x0, dt, T, A):
    # uncontrolled open-loop rollout of the true system (baseline)
    arr = np.zeros((lenT, x0.shape[0]))
    arr[0] = x0
    x = x0
    for i in range(1, lenT):
        x_dot = A @ x
        x = x + x_dot * dt
        arr[i] = x
    return arr


# inaccurate model (case 2): sign flip on element (2,2) (0-based [1,1])
Am = A.copy()
Am[1, 1] = -0.55 * A[1, 1]

Bm = np.array([[0, 0],
               [1, 0],
               [0, 0],
               [0, 1]])

Q_lqr = np.diag([1, 0.1, 1, 0.25])
R_lqr = 0.2 * np.eye(2)

# model-based gain from the inaccurate model
Pm = solve_continuous_are(Am, Bm, Q_lqr, R_lqr)
K = np.linalg.inv(R_lqr) @ Bm.T @ Pm

# true optimal LQR (solves the original problem 7) -- for reference/comparison only
P_true = solve_continuous_are(A, Bm, Q_lqr, R_lqr)
K_true = np.linalg.inv(R_lqr) @ Bm.T @ P_true

# optimal augmentation (analytic stand-in for the RL): LQR on the augmented plant A - Bm@K,
# cost penalizing only the augmentation. Composite applied control after the switch is
# u = -(K + K_tilde) x. NOTE: K + K_tilde != K_true in general -- see RESEARCH-NOTES OBS-1.
P_tilde = solve_continuous_are(A - Bm @ K, Bm, Q_lqr, R_lqr)
K_tilde = np.linalg.inv(R_lqr) @ Bm.T @ P_tilde

print("K_true:\n", K_true)
print("K + K_tilde:\n", K + K_tilde)



def getArrControlled(x0, dt, T, A, K, A_m, allow_switch=True):
    xm = x0
    eta = np.zeros((lenT, x0.shape[0]))
    arr_m = np.zeros((lenT, x0.shape[0]))
    arr_m[0] = xm

    arr = np.zeros((lenT, x0.shape[0]))
    arr[0] = x0
    x = x0

    thresh_arr = np.zeros(lenT)
    u_norm_arr = np.zeros(lenT)
    u_arr = np.zeros((lenT, 2))

    # Numerator uses lam_MAX(Q) to match the paper's implementation (gives
    # t_s ~ 0.25 s and threshold ~ 120). NOTE: Theorem 1's *proof* (Eq. 17 deriv.)
    # rigorously supports lam_min(Q) -- the paper's experiments use the looser
    # lam_max(Q). See RESEARCH-NOTES OBS-2 / OBS-5.
    lam_max_Q = np.max(np.diag(Q_lqr))
    lam_min_R = np.min(np.diag(R_lqr))
    lam_max_R = np.max(np.diag(R_lqr))
    xi = 1.35
    t_s = None
    switched = False

    x_min = 0.005 * np.linalg.norm(x0)

    for i in range(1, lenT):
        if not switched:
            u = -K @ x
        else:
            u = -(K + K_tilde) @ x

        eta_now = x - xm
        eta_norm_sq = np.linalg.norm(eta_now)**2
        x_norm = np.linalg.norm(x)
        u_norm = np.linalg.norm(u)
        threshold = (1 / (xi**2 * lam_max_R)) * (lam_max_Q * x_norm**2 + lam_min_R * u_norm**2)
        thresh_arr[i] = threshold
        u_norm_arr[i] = np.linalg.norm(u)
        u_arr[i] = u

        if not switched and eta_norm_sq >= threshold and x_norm > x_min and allow_switch:
            switched = True
            t_s = i

        x_dot = A @ x + Bm @ u
        xm_dot = A_m @ xm + Bm @ u
        x = x + x_dot * dt
        xm = xm + xm_dot * dt
        arr[i] = x
        arr_m[i] = xm
        eta[i] = x - xm
    eta_norm = np.linalg.norm(eta, axis=1)
    return arr, arr_m, eta, eta_norm, t_s, thresh_arr, u_norm_arr, u_arr

def phi_v(x):
    x1 = x[0]
    x2 = x[1]
    x3 = x[2]
    x4 = x[3]

    arr = [x1**2, x1*x2, x1*x3, x1*x4, x2**2, x2*x3, x2*x4, x3**2, x3*x4, x4**2]
    return np.array(arr)

def build_regression(X, U_tilde, K_i, W, dt):
    rows = []
    costs = []
    k = 0
    while k+W < len(X):
        psi_v = phi_v(X[k+W]) - phi_v(X[k])
        phi_cost = 0.0
        psi_u = np.zeros(8)
        for j in range(k, k+W):
            xj = X[j]
            mj = -K_i @ xj
            diff = U_tilde[j] - mj
            phi_cost += (xj @ Q_lqr @ xj + mj @ R_lqr @ mj) * dt
            psi_u    += 2 * np.kron(diff @ R_lqr, xj) * dt
        row = np.concatenate([psi_v, psi_u])
        rows.append(row)
        costs.append(phi_cost)

        k += W
    Psi = np.array(rows)
    Phi = np.array(costs)
    return Psi, Phi

def policy_iteration(X, U_tilde, W, dt, eps=1e-6):
    K_i = np.zeros((2, 4))
    W_prev = 0
    history = []
    while True:
        Psi, Phi = build_regression(X, U_tilde, K_i, W, dt)
        W_hat = np.linalg.lstsq(Psi, -Phi, rcond =None)[0]
        w_v = W_hat[:10]
        w_u_vec = W_hat[10:]
        w_u = w_u_vec.reshape(4, 2, order = 'F')
        K_next = -w_u.T
        K_i = K_next
        history.append(W_hat)
        if np.linalg.norm(W_hat - W_prev) < eps:
            break
        W_prev = W_hat
    return K_i, w_v, history

def P_from_wv(w_v):
    P = np.zeros((4, 4))
    idx = 0
    for i in range(4):
        for j in range(i, 4):
            if i == j:
                P[i, i] = w_v[idx]
            elif i < j:
                P[i, j] = w_v[idx]/2
                P[j, i] = w_v[idx]/2
            idx += 1
    return P

def inverseP_from_wv(K_tilde, P_tilde):
    w_u_vec_star = (-K_tilde.T).flatten(order='F')
    w_v_star = np.zeros(10)
    idx = 0
    for i in range(4):
        for j in range(i, 4):
            if i == j:
                w_v_star[idx] = P_tilde[i, i]
            elif i < j:
                w_v_star[idx] = 2 * P_tilde[i, j]
            idx += 1
    w_star = np.concatenate([w_v_star, w_u_vec_star])
    return w_star

def run_experiment(x0, dt, T, A, Am, K, Bm, T_PE, W):
    """One continuous trajectory reproducing the paper's experiment:
      phase 1 [0, t_s)        : model-based  u = -K x          (smooth)
      phase 2 [t_s, t_s+T_PE) : behavior     u = -K x + probe  (PE excitation -> collect data)
      phase 3 [t_s+T_PE, T)   : composite    u = -(K + K_learned) x
    The probe segment IS the data fed to policy_iteration -- one source of truth, so the
    oscillatory control pane and the RL training set are the same trajectory."""
    lenT = int(T / dt)
    arr = np.zeros((lenT, 4)); arr[0] = x0
    arr_m = np.zeros((lenT, 4)); arr_m[0] = x0
    eta = np.zeros((lenT, 4))
    thresh_arr = np.zeros(lenT)
    u_arr = np.zeros((lenT, 2))

    lam_max_Q = np.max(np.diag(Q_lqr))   # paper's figures use MAX(Q); Eq.17 writes MIN -- see OBS-5
    lam_min_R = np.min(np.diag(R_lqr))
    lam_max_R = np.max(np.diag(R_lqr))
    xi = 1.35
    x_min = 0.005 * np.linalg.norm(x0)

    freqs = np.array([1.0, 2.3, 3.7, 5.1, 7.9, 11.3]); amp = 6; phase = 2.3   # amp sized to paper's control envelope
    def probe(tau):
        return np.array([amp * np.sum(np.sin(freqs * tau)),
                         amp * np.sum(np.sin(freqs * tau + phase))])

    N_PE = int(T_PE / dt)
    X_data = np.zeros((N_PE, 4)); U_data = np.zeros((N_PE, 2))

    x = x0.astype(float).copy(); xm = x0.astype(float).copy()
    t_s = None; collect_start = None
    K_learned = None; w_v = None; history = None

    for i in range(1, lenT):
        # switching test (phase 1 only), evaluated on the model-based control
        if collect_start is None:
            u_m = -K @ x
            thr = (1 / (xi**2 * lam_max_R)) * (lam_max_Q * np.linalg.norm(x)**2
                                               + lam_min_R * np.linalg.norm(u_m)**2)
            if np.linalg.norm(x - xm)**2 >= thr and np.linalg.norm(x) > x_min:
                t_s = i; collect_start = i

        # control for this step
        if collect_start is None:
            u = -K @ x
        elif (i - collect_start) < N_PE:
            k = i - collect_start
            ut = probe(k * dt)
            u = -K @ x + ut
            X_data[k] = x          # state at collection step k
            U_data[k] = ut         # augmentation (probe) applied at X_data[k] going forward
        else:
            u = -(K + K_learned) @ x

        thresh_arr[i] = (1 / (xi**2 * lam_max_R)) * (lam_max_Q * np.linalg.norm(x)**2
                                                     + lam_min_R * np.linalg.norm(u)**2)
        u_arr[i] = u

        # learn once the collection window is full (used from the next step on)
        if collect_start is not None and (i - collect_start) == N_PE - 1 and K_learned is None:
            K_learned, w_v, history = policy_iteration(X_data, U_data, W, dt)

        x = x + (A @ x + Bm @ u) * dt
        xm = xm + (Am @ xm + Bm @ u) * dt
        arr[i] = x; arr_m[i] = xm; eta[i] = x - xm

    eta_norm = np.linalg.norm(eta, axis=1)
    return arr, arr_m, eta, eta_norm, thresh_arr, u_arr, t_s, K_learned, w_v, X_data, U_data, history

arr_bare = getArr(x0, dt, T, A)
arr_mb, _, _, eta_norm_mb, _, thresh_mb, _, _ = getArrControlled(x0, dt, T, A, K, Am, allow_switch=False)

# Single continuous trajectory: model-based -> switch -> probe/collect -> learned composite.
# This run carries the inline PE probe -> use it for the control and model-error panes.
(arr_controlled, arr_m, eta, eta_norm, thresh_arr, u_arr,
 t_s_idx, K_learned, w_v, X, U_tilde, history) = run_experiment(x0, dt, T, A, Am, K, Bm, 8.0, 10)
t_s = t_s_idx

# Deployed composite WITHOUT the probe (model-based -> switch -> -(K+K_tilde)x). The probe
# injects exploration energy that masks the composite's ~18% regulation edge over model-based,
# so the state-norm pane shows this clean deployment (the control pane keeps the probe). See OBS-6.
arr_clean = getArrControlled(x0, dt, T, A, K, Am, allow_switch=True)[0]

# RL validation: learned augmentation vs analytic ground truth ---
# Actor must converge to K_tilde, critic to P_tilde (NOT K_true / P_true).
# Residual gap is forward-Euler / Riemann-sum discretization error (O(dt)).
P_learned = P_from_wv(w_v)

K_max_err = np.abs(K_learned - K_tilde).max()
K_rel_err = np.linalg.norm(K_learned - K_tilde) / np.linalg.norm(K_tilde)
P_max_err = np.abs(P_learned - P_tilde).max()
P_rel_err = np.linalg.norm(P_learned - P_tilde) / np.linalg.norm(P_tilde)

print(f"actor:  max|K_learned - K_tilde| = {K_max_err:.4f}   rel = {K_rel_err*100:.2f}%")
print(f"critic: max|P_learned - P_tilde| = {P_max_err:.4f}   rel = {P_rel_err*100:.2f}%")

print("t_s =", None if t_s is None else f"{t_s * dt:.2f} s")

t = np.arange(0, lenT) * dt

# Figure 3 (paper 7.1.2): norm of states (switched vs model-based-only),
# control norm, and model error vs threshold. Gray line = switching moment t_s.

state_norm = np.linalg.norm(arr_controlled, axis=1)   # ||x|| including probe window (matches paper)
state_norm_mb = np.linalg.norm(arr_mb, axis=1)        # ||x|| under model-based only

fig, ax = plt.subplots(3, 1, figsize=(8, 9))

# top + middle: full-horizon regulation story (||x||, ||u||) over the full 15 s sim
x_min = 0.005 * np.linalg.norm(x0)   # switching guard (same as inside getArrControlled)

ax[0].plot(t[1:], state_norm[1:], label='switched composite (deployed)')
ax[0].plot(t[1:], state_norm_mb[1:], '--', label='model-based only')
ax[0].axhline(x_min, color='black', linestyle=':', linewidth=0.8, label='x_min')
ax[0].set_ylabel('||x||')
ax[0].set_title('Norm of the states')
ax[0].legend(fontsize=8)
ax[0].set_xlim(0, 15)

# control components u1, u2: oscillatory during [t_s, t_s+T_PE] -- the persistent-
# excitation probe injected inline for data collection -- then smooth composite.
ax[1].plot(t[1:], u_arr[1:, 0], label='u1', linewidth=0.7)
ax[1].plot(t[1:], u_arr[1:, 1], label='u2', linewidth=0.7)
ax[1].set_ylabel('u')
ax[1].set_xlabel('Time (s)')
ax[1].set_title('Control (probe excitation during learning, then composite)')
ax[1].legend(fontsize=8)
ax[1].set_xlim(0, 15)

# bottom: model error vs threshold over the FULL run (eta diverges to ~1e100 after
# t_s on the unstable A_m -- matches the paper's x10^100 axis), with an inset zoomed
# to the switching window showing ||eta||^2 crossing the threshold at t_s.
ax[2].plot(t[1:], eta_norm[1:]**2, label='||eta||^2')
ax[2].plot(t[1:], thresh_arr[1:], label='threshold')
ax[2].set_ylabel('model error')
ax[2].set_xlabel('Time (s)')
ax[2].set_title('Norm of the model error vs. threshold (full run + zoom)')
ax[2].set_xlim(0, 15)
ax[2].legend(fontsize=8, loc='upper left')

axin = ax[2].inset_axes([0.30, 0.30, 0.45, 0.6])   # inset on the switching window
axin.plot(t[1:], eta_norm[1:]**2)
axin.plot(t[1:], thresh_arr[1:])
axin.set_xlim(0.04, 0.4)
axin.set_ylim(0, 1300)
if t_s is not None:
    axin.axvline(t_s * dt, color='gray', linestyle=':')

for a in (ax[0], ax[1]):
    if t_s is not None:
        a.axvline(t_s * dt, color='gray', linestyle=':')
if t_s is not None:
    ax[2].axvline(t_s * dt, color='gray', linestyle=':')
fig.suptitle('Figure 3 (Section 7.1.2): switched vs. model-based controller')
plt.tight_layout()

# Figure 4 (paper 7.1.2): convergence of Algorithm-1 weights to the optimal
# weights w* over policy-iteration steps. Floor at the O(dt) Euler error
w_star = inverseP_from_wv(K_tilde, P_tilde)
v_errs = [np.linalg.norm(w[:10] - w_star[:10]) for w in history]
u_errs = [np.linalg.norm(w[10:] - w_star[10:]) for w in history]

plt.figure(figsize=(7, 5))
plt.semilogy(range(len(v_errs)), v_errs, 'o-', label=r'$\|\hat{w}_v^i - w_v^*\|$')
plt.semilogy(range(len(u_errs)), u_errs, 's-', label=r'$\|\hat{w}_u^i - w_u^*\|$')
plt.xlabel('Iteration Number')
plt.ylabel('Weights Norm')
plt.title('Figure 4 (7.1.2): weight error vs iteration')
plt.legend(fontsize=9)
plt.grid(True, which='both', alpha=0.3)
plt.tight_layout()
plt.show()   # display Figure 3 and Figure 4 interactively
