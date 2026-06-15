from scipy.linalg import solve_continuous_are
import numpy as np
import matplotlib.pyplot as plt

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

def phi_v(x):
    x1 = x[0]
    x2 = x[1]
    x3 = x[2]
    x4 = x[3]

    arr = [x1**2, x1*x2, x1*x3, x1*x4, x2**2, x2*x3, x2*x4, x3**2, x3*x4, x4**2]
    return np.array(arr)

def collect_data(x_start, K, T_PE, dt):
    x = x_start

    freqs = np.array([1.0, 2.3, 3.7, 5.1, 7.9, 11.3])
    amp = 3
    phase = 2.3
    u_tilde = np.array([amp * np.sum(np.sin(freqs * 0)),
    amp * np.sum(np.sin(freqs * 0 + phase))])

    N = int(T_PE/dt)
    X = np.zeros((N, 4))
    U_tilde = np.zeros((N, 2))
    U_tilde[0] = u_tilde
    X[0] = x

    for i in range(1, N):
        t = i * dt
        u = -K @ x + u_tilde
        x = x + (A @ x + Bm @ u) * dt
        u_tilde = np.array([amp * np.sum(np.sin(freqs * t)),
        amp * np.sum(np.sin(freqs * t + phase))])

        U_tilde[i] = u_tilde
        X[i] = x
    return X, U_tilde
        

def getArrControlled(x0, dt, T, A, K, A_m, allow_switch=True):
    xm = x0
    eta = np.zeros((lenT, x0.shape[0]))
    arr_m = np.zeros((lenT, x0.shape[0]))
    arr_m[0] = xm

    arr = np.zeros((lenT, x0.shape[0]))
    arr[0] = x0
    x = x0

    thresh_arr = np.zeros(lenT)

    lam_min_Q = np.min(np.diag(Q_lqr))
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
        threshold = (1 / (xi**2 * lam_max_R)) * (lam_min_Q * x_norm**2 + lam_min_R * u_norm**2)
        thresh_arr[i] = threshold

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
    return arr, arr_m, eta, eta_norm, t_s, thresh_arr


arr_bare = getArr(x0, dt, T, A)
arr_controlled, arr_m, eta, eta_norm, t_s, thresh_arr = getArrControlled(x0, dt, T, A, K, Am)
arr_mb, _, _, eta_norm_mb, _, thresh_mb = getArrControlled(x0, dt, T, A, K, Am, allow_switch=False)
X, U_tilde = collect_data(arr_controlled[t_s], K, 8.0, dt)
print(X.shape, U_tilde.shape, np.isfinite(X).all(), np.abs(X).max())

print("t_s =", None if t_s is None else f"{t_s * dt:.2f} s")

t = np.arange(0, lenT) * dt

# states: switched composite vs model-based-only (switch helps -> curves diverge after t_s)
# fig, axes = plt.subplots(2, 2, figsize=(10, 7))
# axes = axes.flatten()
# for i in range(x0.shape[0]):
#     axes[i].plot(t, arr_controlled[:, i], '--', label='switched', linewidth=0.8)
#     axes[i].plot(t, arr_mb[:, i], label='model-based')
#     if t_s is not None:
#         axes[i].axvline(t_s * dt, color='gray', linestyle=':', label='switch')
#     axes[i].axhline(0, color='black', linewidth=0.5)
#     axes[i].set_xlabel('Time (s)')
#     axes[i].set_ylabel(f'State {i+1}')
#     axes[i].legend(fontsize=8)
#     axes[i].set_xlim(0, 10)
# plt.tight_layout()
# plt.savefig('states.png', dpi=150)
# plt.show()

# # model error vs threshold near the switch (reproduces Figure 3, bottom pane)
# plt.figure(figsize=(8, 5))
# plt.plot(t, eta_norm**2, label='||eta||^2')
# plt.plot(t, thresh_arr, label='threshold')
# if t_s is not None:
#     plt.axvline(t_s * dt, color='gray', linestyle=':', label=f'switch t_s = {t_s*dt:.2f}s')
# plt.xlabel('Time (s)')
# plt.ylabel('value')
# plt.xlim(0, 0.4)
# plt.ylim(0, 50)
# plt.legend()
# plt.tight_layout()
# plt.savefig('eta_threshold.png', dpi=150)
# plt.show()
