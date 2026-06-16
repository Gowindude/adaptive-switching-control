from scipy.linalg import solve_continuous_are
import numpy as np
import matplotlib.pyplot as plt

# Section 7.1.1 -- accurate (ideal) model case.
# True 2-DOF mass-spring-damper system.
A = np.array([[0, 1, 0, 0],
              [-8.5, -10.625, 4.25, 6.375],
              [0, 0, 0, 1],
              [4.25, 6.375, -4.25, -6.375]])
x0 = np.array([3, -2, 5, -1])
dt = 0.01  # seconds
T = 8      # seconds (paper simulates case 1 for 8 s)

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


# accurate model (case 1): scale row p=2 (0-based index 1) by 0.9
Am = A.copy()
Am[1, :] = 0.9 * A[1, :]

Bm = np.array([[0, 0],
               [1, 0],
               [0, 0],
               [0, 1]])

Q_lqr = np.diag([1, 0.1, 1, 0.25])
R_lqr = 0.2 * np.eye(2)

# model-based gain from the accurate model
Pm = solve_continuous_are(Am, Bm, Q_lqr, R_lqr)
K = np.linalg.inv(R_lqr) @ Bm.T @ Pm

# optimal augmentation (composite stand-in for the RL). Unused in case 1 since the
# switch never fires, but referenced by getArrControlled after a switch -- defined
# so the function body is identical to the 7.1.2 script.
P_tilde = solve_continuous_are(A - Bm @ K, Bm, Q_lqr, R_lqr)
K_tilde = np.linalg.inv(R_lqr) @ Bm.T @ P_tilde


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

    # Numerator uses lam_MAX(Q) to match the paper's implementation (threshold ~ 120).
    # NOTE: Theorem 1's *proof* (Eq. 17 deriv.) rigorously supports lam_min(Q) -- the
    # paper's experiments use the looser lam_max(Q). See RESEARCH-NOTES OBS-2 / OBS-5.
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
        u_norm_arr[i] = u_norm
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


arr_bare = getArr(x0, dt, T, A)
arr_controlled, arr_m, eta, eta_norm, t_s, thresh_arr, u_norm_arr, u_arr = getArrControlled(x0, dt, T, A, K, Am)

print("t_s =", t_s, "(expected None: accurate model regulates without switching)")

t = np.arange(0, lenT) * dt

# ============================================================================
# Figure 2 (paper 7.1.1, accurate model): norm of states, control components,
# and model error vs threshold. The model-based controller runs the whole time:
# ||eta||^2 stays UNDER the threshold, so the switch never fires (t_s = None).
# ============================================================================
state_norm = np.linalg.norm(arr_controlled, axis=1)
x_min = 0.005 * np.linalg.norm(x0)   # switching guard (same as inside getArrControlled)

fig, ax = plt.subplots(3, 1, figsize=(8, 9))

ax[0].plot(t[1:], state_norm[1:], label='model-based')
ax[0].axhline(x_min, color='black', linestyle=':', linewidth=0.8, label='x_min')
ax[0].set_ylabel('||x||')
ax[0].set_title('Norm of the states')
ax[0].legend(fontsize=8)
ax[0].set_xlim(0, 8)

ax[1].plot(t[1:], u_arr[1:, 0], label='u1')
ax[1].plot(t[1:], u_arr[1:, 1], label='u2')
ax[1].set_ylabel('u')
ax[1].set_xlabel('Time (s)')
ax[1].set_title('Control')
ax[1].legend(fontsize=8)
ax[1].set_xlim(0, 8)

# model error stays below threshold for all t -> no switch (the whole point of case 1)
ax[2].plot(t[1:], eta_norm[1:]**2, label='||eta||^2')
ax[2].plot(t[1:], thresh_arr[1:], label='threshold')
ax[2].set_ylabel('model error')
ax[2].set_xlabel('Time (s)')
ax[2].set_title('Norm of the model error vs. threshold (never crosses)')
ax[2].legend(fontsize=8)
ax[2].set_xlim(0, 8)

fig.suptitle('Figure 2 (Section 7.1.1): accurate model, no switch')
plt.tight_layout()
plt.savefig('figure2_accurate.png', dpi=150)
