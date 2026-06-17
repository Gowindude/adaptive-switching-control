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
        u_m = float(-(K @ x))   # scalar

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
            pass  # TODO: w_v, w_u, history = policy_iteration(X_data, U_data, W, dt)

        x  = x  + (f(x, rho)  + g(x, rho)  * u)  * dt   # true nonlinear system
        xm = xm + (Am @ xm + Bm.flatten() * u_m) * dt   # linear model
        arr[i] = x; arr_m[i] = xm; eta[i] = x - xm

    eta_norm = np.linalg.norm(eta, axis=1)
    return arr, arr_m, eta, eta_norm, thresh_arr, u_arr, t_s, w_v, w_u, X_data, U_data, history

