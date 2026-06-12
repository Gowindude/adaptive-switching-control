from scipy.linalg import solve_continuous_are
import numpy as np
import matplotlib.pyplot as plt

# accurate model (case 1 from 7.1)
A = np.array([[0, 1, 0, 0],
              [-8.5, -10.625, 4.25, 6.375],
              [0, 0, 0, 1],
              [4.25, 6.375, -4.25, -6.375]])
x0 = np.array([3, -2, 5, -1])
dt = 0.01 # in seconds
T = 15 # in seconds

lenT = int(T / dt)

def getArr(x0, dt, T, A):  
    arr = np.zeros((lenT, x0.shape[0]))
    arr[0] = x0
    
    x = x0
    for i in range(1, lenT):
        x_dot = A @ x
        x = x + x_dot * dt
        arr[i] = x
    return arr

# inaccurate model (case 2 from section 7.1)
Am = A.copy()
Am[1, 1] = -0.55 * A[1, 1]  # sign flip on one element

Bm = np.array([[0, 0],
               [1, 0],
               [0, 0],
               [0, 1]])

Q_lqr = np.diag([1, 0.1, 1, 0.25])
R_lqr = 0.2 * np.eye(2)

Pm = solve_continuous_are(Am, Bm, Q_lqr, R_lqr)
K = np.linalg.inv(R_lqr) @ Bm.T @ Pm

def getArrControlled(x0, dt, T, A, K):
    xm = x0
    eta = np.zeros((lenT, x0.shape[0]))
    arr_m = np.zeros((lenT, x0.shape[0]))
    arr_m[0] = xm
    
    arr = np.zeros((lenT, x0.shape[0]))
    arr[0] = x0
    x = x0
    for i in range(1, lenT):
        u = -K @ x
        x_dot = A@x + Bm@u
        xm_dot = Am@xm + Bm@u
        x = x + (x_dot) * dt
        xm = xm + (xm_dot) * dt
        arr[i] = x
        arr_m[i] = xm
        eta[i] = x - xm
    eta_norm = np.linalg.norm(eta, axis=1)
    return arr, arr_m, eta, eta_norm


arr_bare = getArr(x0, dt, T, A)
arr_controlled, arr_m, eta, eta_norm = getArrControlled(x0, dt, T, A, K)
t = np.arange(0, lenT)

fig, axes = plt.subplots(2, 2)
axes = axes.flatten()

for i in range(x0.shape[0]):
    axes[i].plot(t, arr_bare[:, i], '--', label='uncontrolled', linewidth=0.8)
    axes[i].plot(t, arr_controlled[:, i], label='LQR controlled')
    axes[i].axhline(y=0, color='black', linewidth=0.5)
    axes[i].set_xlabel('Timestep')
    axes[i].set_ylabel('State Value')
    axes[i].set_title(f'State {i+1}')
    axes[i].legend()

plt.tight_layout()
plt.show()

plt.figure()
plt.plot(t, eta_norm, label='Gap normed')
plt.axhline(y=0, color='black', linewidth=0.5)
plt.xlabel('Timestep')
plt.ylabel('N value (normed)')
plt.legend()
plt.tight_layout()
plt.show()