import numpy as np
import matplotlib.pyplot as plt

A = np.array([[0, 1, 0, 0],
              [-8.5, -10.625, 4.25, 6.375],
              [0, 0, 0, 1],
              [4.25, 6.375, -4.25, -6.375]])
x0 = np.array([3, -2, 5, -1])
dt = 0.01 # in seconds
T = 8 # in seconds

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

arr = getArr(x0, dt, T, A)
t = np.arange(1, lenT+1, 1)

fig, axes = plt.subplots(2, 2)
axes = axes.flatten()

for i in range(x0.shape[0]):
    axes[i].plot(t-1, arr[:, i])
    axes[i].axhline(y=0, color='black', linewidth=0.5)
    axes[i].set_xlabel('Timestep')
    axes[i].set_ylabel('State Value')
    axes[i].set_title(f'State {i+1}')

plt.tight_layout()
plt.show()