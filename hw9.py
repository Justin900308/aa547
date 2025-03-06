import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
from scipy.integrate import odeint
import numpy as np
from numpy import linalg as LA
from scipy.integrate import solve_ivp

################### P2
k = np.array([-3, - 2])
t0 = 0
tf = 20
dt = 0.05
T = int(tf / dt + 1)
t_all = np.linspace(t0, tf, T)
x0 = np.array([-np.pi / 3, -np.pi / 5])
r = 1
A = np.array([[0, 1], [1, 0]])


def dny_sys_linear(
        t: np.ndarray,
        x: np.ndarray
) -> np.ndarray:
    u = k @ x
    x_dot = np.zeros(2)
    x_dot[0] = x[1]
    x_dot[1] = x[0] + u
    return x_dot


def dny_sys(
        t: np.ndarray,
        x: np.ndarray
) -> np.ndarray:
    u = k @ x
    x_dot = np.zeros(2)
    x_dot[0] = x[1]
    x_dot[1] = np.sin(x[0]) + u
    return x_dot


sol = solve_ivp(dny_sys, [t0, tf], x0, method='RK45', t_eval=t_all)

## Plotting
u0 = np.array([0, 1])  ## initial directional vector
plt.plot(u0[0], u0[1], 'r.')
x_traj = sol.y
plt.plot(t_all, x_traj[0, :])
plt.plot(t_all, x_traj[1, :])
plt.show()
theta = np.linspace(0, 2 * np.pi, 301)

for t in range(T):
    u = np.zeros(2)
    x_t = x_traj[:, t]
    u[0] = np.sin(x_t[0])
    u[1] = np.cos(x_t[0])
    plt.plot(np.cos(theta),np.sin(theta))
    plt.plot([0, u[0]], [0, u[1]], 'r-',linewidth = 4)
    print(LA.norm(u, 2))
    plt.xlim([-1, 1])
    plt.ylim([0, 1.2])
    plt.pause(0.02)
    plt.clf()
