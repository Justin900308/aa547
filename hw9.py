import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
from scipy.integrate import odeint
import numpy as np
from numpy import linalg as LA
from scipy.integrate import solve_ivp

################### P2,3
k = np.array([-3, - 2])
t0 = 0
tf = 5
dt = 0.05
T = int(tf / dt + 1)
t_all = np.linspace(t0, tf, T)

# Initial condition for system 1
x0 = np.array([-np.pi / 3, -np.pi / 5])

# Initial condition for system 2
x20 = np.array([-np.pi / 3, -2])  # physical states
e0 = np.array([5, -3])  # initial measurement error
x_hat_0 = x0 - e0  # control states
x2_0 = np.concatenate((x20, x_hat_0), 0)  # total states
r = 1
A = np.array([[0, 1], [1, 0]])


def dny_sys_linear(
        t: np.ndarray,
        x: np.ndarray
) -> np.ndarray:
    u = k @ x
    x_dot = np.zeros(2)
    x_dot[0] = x[1]
    x_dot[1] = x[0]
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


def dny_sys2(
        t: np.ndarray,
        x_2: np.ndarray,
) -> np.ndarray:
    x = x_2[0:2]
    x_hat = x_2[2:4]

    x_hat_dot = np.zeros(2)
    x_hat_dot[0] = -11 * x_hat[0] + x_hat[1] + 11 * x[0]
    x_hat_dot[1] = -33 * x_hat[0] - 2 * x_hat[1] + 31 * x[0]

    x_dot = np.zeros(2)
    x_dot[0] = x[1]
    x_dot[1] = np.sin(x[0]) - 3 * x_hat[0] - 2 * x_hat[1]
    # x_dot[1] = x[0] - 3 * x_hat[0] - 2 * x_hat[1]
    x_2_dot = np.concatenate((x_dot, x_hat_dot), 0)

    return x_2_dot


sol1 = solve_ivp(dny_sys, [t0, tf], x0, method='RK45', t_eval=t_all)
sol2 = solve_ivp(dny_sys2, [t0, tf], x2_0, method='RK45', t_eval=t_all)
## Plotting

# for p2
u0 = np.array([0, 1])  ## initial directional vector
plt.plot(u0[0], u0[1], 'r.')
x_traj = sol1.y
plt.plot(t_all, x_traj[0, :])
plt.plot(t_all, x_traj[1, :])
plt.show()
theta = np.linspace(0, 2 * np.pi, 301)

for t in range(T):
    u = np.zeros(2)
    x_t = x_traj[:, t]
    u[0] = np.sin(x_t[0])
    u[1] = np.cos(x_t[0])
    plt.plot(np.cos(theta), np.sin(theta))
    plt.plot([0, u[0]], [0, u[1]], 'r-', linewidth=4)
    print(LA.norm(u, 2))
    plt.xlim([-1, 1])
    plt.ylim([0, 1.2])
    plt.pause(0.02)
    plt.clf()

# for p3
u0 = np.array([0, 1])  ## initial directional vector
plt.plot(u0[0], u0[1], 'r.')
x_traj = sol2.y
plt.plot(t_all, x_traj[0, :])
plt.plot(t_all, x_traj[1, :])
plt.plot(t_all, x_traj[2, :])
plt.plot(t_all, x_traj[3, :])
plt.show()
theta = np.linspace(0, 2 * np.pi, 301)

for t in range(T):
    u = np.zeros(2)
    x_t = x_traj[:, t]
    u[0] = np.sin(x_t[0])
    u[1] = np.cos(x_t[0])
    plt.plot(np.cos(theta), np.sin(theta))
    plt.plot([0, u[0]], [0, u[1]], 'r-', linewidth=4)
    print(LA.norm(u, 2))
    plt.xlim([-1, 1])
    plt.ylim([0, 1.2])
    plt.pause(0.02)
    plt.clf()
