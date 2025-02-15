import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
from scipy.integrate import odeint
import numpy as np
from numpy import linalg as LA
from scipy import signal
from scipy.linalg import fractional_matrix_power

## Global variables
omega = 1
m = 1
n = 2
N = 100
tf = 10
dt = tf / N
t_traj = np.linspace(0, tf, N)

## Control input
u_traj = np.zeros((1, N - 1))
for t in range(N - 1):
    u_traj_t = np.sin(2 * t_traj[t])
    u_traj[:, t] = u_traj_t

## Continuous system
A1 = np.array([[0, 1],
               [0, 0]])

B1 = np.array([[0, 1]])

A2 = np.array([[0, omega],
               [-omega, 0]])

B2 = np.array([[0, 1]])
## Exact discretization
A1_d = np.array([[1, dt],
                 [0, 1]])

B1_d = np.array([dt * dt / 2, dt])

[D, V] = LA.eig(A2)
D = np.diag(D)
D2_d = la.expm(dt * D)
A2_d = np.real(V @ D2_d @ LA.inv(V))
B2_d = np.real((A2_d - np.eye(2)) @ LA.inv(A2)) @ np.array([[0], [1]])

## Sanity check with sys_to_dist function
A = np.array([[0, omega],
              [-omega, 0]])
B = np.array([[0], [1]])
C = np.eye(2)
D = np.zeros((2, 1))
sys = signal.StateSpace(A, B, C, D)
sysd = sys.to_discrete(dt)
A2_d_check = sysd.A
B2_d_check = sysd.B
print('A2_d from calculation', A2_d, '\n A2_d from scipy', A2_d_check)
print('B2_d from calculation', B2_d, '\n B2_d from scipy', B2_d_check)

ss = 1
x_exa_traj_1 = np.zeros((n, N))
x_exa_traj_2 = np.zeros((n, N))
for t in range(N - 1):
    x_exa_traj_1[:, t + 1] = A1_d @ x_exa_traj_1[:, t] + B1_d * np.sin(2 * t_traj[t])
    x_exa_traj_2[:, t + 1] = A2_d @ x_exa_traj_2[:, t] + np.array([B2_d_check[0, 0], B2_d_check[1, 0]]) * np.sin(
        2 * t_traj[t])
## Euler discretization
x_eul_traj_1 = np.zeros((n, N))
x_eul_traj_2 = np.zeros((n, N))
for t in range(N - 1):
    x_eul_traj_1[:, t + 1] = (np.eye(2) + dt * A1) @ x_eul_traj_1[:, t] + dt * B1 * np.sin(2 * t_traj[t])
    x_eul_traj_2[:, t + 1] = (np.eye(2) + dt * A2) @ x_eul_traj_2[:, t] + dt * B2 * np.sin(2 * t_traj[t])

## Analytical solution
x_an_traj_1 = np.zeros((n, N))
x_an_traj_2 = np.zeros((n, N))
for t in range(N):
    t_c = t_traj[t]
    x_an_traj_1[:, t] = np.array([t_c / 2 - np.sin(2 * t_c) / 4, 1 / 2 - np.cos(2 * t_c) / 2])
    x_an_traj_2[:, t] = np.array(
        [2 * np.sin(t_c) / 3 - np.sin(2 * t_c) / 3, 2 * np.cos(t_c) / 3 - 2 * np.cos(2 * t_c) / 3])
## Plotting
plt.subplot(1, 2, 1)
plt.plot(x_exa_traj_1[0, :], x_exa_traj_1[1, :], 'r.')
plt.plot(x_eul_traj_1[0, :], x_eul_traj_1[1, :], 'g-')
plt.plot(x_an_traj_1[0, :], x_an_traj_1[1, :], 'b-.')
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend(['Exact', 'Euler', 'Analytical'], fontsize=6)
plt.xlim(0, 5)
plt.ylim(0, 1.2)
plt.title('system 1')
plt.grid()

plt.subplot(1, 2, 2)
plt.plot(x_exa_traj_2[0, :], x_exa_traj_2[1, :], 'r.')
plt.plot(x_eul_traj_2[0, :], x_eul_traj_2[1, :], 'g-')
plt.plot(x_an_traj_2[0, :], x_an_traj_2[1, :], 'b-.')
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend(['Exact', 'Euler', 'Analytical'], fontsize=6)
plt.xlim(-1.5, 1.5)
plt.ylim(-2, 1)
plt.title('system 2')
plt.grid()
plt.show()

plt.subplot(2, 2, 1)
plt.plot(t_traj, x_exa_traj_1[0, :], 'r.')
plt.plot(t_traj, x_eul_traj_1[0, :], 'g-')
plt.plot(t_traj, x_an_traj_1[0, :], 'b-.')
plt.xlabel('t')
plt.ylabel('x1')
plt.legend(['Exact', 'Euler', 'Analytical'], fontsize=6)
plt.xlim(0, 10)
plt.ylim(0, 10)
plt.grid()
plt.title('system 1')

plt.subplot(2, 2, 2)
plt.plot(t_traj, x_exa_traj_1[1, :], 'r.')
plt.plot(t_traj, x_eul_traj_1[1, :], 'g-')
plt.plot(t_traj, x_an_traj_1[1, :], 'b-.')
plt.xlabel('t')
plt.ylabel('x2')
plt.legend(['Exact', 'Euler', 'Analytical'], fontsize=6)
plt.xlim(0, 10)
plt.ylim(-1.5, 1.5)
plt.grid()
plt.title('system 1')

plt.subplot(2, 2, 3)
plt.plot(t_traj, x_exa_traj_2[0, :], 'r.')
plt.plot(t_traj, x_eul_traj_2[0, :], 'g-')
plt.plot(t_traj, x_an_traj_2[0, :], 'b-.')
plt.xlabel('t')
plt.ylabel('x1')
plt.legend(['Exact', 'Euler', 'Analytical'], fontsize=6)
plt.xlim(0, 10)
plt.ylim(-1.5, 1.5)
plt.grid()
plt.title('system 2')

plt.subplot(2, 2, 4)
plt.plot(t_traj, x_exa_traj_2[1, :], 'r.')
plt.plot(t_traj, x_eul_traj_2[1, :], 'g-')
plt.plot(t_traj, x_an_traj_2[1, :], 'b-.')
plt.xlabel('t')
plt.ylabel('x2')
plt.legend(['Exact', 'Euler', 'Analytical'], fontsize=6)
plt.xlim(0, 10)
plt.ylim(-1.5, 1.5)
plt.grid()
plt.title('system 2')
plt.show()
