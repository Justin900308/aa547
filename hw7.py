import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
from scipy.integrate import odeint
import numpy as np
from numpy import linalg as LA
from scipy import signal
from scipy.linalg import fractional_matrix_power
from scipy import sparse

## Global parameters
k = 1
n = 4
N = 10000
tf = 1
rf = 1
dt = tf / N
A = np.array([[0, 0, 1, 0],
              [0, 0, 0, 1],
              [-2 * k, k, 0, 0],
              [k, -2 * k, 0, 0]])
x_tf = np.array([[rf], [0], [0], [0]])


## B matrix
def B_mat(
        b1: float,
        b2: float
) -> list:
    B = np.array([0, 0, b1, b2])
    B_col = np.array([[0], [0], [b1], [b2]])
    return B, B_col


def dist_sys(
        A: np.ndarray,
        B: np.ndarray
) -> list:
    C = np.eye(4)
    D = np.zeros((n, 1))

    sys = signal.StateSpace(A, B, C, D)
    sysd = sys.to_discrete(dt)
    Ad = sysd.A
    Bd = sysd.B

    return Ad, Bd


## W matrix
def W_mat(
        A: np.ndarray,
        B_col: np.ndarray
) -> np.ndarray:
    W = np.zeros((n, n))
    for k in range(N-1):
        W = W + LA.matrix_power(la.expm(dt * A), N - 1 - k) @ B_col @ B_col.T @ LA.matrix_power(la.expm(dt * A.T),
                                                                                                N - 1 - k)
    W = W * dt
    return W


## Case 1
b1 = 0
b2 = 1
x_traj = np.zeros((n, N))
[B1, B1_col] = B_mat(b1, b2)
[A_d, B1_d_col] = dist_sys(A, B1_col)
B1_d = np.array([B1_d_col[0, 0], B1_d_col[1, 0], B1_d_col[2, 0], B1_d_col[3, 0]])
W1 = W_mat(A, B1_col)
cond_W1= LA.norm(W1,2) * LA.norm(LA.inv(W1),2)
print('Cond of W:   ',cond_W1)
u1 = np.zeros(N)
x = np.zeros((n, N + 1))
t = np.linspace(0, tf, N + 1)
for k in range(N):
    u1[k] = B1_col.T @ la.expm((tf - t[k]) * A.T) @ la.inv(W1) @ x_tf
    x[:, k + 1] = A_d @ x[:, k] + B1_d * u1[k]
plt.subplot(2, 1, 1)
plt.plot(t, x[0, :])
plt.plot(t, x[1, :])
plt.plot(tf, x[0, N], 'r.')
plt.xlabel('time')
plt.ylabel('position')
plt.legend(['x1', 'x2', 'desired x1'], fontsize=8)
plt.grid()
plt.subplot(2, 1, 2)
plt.plot(t[0:N], u1[:])
plt.xlabel('time')
plt.ylabel('control input')
plt.legend(['u'])
plt.grid()
plt.show()

# ## Animation
# for k in range(N-1):
#     plt.plot(x[0, k], 0, 'r.')
#     plt.plot(x[1, k], 0, 'b.')
#     plt.xlim([-5, 5])
#     plt.ylim([-0.1, 0.1])
#     plt.pause(0.0001)
#     plt.clf()
# plt.plot(x[0, N], 0, 'r.')
# plt.plot(x[1, N], 0, 'b.')
# plt.xlim([-3, 3])
# plt.ylim([-0.1, 0.1])
# plt.show()


## Case 2
b1 = 0.99
b2 = 1
x_traj = np.zeros((n, N))
[B1, B1_col] = B_mat(b1, b2)
[A_d, B1_d_col] = dist_sys(A, B1_col)
B1_d = np.array([B1_d_col[0, 0], B1_d_col[1, 0], B1_d_col[2, 0], B1_d_col[3, 0]])
W1 = W_mat(A, B1_col)
cond_W1= LA.norm(W1,2) * LA.norm(LA.inv(W1),2)
print('Cond of W:   ',cond_W1)
u1 = np.zeros(N)
x = np.zeros((n, N + 1))
t = np.linspace(0, tf, N + 1)
for k in range(N):
    u1[k] = B1_col.T @ la.expm((tf - t[k]) * A.T) @ la.inv(W1) @ x_tf
    x[:, k + 1] = A_d @ x[:, k] + B1_d * u1[k]
plt.subplot(2, 1, 1)
plt.plot(t, x[0, :])
plt.plot(t, x[1, :])
plt.plot(tf, x[0, N], 'r.')
plt.xlabel('time')
plt.ylabel('position')
plt.legend(['x1', 'x2', 'desired x1'], fontsize=8)
plt.grid()
plt.subplot(2, 1, 2)
plt.plot(t[0:N], u1[:])
plt.xlabel('time')
plt.ylabel('control input')
plt.legend(['u'])
plt.grid()
plt.show()

## Animation
for k in range(N - 1):
    plt.plot(x[0, k], 0, 'r.')
    plt.plot(x[1, k], 0, 'b.')
    plt.xlim([-5, 5])
    plt.ylim([-0.1, 0.1])
    plt.pause(0.0001)
    plt.clf()
plt.plot(x[0, N], 0, 'r.')
plt.plot(x[1, N], 0, 'b.')
plt.xlim([-3, 3])
plt.ylim([-0.1, 0.1])
plt.show()
