import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
from scipy.integrate import odeint
import numpy as np
from numpy import linalg as LA

################### P1

##Part a

A = np.array([[-1, 2],
              [0, -1]])

Q = np.eye(2)

P = la.solve_continuous_lyapunov(A, -Q)
[D, V] = LA.eig(P)
print('P:   ', P)

##Part b
cond_P = LA.norm(P, 2) * LA.norm(LA.inv(P), 2)
C = np.sqrt(cond_P)
[D1, V1] = LA.eig(la.fractional_matrix_power(P, -0.5) @ Q @ la.fractional_matrix_power(P, -0.5))
lambda_min = 0.5 * np.min(D1)

## Part c
T = 201
t = np.linspace(0, 10, T)
value1 = np.zeros(T)

for i in range(T):
    value1[i] = C * np.exp(-lambda_min * t[i])

############################## P2
Kf = 50
dt = 0.02
n = 4
m = 2
A1 = np.array([[1, 0, dt, 0],
               [0, 1, 0, dt],
               [0, 0, 1, 0],
               [0, 0, 0, 1]])
B1 = np.array([[0.5 * dt * dt, 0],
               [0, 0.5 * dt * dt],
               [dt, 0],
               [0, dt]])

L_0 = A1
L = np.zeros((n, (Kf - 1) * m))
L[0:n, 0:m] = B1
for k in range(Kf - 2):
    L_0 = L_0 @ A1
    L[0:n, (k + 1) * m:(k + 1) * m + m] = A1 @ L[0:n, k * m:k * m + m]


def u_vec(
        X_0: np.ndarray,
        X_f: np.ndarray
) -> np.ndarray:
    u_bar = np.zeros((m, Kf))
    u_bar = L.T @ LA.inv(L @ L.T) @ (-L_0 @ X_0 + X_f)
    return u_bar


######## Case 1:
X_0 = np.array([-1, 0, 0, 1])
X_f = np.array([1, 0, 0, -1])

X_traj1 = np.zeros((n, Kf))
X_traj1[:, 0] = X_0
u_bar = u_vec(X_0, X_f)
u_bar = np.flip(u_bar)
for k in range(Kf - 1):
    u_k = u_bar[m * k:m * (k + 1)]
    u_k = np.flip(u_k)
    X_traj1[:, k + 1] = A1 @ X_traj1[:, k] + B1 @ u_k

######## Case 2:
X_0 = np.array([-1, 0, 0, 1])
X_f = np.array([1, 0, 0, 1])

X_traj2 = np.zeros((n, Kf))
X_traj2[:, 0] = X_0
u_bar = u_vec(X_0, X_f)
u_bar = np.flip(u_bar)
for k in range(Kf - 1):
    u_k = u_bar[m * k:m * (k + 1)]
    u_k = np.flip(u_k)
    X_traj2[:, k + 1] = A1 @ X_traj2[:, k] + B1 @ u_k

######## Case 3:
X_0 = np.array([-1, 0, 0, 1])
X_f = np.array([0, -1, 1, 0])

X_traj3 = np.zeros((n, Kf))
X_traj3[:, 0] = X_0
u_bar = u_vec(X_0, X_f)
u_bar = np.flip(u_bar)
for k in range(Kf - 1):
    u_k = u_bar[m * k:m * (k + 1)]
    u_k = np.flip(u_k)
    X_traj3[:, k + 1] = A1 @ X_traj3[:, k] + B1 @ u_k


######################### Plotting
plt.subplot(1, 3, 1)
plt.plot(X_traj1[0,:], X_traj1[1,:])
plt.xlabel('x')
plt.ylabel('y')
plt.xlim(-1,1)
plt.ylim(-1,1)

plt.subplot(1, 3, 2)
plt.plot(X_traj2[0,:], X_traj2[1,:])
plt.xlabel('x')
plt.ylabel('y')
plt.xlim(-1,1)
plt.ylim(-1,1)
plt.subplot(1, 3, 3)
plt.plot(X_traj3[0,:], X_traj3[1,:])
plt.xlabel('x')
plt.ylabel('y')
plt.xlim(-1,1)
plt.ylim(-1,1)
plt.show()
