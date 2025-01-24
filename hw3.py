import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
from scipy.integrate import odeint
import numpy as np
from numpy import linalg as LA

n = 4  # state dimension

A = np.array([[-2, 1, 0, 1],
              [1, -2, 1, 0],
              [0, 1, -2, 1],
              [1, 0, 1, -2]])

[ls_A, vs_A] = la.eig(A)

ls_A = ls_A.real
vs_A = vs_A.real

print(ls_A)
print(vs_A)

################################


x_0 = np.array([2, 0.9, 0.5, 2.5])  # initial state
#
N = 100  # time discretization size
T = 10  # final time
dt = T / N  # grid size
#
t = np.linspace(0, T, N)  # time array
#
c_0 = LA.inv(vs_A) @ x_0  # components of the initial condition x_0 along each mode (eigenvector)
c_0 = vs_A.T @ x_0
c_t = np.zeros([n, N])  # component of the solution along each mode.
for i in range(n):
    c_t[i] = np.exp(t * ls_A[i]) * c_0[i]

x_t = np.zeros((4, N))
for i in range(n):
    x_t = vs_A @ c_t
plt.figure()
for i in range(n):
    plt.plot(t, c_t[i], label=r"$\lambda_%i = %.1f$" % (i + 1, ls_A[i]))

plt.legend(fontsize=16, ncols=2)
plt.show()
#
# ################################################
#
## mode,data,time
xt_modes = np.zeros([n, n, N])
for i in range(n):
    xt_modes[i, :, :] = np.outer(vs_A[:, i], c_t[i, :])  # compute x_i(t)



#
#
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
for i in range(4):
    plt.plot(t, xt_modes[i, 0, :], label=r'$x^{(%i)}_1(t)$' % (i + 1))

plt.legend(fontsize=16)

plt.subplot(2, 2, 2)
for i in range(4):
    plt.plot(t, xt_modes[i, 1, :], label=r'$x^{(%i)}_2(t)$' % (i + 1))

plt.legend(fontsize=16)

plt.subplot(2, 2, 3)
for i in range(4):
    plt.plot(t, xt_modes[i, 2, :], label=r'$x^{(%i)}_3(t)$' % (i + 1))

plt.legend(fontsize=16)

plt.subplot(2, 2, 4)
for i in range(4):
    plt.plot(t, xt_modes[i, 3, :], label=r'$x^{(%i)}_4(t)$' % (i + 1))

plt.legend(fontsize=16)

plt.show()

# #######################################################
xt_modes_sum = np.sum(xt_modes, 0)


def dyn(x, t):
    return A @ x


xt_ode = odeint(dyn, x_0, t).T



plt.subplot(1, 2, 1)
plt.plot(t, xt_modes_sum[0], label=r'$x_1(t)~(modes)$', lw=2)
plt.plot(t, xt_modes_sum[1], label=r'$x_2(t)~(modes)$', lw=2)
plt.plot(t, xt_modes_sum[2], label=r'$x_3(t)~(modes)$', lw=2)
plt.plot(t, xt_modes_sum[3], label=r'$x_4(t)~(modes)$', lw=2)
plt.legend(fontsize=8)
plt.subplot(1, 2, 2)
plt.plot(t, xt_ode[0], label=r'$x_1(t)~(ode)$', lw=2, ls='--')
plt.plot(t, xt_ode[1], label=r'$x_2(t)~(ode)$', lw=2, ls='--')
plt.plot(t, xt_ode[2], label=r'$x_3(t)~(ode)$', lw=2, ls='--')
plt.plot(t, xt_ode[3], label=r'$x_4(t)~(ode)$', lw=2, ls='--')
plt.legend(fontsize=8)



plt.show()
