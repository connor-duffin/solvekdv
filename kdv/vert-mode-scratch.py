import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sparse

from scipy.integrate import solve_ivp
from scipy.integrate import trapz
from scipy.optimize import least_squares
from scipy.optimize import root
from scipy.linalg import eigh
from scipy.sparse.linalg import eigs
from scipy.sparse.linalg import eigsh


def system(t, y):
    phi0, phi1, lmbda = y[0], y[1], y[2]
    return([phi1, -lmbda**2 * phi0 * np.exp(-t), 0])


def target(x):
    a, b = x[0], x[1]
    rk_run = solve_ivp(
        system, [0, 1], [0, a, b], t_eval=np.linspace(0, 1, 100)
    )
    return rk_run.y[0, -1]**2


soln = least_squares(target, [1, 10])
output = solve_ivp(
    system, [0, 1], [0, soln.x[0], soln.x[1]], t_eval=np.linspace(0, 1, 100)
)
plt.plot(output.y[0, :])
plt.plot(output.y[1, :])
plt.plot(np.gradient(output.y[0, :], 0.01))
plt.show()


dx = 0.0001
x_grid = np.arange(0, 1, dx)
n_x = len(x_grid)

# sparse version
A = - 1 / (dx**2) * sparse.diags(
    diagonals=[np.full(n_x - 1, 1), np.full(n_x, -2), np.full(n_x - 1, 1)],
    offsets=[-1, 0, 1],
    format="csc",
    dtype=np.float64
)
M = sparse.diags(
    2 / (np.exp(-x_grid) + np.exp(x_grid)),
    0,
    format="csc",
    dtype=np.float64
)

%timeit kappa, phi = eigsh(A, k=1, which="SM")
phi = np.ndarray.flatten(phi)

# dense is faster
A = - 1 / (dx**2) * (
    np.diag(np.full(n_x - 1, 1), k=-1)
    + np.diag(np.full(n_x, -2), k=0)
    + np.diag(np.full(n_x - 1, 1), k=1)
)
M = np.diag(np.full(n_x, 2 / (np.exp(-x_grid) + np.exp(-x_grid))))
%timeit lmbda, phi = eigh(A, b=M, eigvals=(0, 0))
