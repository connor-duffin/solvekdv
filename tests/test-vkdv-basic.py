import numpy as np
import pandas as pd

from context import vvert
from context import vkdv


bathymetry = pd.DataFrame(pd.read_csv(
    "data/nws-bathymetry-5km.csv",
    names=["x", "depth"]
))
x = np.asarray(bathymetry.x)
depth = np.asarray(bathymetry.depth)

vert = vvert.VVerticalMode(
    dx=40,
    x_start=0,
    x_end=x[-1],
    dz0=0.5,
    z0_start=0,
    z0_end=-depth[0],
    n_eigen=200,
    rho_0=1000
)
vert.bathymetry = -np.interp(vert.x_grid, x, depth)
vert.initialize_dht_density()
vert.compute_parameters()

# set up the Kdv object
test = vkdv.Kdv(
    dt=15,
    dx=40,
    x_start=0,
    x_end=x[-1],
    t_start=0,
    t_end=3 * 24 * 60**2
)
test.set_initial_condition(
    np.array(
        20 * np.sin(2 * np.pi * test.x_grid / x[-1]),
        ndmin=2
    ).T
)
# set all the coefficients
test.a = np.array(vert.alpha, ndmin=2).T  # a = 3c / 2
test.b = np.array(vert.beta, ndmin=2).T  # b = c / 2
test.c = np.array(vert.c, ndmin=2).T
test.q = np.array(vert.q, ndmin=2).T
test.q_grad = np.array(vert.q_grad, ndmin=2).T
test.bathymetry_term = 2 * (test.c / test.q) * test.q_grad  # 2 c q_x / q

# set all the matrices
test.set_first_order_matrix()
test.set_third_order_matrix()
test.set_lhs_matrix()

u = np.zeros((test.n_x, test.n_t))
for i in range(test.n_t):
    print(f"\rIteration {i + 1:5} / {test.n_t}", end="")
    u[:, i] = test.solve_step()
    if (np.isnan(np.sum(u[:, i]))):
        print(f"Simulation failed at i = {i}: nan's appeared. Exiting.")
        break

print()
