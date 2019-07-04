"""
================================
KdV IMEX --- Lamb's formulation
================================

This solves the KdV equation using my own Implicit-Explicit routine, as in
Durran and Blossey (2012).
"""
import numpy as np
import matplotlib.pyplot as plt

from context import kdv
from context import vert


# solve the vertical mode problem
vertical = vert.VerticalMode(0.1, 0, 500, 1000)
vertical.compute_density("lamb-yan-1")
vertical.find_vertical_mode()
vertical.compute_alpha()
vertical.compute_beta()
print(
    f"alpha: {vertical.alpha:.4f}\n"
    + f"beta: {vertical.beta:.4f}\n"
    + f"c:   {vertical.c:.4f}\n"
)

# set simulation vars, initial conditions, and parameters
test = kdv.Kdv(
    dt=10, dx=50, start_x=-150000, end_x=150000,
    start_t=0, end_t=24 * 60**2
)
test.set_initial_condition(
    np.array(- 20 * (1/4) * (1 + np.tanh((test.x_grid + 20000) / 2000))
    * (1 - np.tanh(test.x_grid / 2000)), ndmin=2).T
)
test.set_kdv_parameters(
    a=vertical.alpha,
    b=vertical.beta,
    c=vertical.c
)
test.set_first_order_matrix()
test.set_third_order_matrix()
test.set_lhs_matrix()

# run the solver
u = np.zeros([test.n_x, test.n_t])
for i in range(test.n_t):
    print(f"\rIteration {i + 1:5} / {test.n_t}", end="")
    u[:, i] = test.solve_step()

print()
# plot results
xmesh, ymesh = np.meshgrid(test.x_grid, test.t_grid)
plt.figure()
plt.pcolormesh(xmesh, ymesh, u.transpose(), cmap="viridis_r")
plt.colorbar()
plt.xlabel("x")
plt.ylabel("t")
plt.title(
    "KdV, ($a$, $b$, $c$)=(%g, %g, %g)" % (test.a, test.b, test.c)
)
plt.show()


plt.plot(test.x_grid, u[:, 1000])
plt.plot(test.x_grid, u[:, 2000])
plt.plot(test.x_grid, u[:, 3000])
plt.show()
