import numpy as np
import matplotlib.pyplot as plt

from solvekdv import kdv
from solvekdv import vert


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
soln = kdv.Kdv(
    dt=10, dx=50, x_start=-150000, x_end=150000,
    t_start=0, t_end=24 * 60**2
)
soln.set_initial_condition(
    np.array(
        - 20 * (1/4) * (1 + np.tanh((soln.x_grid + 20000) / 2000))
        * (1 - np.tanh(soln.x_grid / 2000)), ndmin=2
    ).T
)

soln.alpha = vertical.alpha
soln.beta = vertical.beta
soln.c = vertical.c

soln.set_first_order_matrix()
soln.set_third_order_matrix()
soln.set_imex_lhs_matrix()

# run the solver
u = np.zeros((soln.n_x, soln.n_t))
for i in range(soln.n_t):
    print(f"\rIteration {i + 1:5} / {soln.n_t}", end="")
    u[:, i] = soln.solve_step_imex()

print()

# plot results
xmesh, ymesh = np.meshgrid(soln.x_grid, soln.t_grid)

plt.figure()
plt.pcolormesh(xmesh, ymesh, u.transpose(), cmap="RdBu_r")
plt.colorbar()
plt.xlabel("x")
plt.ylabel("t")
plt.title(
    f"KdV, ($\alpha$, $\beta$, $c$)=({soln.alpha:.5f}, {soln.beta:.5f}, {soln.c:.5f})"
)
plt.show()
