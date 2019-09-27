import numpy as np
import matplotlib.pyplot as plt

from solvekdv import kdv


test = kdv.Kdv(
    dt=0.01, dx=0.01, x_start=0, x_end=2, t_start=0, t_end=10
)
test.set_initial_condition(
    np.array(np.cos(np.pi * test.x_grid), ndmin=2).T
)
test.alpha = 1
test.beta = 0.022**2
test.c = 0
test.set_first_order_matrix()
test.set_third_order_matrix()
test.set_imex_lhs_matrix()

u = np.zeros((test.n_x, test.n_t))
for i in range(test.n_t):
    print(f"\rIteration {i + 1:5} / {test.n_t}", end="")
    u[:, i] = test.solve_step_imex()

print()

xmesh, ymesh = np.meshgrid(test.x_grid, test.t_grid)
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.pcolormesh(xmesh, ymesh, u.T)
plt.colorbar(im, ax=ax)
ax.set_xlabel("x")
ax.set_ylabel("t")
ax.set_title(
    f"KdV, (alpha, beta, c) = "
    + f"({test.alpha:.5f}, {test.beta:.5f}, {test.c:.5f})\n"
    + "IMEX method"
)
fig.savefig("outputs/zk-imex.png", dpi=500)
