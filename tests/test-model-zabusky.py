import numpy as np
import matplotlib.pyplot as plt

from context import kdv


test = kdv.Kdv(
    dt=0.01, dx=0.01, start_x=0, end_x=1, start_t=0, end_t=10
)
test.set_initial_condition(
    np.array(np.cos(2 * np.pi * test.x_grid), ndmin=2).T
)
test.a = 1
test.b = 0.022**2
test.c = 0
test.set_first_order_matrix()
test.set_third_order_matrix()
test.set_lhs_matrix()

u = np.zeros((test.n_x, test.n_t))
for i in range(test.n_t):
    print(f"\rIteration {i + 1:5} / {test.n_t}", end="")
    u[:, i] = test.solve_step()

print()

xmesh, ymesh = np.meshgrid(test.x_grid, test.t_grid)

plt.figure()
plt.pcolormesh(xmesh, ymesh, u.transpose(), cmap="RdBu_r")
plt.colorbar()
plt.xlabel("x")
plt.ylabel("t")
plt.title(
    f"KdV, ($a$, $b$, $c$)=({test.a:.5f}, {test.b:.5f}, {test.c:.5f})"
)
plt.show()
