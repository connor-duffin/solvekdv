import numpy as np
import matplotlib.pyplot as plt

from solvekdv import kdv


test = kdv.Kdv(
    dt=0.001, dx=0.01, x_start=0, x_end=2, t_start=0, t_end=10
)
test.set_initial_condition(
    np.cos(np.pi * test.x_grid)
)
test.alpha = 1
test.beta = 0.022**2
test.c = 0
test.set_first_order_matrix()
test.set_third_order_matrix()

u = np.zeros((test.n_x, test.n_t))
for i in range(test.n_t):
    print(f"\rIteration {i + 1:5} / {test.n_t}", end="")
    u[:, i] = test.solve_step_im_euler()
    if np.isnan(np.sum(u[:, i])):
        print("\nNaN's encountered. Breaking loop")
        break

print()

xmesh, ymesh = np.meshgrid(test.x_grid, test.t_grid)

plt.figure(figsize=(6, 5))
plt.pcolormesh(xmesh, ymesh, u.transpose())
plt.colorbar()
plt.xlabel("x")
plt.ylabel("t")
plt.title(
    f"KdV, (alpha, beta, c)=({test.alpha:.5f}, {test.beta:.5f}, {test.c:.5f})"
)
plt.show()
