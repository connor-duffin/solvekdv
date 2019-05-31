import numpy as np
import matplotlib.pyplot as plt

from context import kdv
from context import vert


vertical = vert.VerticalMode(0.1, 0, 300, 1000)
vertical.compute_density("lamb-yan-1")

vertical.find_vertical_mode()

vertical.compute_r10()
vertical.compute_r01()

print(
    (f"r10: {vertical.r10:.4f}\n"
    + f"r01: {vertical.r01:.4f}\n"
    + f"c:   {vertical.c:.4f}\n")
)

test = kdv.Kdv(
    dt=0.01, dx=0.01, start_x=0, end_x=1, start_t=0, end_t=1
)

test.set_initial_condition(np.cos(2 * np.pi * test.x_grid))
test.set_kdv_parameters(
    a=2 * vertical.r10 * vertical.c,
    b=vertical.r01
)

test.set_first_order_matrix()
test.set_third_order_matrix()
test.set_lhs_matrix()

u = np.zeros([test.n_x, test.n_t])
for i in range(test.n_t):
    if (i % int(0.1 * test.n_t)) == 0:
        print(f"Iteration {100 * i / test.n_t:.1f} % complete.")
    u[:, i] = test.solve_step()

plt.plot(u[:, 0])
plt.show()
