import matplotlib.pyplot as plt
import numpy as np

from context import kdv


test = kdv.Kdv(
    dt=0.01, dx=0.01, start_x=0, end_x=1, start_t=0, end_t=1
)
test.set_initial_condition(np.cos(2 * np.pi * test.x_grid))
test.set_kdv_parameters()
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
