import matplotlib.pyplot as plt
import numpy as np

from context import kdv


test = kdv.Kdv(
    dt=0.01, dx=0.01, start_x=0, end_x=1, start_t=0, end_t=10
)
test.set_initial_condition(np.array(np.cos(2 * np.pi * test.x_grid), ndmin=2).T)
test.a = 1
test.b = 0.022**2
test.c = 0
test.set_first_order_matrix()
test.set_third_order_matrix()
test.set_lhs_matrix()

u = np.zeros([test.n_x, test.n_t])
for i in range(test.n_t):
    print(f"\rIteration {i + 1:5} / {test.n_t}", end="")
    u[:, i] = test.solve_step()

plot_initial, = plt.plot(test.x_grid, u[:, 0])
plot_400, = plt.plot(test.x_grid, u[:, 200])
plot_800, = plt.plot(test.x_grid, u[:, 400])
plt.xlabel("$x$ (distance)")
plt.ylabel("$A$ (amplitude)")
plt.title(
    f"Solution for KdV equation, a = {test.a}, b = {test.b:.4f}, c = {test.c}"
)
plt.legend(
    [plot_initial, plot_400, plot_800],
    ["Initial conditions", "$t = 2$", "$t = 4$"]
)
plt.show()
