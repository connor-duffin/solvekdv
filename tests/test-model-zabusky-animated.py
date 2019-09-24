import numpy as np
import matplotlib.animation as animation
import matplotlib.pyplot as plt

from solvekdv import kdv


test = kdv.Kdv(
    dt=0.001, dx=0.001, x_start=0, x_end=2, t_start=0, t_end=10
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

u = np.zeros([test.n_x, test.n_t])
for i in range(test.n_t):
    print(f"\rIteration {i + 1:5} / {test.n_t}", end="")
    u[:, i] = test.solve_step_imex()

print()

fig = plt.figure()
ax = plt.axes(xlim=(0, 2), ylim=(-2, 4))
line, = plt.plot([], [])
plt.xlabel("$x$ (distance)")
plt.ylabel("$A$ (amplitude)")
plt.title(
    "KdV Zabusky-Kruskal solution\n" +
    f"(parameters: alpha = {test.alpha:.5f}, b = {test.beta:.5f}, c = {test.c:.5f})\n"
    f"(grid: {test.n_x} $\\times$ {test.n_t} ($x \\times t$))"
)


def init():
    line.set_data([], [])
    return line,


def animate(i):
    line.set_data(test.x_grid, u[:, i])
    return line,


anim = animation.FuncAnimation(
    fig, animate, init_func=init, frames=10000, interval=1, blit=True
)
plt.show()
