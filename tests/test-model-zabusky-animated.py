import numpy as np
import matplotlib.animation as animation
import matplotlib.pyplot as plt

from context import kdv


test = kdv.Kdv(
    dt=0.001, dx=0.001, start_x=0, end_x=2, start_t=0, end_t=20
)

test.set_initial_condition(np.cos(2 * np.pi * test.x_grid / 2))
test.set_kdv_parameters(a=1, b=0.022**2, c=0)
test.set_first_order_matrix()
test.set_third_order_matrix()
test.set_lhs_matrix()

u = np.zeros([test.n_x, test.n_t])
for i in range(test.n_t):
    if (i % int(0.1 * test.n_t)) == 0:
        print(f"Simulation {100 * i / test.n_t:.1f} % complete.")
    u[:, i] = test.solve_step()

fig = plt.figure()
ax = plt.axes(xlim=(0, 2), ylim=(-2, 4))
line, = plt.plot([], [])


def init():
    line.set_data([], [])
    return line,


def animate(i):
    line.set_data(test.x_grid, u[:, i])
    return line,


anim = animation.FuncAnimation(
    fig, animate, init_func=init, frames=20000, interval=1, blit=True
)
plt.show()
