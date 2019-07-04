import numpy as np
import matplotlib.animation as animation
import matplotlib.pyplot as plt

from context import kdv
from context import vert


vertical = vert.VerticalMode(0.1, 0, 300, 1000)
vertical.compute_density("lamb-yan-1")

vertical.find_vertical_mode()

vertical.compute_r10()
vertical.compute_r01()

print(
    f"r10: {vertical.r10:.4f}\n"
    + f"r01: {vertical.r01:.4f}\n"
    + f"c:   {vertical.c:.4f}\n"
)

test = kdv.Kdv(
    dt=10, dx=50, start_x=-150000, end_x=150000, start_t=0, end_t=24 * 60**2
)

test.set_initial_condition(
    - 20 * (1/4) * ((1 + np.tanh((test.x_grid + 20000) / 2000))
    * (1 - np.tanh(test.x_grid / 2000)))
)

test.set_kdv_parameters(
    a=2 * vertical.r10 * vertical.c,
    b=vertical.r01,
    c=vertical.c
)

test.set_first_order_matrix()
test.set_third_order_matrix()
test.set_lhs_matrix()

u = np.zeros([test.n_x, test.n_t])
for i in range(test.n_t):
    if (i % int(0.1 * test.n_t)) == 0:
        print(f"Simulation {100 * i / test.n_t:.1f} % complete.")
    u[:, i] = test.solve_step()

fig = plt.figure()
ax = plt.axes(xlim=(0, 150), ylim=(-500, 10))
line, = plt.plot([], [])


def init():
    line.set_data([], [])
    return line,


def animate(i):
    line.set_data(test.x_grid / 1000, u[:, i])
    return line,


anim = animation.FuncAnimation(
    fig, animate, init_func=init, frames=8000, interval=2, blit=True
)
plt.show()
