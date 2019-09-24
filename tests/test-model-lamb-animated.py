import numpy as np
import matplotlib.animation as animation
import matplotlib.pyplot as plt

from solvekdv import kdv
from solvekdv import vert


# solve the vertical model problem
vertical = vert.VerticalMode(0.1, 0, 300, 1000)
vertical.compute_density("lamb-yan-1")
vertical.find_vertical_mode()
vertical.compute_alpha()
vertical.compute_beta()
print(
    f"alpha: {vertical.alpha:.4f}\n"
    + f"beta: {vertical.beta:.4f}\n"
    + f"c:   {vertical.c:.4f}\n"
)

# initialize a Kdv class
soln = kdv.Kdv(
    dt=10, dx=50, x_start=-150000, x_end=150000, t_start=0, t_end=24 * 60**2
)
soln.set_initial_condition(
    np.array(
        -20 * (1/4)
        * (1 + np.tanh((soln.x_grid + 20000) / 2000))
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

# plot the animation
fig = plt.figure()
ax = plt.axes(xlim=(-150, 150), ylim=(-500, 10))
line, = plt.plot([], [])
plt.xlabel("$x$ (distance)")
plt.ylabel("$A$ (wave profile)")
plt.title(
    "KdV Zabusky-Kruskal solution\n" +
    f"(parameters: a = {soln.alpha:.5f}, b = {soln.beta:.5f}, c = {soln.c:.5f})\n" +
    f"(grid: {soln.n_x} $\\times$ {soln.n_t} ($x \\times t$))"
)


def init():
    line.set_data([], [])
    return line,


def animate(i):
    line.set_data(soln.x_grid / 1000, u[:, i])
    return line,


anim = animation.FuncAnimation(
    fig, animate, init_func=init, frames=8000, interval=2, blit=True
)
plt.show()
