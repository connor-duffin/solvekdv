import numpy as np
import matplotlib.animation as animation
import matplotlib.pyplot as plt

from context import kdv
from context import vert


# solve the vertical model problem
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

# initialize a Kdv class
soln = kdv.Kdv(
    dt=10, dx=50, start_x=-150000, end_x=150000, start_t=0, end_t=24 * 60**2
)
soln.set_initial_condition(
    np.array(
        - 20 * (1/4) * (1 + np.tanh((soln.x_grid + 20000) / 2000))
        * (1 - np.tanh(soln.x_grid / 2000)), ndmin=2
    ).T
)

soln.a = vertical.alpha
soln.b = vertical.beta
soln.c = vertical.c

soln.set_first_order_matrix()
soln.set_third_order_matrix()
soln.set_lhs_matrix()

# run the solver
u = np.zeros((soln.n_x, soln.n_t))
for i in range(soln.n_t):
    print(f"\rIteration {i + 1:5} / {soln.n_t}", end="")
    u[:, i] = soln.solve_step()

print()

# plot the animation
fig = plt.figure()
ax = plt.axes(xlim=(-150, 150), ylim=(-500, 10))
line, = plt.plot([], [])
plt.xlabel("$x$ (distance)")
plt.ylabel("$A$ (wave profile)")
plt.title(
    "KdV Zabusky-Kruskal solution\n" +
    f"(parameters: a = {soln.a:.5f}, b = {soln.b:.5f}, c = {soln.c:.5f})\n"
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
