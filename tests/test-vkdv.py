import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from solvekdv import vvert
from solvekdv import vkdv


bathymetry = pd.DataFrame(pd.read_csv(
    "data/nws-bathymetry-5km.csv",
    names=["x", "depth"]
))
x = np.asarray(bathymetry.x)
depth = np.asarray(bathymetry.depth)

vert = vvert.VVerticalMode(
    dx=40,
    x_start=0,
    x_end=x[-1],
    dz0=0.5,
    z0_start=0,
    z0_end=-depth[0],
    n_eigen=200,
    rho_0=1000
)
vert.bathymetry = -np.interp(vert.x_grid, x, depth)
vert.initialize_dht_density()
vert.compute_parameters()

# plot all of the parameters: these should be smooth functions
x_grid = vert.x_grid
plt.subplot(231)
plt.plot(x_grid / 1000, vert.c)
plt.title("$c$ parameter")
plt.subplot(232)
plt.plot(x_grid / 1000, vert.q)
plt.title("$q$ parameter")
plt.subplot(233)
plt.plot(x_grid / 1000, vert.alpha)
plt.title("$\\alpha$ parameter")
plt.subplot(234)
plt.plot(x_grid / 1000, vert.beta)
plt.title("$\\beta$ parameter")
plt.subplot(235)
plt.plot(x_grid / 1000, (2 * vert.c / vert.q) * vert.q_grad, "-")
plt.title("$2 c Q_x / Q$ parameter")
plt.show()

# set up the Kdv object
test = vkdv.Kdv(
    dt=15, dx=40, x_start=0, x_end=x[-1], t_start=0, t_end=3 * 24 * 60**2
)
test.set_initial_condition(
    np.array(
        20 * np.sin(2 * np.pi * test.x_grid / x[-1]),
        ndmin=2
    ).T
)
# test.set_initial_condition(
#     np.array(
#         - 20 * (1/4) * (1 + np.tanh((test.x_grid - 10_000) / 2000))
#         * (1 - np.tanh((test.x_grid - 20_000) / 2000)),
#         ndmin=2
#     ).T
# )

# set all the coefficients
test.alpha = np.array(vert.alpha, ndmin=2).T  # a = 3c / 2
test.beta = np.array(vert.beta, ndmin=2).T  # b = c / 2
test.c = np.array(vert.c, ndmin=2).T
test.q = np.array(vert.q, ndmin=2).T
test.q_grad = np.array(vert.q_grad, ndmin=2).T
test.bathymetry_term = 2 * (test.c / test.q) * test.q_grad  # 2 c q_x / q

# set all the matrices
test.set_first_order_matrix()
test.set_third_order_matrix()
test.set_lhs_matrix()

u = np.zeros((test.n_x, test.n_t))
for i in range(test.n_t):
    print(f"\rIteration {i + 1:5} / {test.n_t}", end="")
    u[:, i] = test.solve_step()
    if (np.isnan(np.sum(u[:, i]))):
        print(f"Simulation failed at i = {i}: nan's appeared. Exiting.")
        break

print()

fig = plt.figure()
ax = plt.axes(xlim=(0, 100), ylim=(-300, 40))
line, = plt.plot([], [])


def init():
    line.set_data([], [])
    return line,


def animate(i):
    line.set_data(test.x_grid / 1000, u[:, i])
    return line,


anim = animation.FuncAnimation(
    fig, animate, init_func=init, frames=test.n_t, interval=2, blit=True
)
plt.show()
