import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from context import vvert
from context import vkdv


# working parameters:
# dx = 10
# start_x = 0
# end_x = 150 km
# dz0 = 0.5
# start_z0 = 0
# end_z0 = 250 m
# 200 eigenvalue points
bathymetry = pd.DataFrame(pd.read_csv(
    "data/nws-bathymetry-5km.csv",
    names=["x", "depth"]
))
x = np.asarray(bathymetry.x)
depth = np.asarray(bathymetry.depth)

vert = vvert.VVerticalMode(
    dx=10,
    start_x=0,
    end_x=x[-1],
    dz0=0.5,
    start_z0=0,
    end_z0=-depth[0],
    n_eigen=50,
    rho_0=1000
)
vert.bathymetry = -np.interp(vert.x_grid, x, depth)
vert.compute_density("dht")
vert.compute_parameters()

# plot all of the parameters: these should be smooth functions
x_grid = vert.x_grid
plt.subplot(231)
plt.plot(
    x_grid/1000, vert.c, "-"
)
plt.title("$c$ parameter")
plt.subplot(232)
plt.plot(
    x_grid/1000, vert.q, "-"
)
plt.title("$q$ parameter")
plt.subplot(233)
plt.plot(
    x_grid/1000, vert.alpha, "-"
)
plt.title("$\\alpha$ parameter")
plt.subplot(234)
plt.plot(
    x_grid/1000, vert.beta, "-",
)
plt.title("$\\beta$ parameter")
plt.subplot(235)
plt.plot(
    x_grid / 1000, (vert.c / vert.q) * vert.q_grad, "-"
)
plt.title("$c Q_x / Q$ parameter")
plt.show()

# set up the Kdv object
test = vkdv.Kdv(
    dt=15, dx=50, start_x=0, end_x=150_000, start_t=0, end_t=24 * 60**2
)
test.set_initial_condition(
    np.array(
        - 20 * (1/4) * (1 + np.tanh((test.x_grid - 10_000) / 2000))
        * (1 - np.tanh((test.x_grid - 20_000) / 2000)),
        ndmin=2
    ).T
)

# set all the coefficients
test.a = np.array(vert.alpha, ndmin=2).T  # a = 3c / 2
test.b = np.array(vert.beta, ndmin=2).T  # b = c / 2
test.c = np.array(vert.c, ndmin=2).T
test.q = np.array(vert.q, ndmin=2).T
test.q_grad = np.array(vert.q_grad, ndmin=2).T
test.bathymetry_term = (2 * test.c / test.q) * test.q_grad  # 2c q_x / q

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

fig = plt.figure()
ax = plt.axes(xlim=(0, 60), ylim=(-300, 2.5))
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
