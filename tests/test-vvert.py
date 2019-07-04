import matplotlib.pyplot as plt

from context import vvert


# working parameters:
# dx = 10
# start_x = 0
# end_x = 150 km
# dz0 = 0.5
# start_z0 = 0
# end_z0 = 250 m
# 200 eigenvalue points
vert = vvert.VVerticalMode(
    dx=10, start_x=0, end_x=150_000, dz0=0.5, start_z0=0, end_z0=250,
    n_eigen=200, rho_0=1000
)

vert.compute_bathymetry(vert.x_grid * 5e-4)
vert.compute_density("lamb-yan-1")
vert.compute_parameters()

# plot all of the parameters: these should be smooth
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
plt.plot((vert.c / vert.q) * vert.q_grad)
plt.title("$c Q_x / Q$ parameter")
plt.show()

