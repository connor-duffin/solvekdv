import matplotlib.pyplot as plt

from solvekdv import vvert


vert = vvert.VVerticalMode(
    dx=800,
    x_start=0,
    x_end=150_000,
    dz0=0.5,
    z0_start=0,
    z0_end=250,
    n_eigen=200,
    rho_0=1000
)

vert.bathymetry = -250 + (vert.x_grid * 5e-4)
vert.initialize_lamb_density()
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
plt.plot(x_grid / 1000, (vert.c / vert.q) * vert.q_grad)
plt.title("$c Q_x / Q$ parameter")
plt.show()
