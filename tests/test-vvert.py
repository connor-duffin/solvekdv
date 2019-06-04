import matplotlib.pyplot as plt

from context import vvert


vertical_test = vvert.VVerticalMode(
    dx=100, start_x=0, end_x=150000, dz0=1, start_z0=0, end_z0=500,
    n_interp=50, rho_0=1000
)

vertical_test.compute_bathymetry(vertical_test.x_grid * 2e-3)
vertical_test.compute_density("lamb-yan-1")
vertical_test.compute_parameters()

x_grid = vertical_test.x_grid
plt.subplot(221)
plt.plot(x_grid/1000, vertical_test.c)
plt.title("$c$ parameter")

plt.subplot(222)
plt.plot(x_grid/1000, vertical_test.q)
plt.title("$q$ parameter")

plt.subplot(223)
plt.plot(x_grid/1000, vertical_test.r01)
plt.title("$r_{01}$ parameter")

plt.subplot(224)
plt.plot(x_grid/1000, vertical_test.r10)
plt.title("$r_{10}$ parameter")
plt.show()
