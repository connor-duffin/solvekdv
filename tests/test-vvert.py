import matplotlib.pyplot as plt

from context import vvert


vertical_test = vvert.VVerticalMode(
    dx=1500, start_x=0, end_x=150_000, dz0=0.5, start_z0=0, end_z0=250,
    n_interp=200, rho_0=1000
)
vertical_test.compute_bathymetry(vertical_test.x_grid * 5e-4)
vertical_test.compute_density("lamb-yan-1")
vertical_test.compute_parameters()

x_grid = vertical_test.x_grid
x_subs_index = vertical_test.x_subs_index
plt.subplot(221)
plt.plot(
    x_grid[x_subs_index]/1000, vertical_test.c[x_subs_index], "o",
    x_grid/1000, vertical_test.c, "-"
)
plt.title("$c$ parameter")

plt.subplot(222)
plt.plot(
    x_grid[x_subs_index]/1000, vertical_test.q[x_subs_index], "o",
    x_grid/1000, vertical_test.q, "-"
)
plt.title("$q$ parameter")

plt.subplot(223)
plt.plot(
    x_grid[x_subs_index]/1000, vertical_test.beta[x_subs_index], "o",
    x_grid/1000, vertical_test.beta, "-",
)
plt.title("$r_{01}$ parameter")

plt.subplot(224)
plt.plot(
    x_grid[x_subs_index]/1000, vertical_test.alpha[x_subs_index], "o",
    x_grid/1000, vertical_test.alpha, "-"
)
plt.title("$r_{10}$ parameter")
plt.show()

plt.plot((vertical_test.c / vertical_test.q) * vertical_test.q_grad)
plt.show()
