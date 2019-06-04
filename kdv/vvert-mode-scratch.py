import numpy as np
import matplotlib.pyplot as plt

from scipy import linalg


RHO = 1000
GRAV = 9.81

x_grid = np.linspace(0, 1.5e5, 1000)
z_grid = np.linspace(0, 300, 300)
density = (
    1027.31 - 3.3955 * np.exp((z_grid - 300) / 50)
)

# starting at depth = 300 m
# want to interpolate to 50 grid points on this domain.
# so we take a subset index
xi_subset = range(0, 1000, 10)
bathymetry = z_grid[-1] - x_grid * 5e-4
plt.plot(
    x_grid[xi_subset]/1000, bathymetry[xi_subset], "o",
    x_grid/1000, bathymetry, "-"
)
plt.ylim(300, 0)
plt.xlabel("Distance (km)")
plt.ylabel("Depth (m)")
plt.show()

c = np.zeros(len(xi_subset))
r01 = np.zeros(len(xi_subset))
r10 = np.zeros(len(xi_subset))
q = np.zeros(len(xi_subset))

for i in range(len(xi_subset)):
    z_grid_temp = z_grid[z_grid <= bathymetry[xi_subset[i]]]
    dz = z_grid_temp[1] - z_grid_temp[0]
    n_z_temp = len(z_grid_temp)

    density_temp = (
        1027.31 - 3.3955 * np.exp((z_grid_temp - 300) / 50)
    )
    grad_density_temp = np.gradient(density_temp, dz)
    second_diff = - 1 / (dz**2) * (
            np.diag(np.full(n_z_temp - 1, 1), k=-1)
            + np.diag(np.full(n_z_temp, -2), k=0)
            + np.diag(np.full(n_z_temp - 1, 1), k=1)
        )
    scale = np.diag(np.full(n_z_temp, (- GRAV / RHO) * grad_density_temp))
    eigenvalue, phi = linalg.eigh(second_diff, b=scale, eigvals=(0, 0))
    phi = np.ndarray.flatten(phi)
    phi /= np.max(phi)
    phi_grad = np.gradient(phi, dz)
    c[i] = np.sqrt(1 / eigenvalue[0])
    r10[i] = (
        (3 / 2) * (np.trapz(np.power(phi_grad, 3), dx=dz)
        / np.trapz(np.power(phi_grad, 2), dx=dz))
    )
    r01[i] = (
        (c[i] / 2) * (np.trapz(np.power(phi, 2), dx=dz)
        / np.trapz(np.power(phi_grad, 2), dx=dz))
    )
    q[i] = np.sqrt(
        c[0]**3 * np.trapz(np.power(phi_grad, 2), dx=dz)
        / (c[i]**3 * np.trapz(np.power(phi_grad, 2), dx=dz))
    )


plt.plot(x_grid[xi_subset]/1000, c)
plt.show()
plt.plot(x_grid[xi_subset]/1000, r01)
plt.show()
plt.plot(x_grid[xi_subset]/1000, q)
plt.show()
plt.plot(x_grid[xi_subset]/1000, r10)
plt.show()
