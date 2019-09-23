import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
import scipy.interpolate as interp


RHO = 1000
GRAV = 9.81

# setup grids for X = [0, 150,000] m, Z = [0, 500] m
x_grid = np.linspace(0, 1.5e5, 1000)
z_grid = np.linspace(0, 500, 500)

# compute the density over the z values
density = (
    1027.31 - 3.3955 * np.exp((z_grid - 300) / 50)
)

# want to interpolate from 50 grid points on this domain
x_subset = list(range(0, len(x_grid), 20)) + [len(x_grid) - 1]

# this is the approximate shelf floor
bathymetry = z_grid[-1] - x_grid * 2e-3
plt.plot(
    x_grid[x_subset]/1000, bathymetry[x_subset], "o",
    x_grid/1000, bathymetry, "-"
)
plt.ylim(500, 0)
plt.xlabel("Distance (km)")
plt.ylabel("Depth (m)")
plt.show()

# initialize things
phi0_grad = None
c = np.zeros(len(x_subset))
r01 = np.zeros(len(x_subset))
r10 = np.zeros(len(x_subset))
q = np.zeros(len(x_subset))

for i in range(len(x_subset)):
    # select the z-grid of interest
    z_grid_temp = z_grid[z_grid <= bathymetry[x_subset[i]]]
    dz = z_grid_temp[1] - z_grid_temp[0]
    n_z = len(z_grid_temp)
    # make sure we are getting enough points in our domain
    print(n_z)

    # compute the density on the z-grid
    density_temp = (
        1027.31 - 3.3955 * np.exp((z_grid_temp - 300) / 50)
    )
    density_grad_temp = np.gradient(density_temp, dz)

    # finite diff matrix on the z-grid
    second_diff_temp = - 1 / (dz**2) * (
        np.diag(np.full(n_z - 1, 1), k=-1)
        + np.diag(np.full(n_z, -2), k=0)
        + np.diag(np.full(n_z - 1, 1), k=1)
    )
    scale = np.diag(np.full(n_z, (- GRAV / RHO) * density_grad_temp))

    # eigh faster than any sparse stuff
    eigenvalue, phi = la.eigh(second_diff_temp, b=scale, eigvals=(0, 0))
    phi = np.ndarray.flatten(phi)
    phi /= np.max(phi)
    phi_grad = np.gradient(phi, dz)

    # set the initial element (to be used in computing Q)
    if (i == 0):
        phi0_grad = phi_grad

    # compute c, r10, r01, Q
    c[i] = np.sqrt(1 / eigenvalue[0])
    r10[i] = (
        (3 * c[i] / 2) * (np.trapz(np.power(phi_grad, 3), dx=dz)
        / np.trapz(np.power(phi_grad, 2), dx=dz))
    )
    r01[i] = (
        (c[i] / 2) * (np.trapz(np.power(phi, 2), dx=dz)
        / np.trapz(np.power(phi_grad, 2), dx=dz))
    )
    q[i] = np.sqrt(
        c[0]**3 * np.trapz(np.power(phi0_grad, 2), dx=dz)
        / (c[i]**3 * np.trapz(np.power(phi_grad, 2), dx=dz))
    )

# interpolate out the other parameters, from the grid
f_c = interp.interp1d(x_grid[x_subset], c)
f_r01 = interp.interp1d(x_grid[x_subset], r01)
f_r10 = interp.interp1d(x_grid[x_subset], r10)
f_q = interp.interp1d(x_grid[x_subset], q)

plt.subplot(221)
plt.plot(x_grid[x_subset]/1000, c, "o")
plt.plot(x_grid/1000, f_c(x_grid))
plt.title("$c$ parameter")

plt.subplot(222)
plt.plot(x_grid[x_subset]/1000, q, "o")
plt.plot(x_grid/1000, f_q(x_grid))
plt.title("$q$ parameter")

plt.subplot(223)
plt.plot(x_grid[x_subset]/1000, r01, "o")
plt.plot(x_grid/1000, f_r01(x_grid))
plt.title("$r_{01}$ parameter")

plt.subplot(224)
plt.plot(x_grid[x_subset]/1000, r10, "o")
plt.plot(x_grid/1000, f_r10(x_grid))
plt.title("$r_{10}$ parameter")
plt.show()
