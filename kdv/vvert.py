import logging

import numpy as np

from scipy.interpolate import interp1d
from scipy.linalg import eigh


class VVerticalMode(object):
    def __init__(self, dx, start_x, end_x, dz0, start_z0, end_z0, n_interp,
                 rho_0):
        self.dz0 = dz0
        self.z_grid = np.arange(start_z0, end_z0, dz0)
        self.n_z0 = len(self.z_grid)
        self.dx = dx
        self.x_grid = np.arange(start_x, end_x, dx)
        self.n_x = len(self.x_grid)
        self.x_subset = list(
            range(0, len(self.x_grid), int(len(self.x_grid) / n_interp))
        ) + [len(self.x_grid) - 1]

        self.rho_0 = rho_0
        self.bathymetry = None
        self.density = None
        self.density_grad = None

        self.phi0 = np.zeros((self.n_x, self.n_z0))
        self.phi0_grad = np.zeros((self.n_x, self.n_z0))
        self.c = np.zeros((self.n_x, 1))

        self.r10 = np.zeros((self.n_x, 1))
        self.r01 = np.zeros((self.n_x, 1))
        self.q = np.zeros((self.n_x, 1))
        self.q_grad = np.zeros((self.n_x, 1))

    def compute_bathymetry(self, ocean_floor):
        self.bathymetry = self.z_grid[-1] - ocean_floor

    def compute_density(self, density="sech"):
        z_grid = self.z_grid
        if (
            isinstance(density, np.ndarray)
            and (
                density.shape == (self.n_z, )
                or density.shape == (self.n_z, 1)
            )
        ):
            self.density = density
        elif density == "lamb-yan-1":
            self.density = (
                1027.31 - 3.3955 * np.exp((z_grid - 300) / 50)
            )
        elif density == "lamb-yan-2":
            self.density = (
                1027.31 - 10**(-4) * z_grid / 9.81
                - 1.696 * (1 + np.tanh((z_grid - 200)/40))
            )
        elif density == "tanh":
            self.density = (
                (np.exp(z_grid) - np.exp(-z_grid))
                / (np.exp(z_grid) + np.exp(-z_grid))
            )
        else:
            logging.INFO("Please try another density (e.g. 'lamb-yan-1')")
            logging.ERROR("Density not initialized")
        self.density_grad = np.gradient(self.density, self.dz0)

    def compute_parameters(self):
        dz = self.dz0
        rho_0 = self.rho_0

        x_subset = self.x_subset
        x_grid = self.x_grid
        z_grid = self.z_grid
        bathymetry = self.bathymetry
        density = self.density

        c_temp = np.zeros(len(x_subset))
        q_temp = np.zeros(len(x_subset))
        r01_temp = np.zeros(len(x_subset))
        r10_temp = np.zeros(len(x_subset))

        for i in range(len(x_subset)):
            # select the z-grid of interest
            z_grid_temp = z_grid[z_grid <= bathymetry[x_subset[i]]]
            n_z = len(z_grid_temp)
            # make sure we are getting enough points in our domain
            print(n_z)

            # compute the density on the z-grid
            density_temp = density[range(0, len(z_grid_temp))]
            density_grad_temp = np.gradient(density_temp, dz)

            # finite diff matrix on the z-grid
            second_diff_temp = - 1 / (dz**2) * (
                np.diag(np.full(n_z - 1, 1), k=-1)
                + np.diag(np.full(n_z, -2), k=0)
                + np.diag(np.full(n_z - 1, 1), k=1)
            )
            scale = np.diag(
                np.full(n_z, (- 9.81 / rho_0) * density_grad_temp)
            )

            eigenvalue, phi = eigh(
                second_diff_temp, b=scale, eigvals=(0, 0)
            )
            phi = np.ndarray.flatten(phi)
            phi /= np.max(phi)
            phi_grad = np.gradient(phi, dz)

            # set the initial element (to be used in computing Q)
            if (i == 0):
                self.phi0 = phi
                self.phi0_grad = phi_grad
                phi0_grad = phi_grad

            # compute c, r10, r01, Q
            c_temp[i] = np.sqrt(1 / eigenvalue[0])
            r10_temp[i] = (
                (3 * c_temp[i] / 2) * (np.trapz(np.power(phi_grad, 3), dx=dz)
                / np.trapz(np.power(phi_grad, 2), dx=dz))
            )
            r01_temp[i] = (
                (c_temp[i] / 2) * (np.trapz(np.power(phi, 2), dx=dz)
                / np.trapz(np.power(phi_grad, 2), dx=dz))
            )
            q_temp[i] = np.sqrt(
                c_temp[0]**3 * np.trapz(np.power(phi0_grad, 2), dx=dz)
                / (c_temp[i]**3 * np.trapz(np.power(phi_grad, 2), dx=dz))
            )

        # interpolate out the other parameters, from the grid
        f_c = interp1d(x_grid[x_subset], c_temp)
        f_r01 = interp1d(x_grid[x_subset], r01_temp)
        f_r10 = interp1d(x_grid[x_subset], r10_temp)
        f_q = interp1d(x_grid[x_subset], q_temp)

        self.c = f_c(x_grid)
        self.r01 = f_r01(x_grid)
        self.r10 = f_r10(x_grid)
        self.q = f_q(x_grid)
        self.q_grad = np.gradient(self.q)
