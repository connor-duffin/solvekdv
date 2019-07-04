import logging

import numpy as np
import scipy.interpolate as interpolate
import scipy.linalg as la


class VVerticalMode(object):
    def __init__(self, dx, start_x, end_x, dz0, start_z0, end_z0, n_interp,
                 rho_0):
        self.dz0 = dz0
        self.z0_grid = np.arange(start_z0, end_z0, dz0)
        self.n_z0 = len(self.z0_grid)
        self.dx = dx
        self.x_grid = np.arange(start_x, end_x, dx)
        self.n_x = len(self.x_grid)
        self.x_subs_index = np.round(
            np.linspace(0, self.n_x - 1, n_interp)
        ).astype(int)
        self.rho_0 = rho_0
        self.bathymetry = None
        self.density = None
        self.density_grad = None

        self.phi0 = np.zeros((self.n_x, self.n_z0))
        self.phi0_grad = np.zeros((self.n_x, self.n_z0))
        self.c = np.zeros((self.n_x, 1))

        self.alpha = np.zeros((self.n_x, 1))
        self.beta = np.zeros((self.n_x, 1))
        self.q = np.zeros((self.n_x, 1))
        self.q_grad = np.zeros((self.n_x, 1))

    def compute_bathymetry(self, ocean_floor):
        self.bathymetry = self.z0_grid[-1] - ocean_floor

    def compute_density(self, density="sech"):
        z0_grid = self.z0_grid
        if (
            isinstance(density, np.ndarray)
            and (
                density.shape == (self.n_z0, )
                or density.shape == (self.n_z0, 1)
            )
        ):
            self.density = density
        elif density == "lamb-yan-1":
            self.density = (
                1027.31 - 3.3955 * np.exp((z0_grid - 300) / 50)
            )
        elif density == "lamb-yan-2":
            self.density = (
                1027.31 - 10**(-4) * z0_grid / 9.81
                - 1.696 * (1 + np.tanh((z0_grid - 200)/40))
            )
        elif density == "tanh":
            self.density = (
                (np.exp(z0_grid) - np.exp(-z0_grid))
                / (np.exp(z0_grid) + np.exp(-z0_grid))
            )
        else:
            logging.INFO("Please try another density (e.g. 'lamb-yan-1')")
            logging.ERROR("Density not initialized")
        self.density_grad = np.gradient(self.density, self.dz0)

    def compute_parameters(self):
        dz = self.dz0
        rho_0 = self.rho_0

        x_subs_index = self.x_subs_index
        x_grid = self.x_grid
        z0_grid = self.z0_grid
        bathymetry = self.bathymetry
        density_grad = self.density_grad

        c_temp = np.zeros(len(x_subs_index))
        q_temp = np.zeros(len(x_subs_index))
        beta_temp = np.zeros(len(x_subs_index))
        alpha_temp = np.zeros(len(x_subs_index))

        for i in range(len(x_subs_index)):
            # select the z-grid of interest
            z_grid_temp = z0_grid[z0_grid <= bathymetry[x_subs_index[i]]]
            n_z = len(z_grid_temp)

            # extract the density gradient on the z-grid
            density_grad_temp = density_grad[range(0, len(z_grid_temp))]

            # finite diff matrix on the z-grid
            second_diff_temp = - 1 / (dz**2) * (
                np.diag(np.full(n_z - 1, 1), k=-1)
                + np.diag(np.full(n_z, -2), k=0)
                + np.diag(np.full(n_z - 1, 1), k=1)
            )
            scale = np.diag(
                (- 9.81 / rho_0) * density_grad_temp
            )

            eigenvalue, phi = la.eigh(
                second_diff_temp, b=scale, eigvals=(0, 0)
            )
            phi = np.ndarray.flatten(phi)
            phi /= np.max(phi)
            phi_grad = np.gradient(phi, dz)

            # set the initial element (to be used in computing q)
            if (i == 0):
                self.phi0 = phi
                self.phi0_grad = phi_grad
                phi0_grad = phi_grad

            # compute c, alpha, beta, q
            c_temp[i] = np.sqrt(1 / eigenvalue[0])
            alpha_temp[i] = (
                (3 * c_temp[i] / 2) * (np.trapz(np.power(phi_grad, 3), dx=dz))
                / np.trapz(np.power(phi_grad, 2), dx=dz)
            )
            beta_temp[i] = (
                (c_temp[i] / 2) * np.trapz(np.power(phi, 2), dx=dz)
                / np.trapz(np.power(phi_grad, 2), dx=dz)
            )
            q_temp[i] = np.sqrt(
                c_temp[0]**3 * np.trapz(np.power(phi0_grad, 2), dx=dz)
                / (c_temp[i]**3 * np.trapz(np.power(phi_grad, 2), dx=dz))
            )

        # interpolate the parameters from the grid of values
        f_c = interpolate.interp1d(x_grid[x_subs_index], c_temp)
        f_beta = interpolate.interp1d(x_grid[x_subs_index], beta_temp)
        f_alpha = interpolate.interp1d(x_grid[x_subs_index], alpha_temp)
        f_q = interpolate.interp1d(x_grid[x_subs_index], q_temp, kind="cubic")

        self.c = f_c(x_grid)
        self.beta = f_beta(x_grid)
        self.alpha = f_alpha(x_grid)
        self.q = f_q(x_grid)
        self.q_grad = np.gradient(self.q)
