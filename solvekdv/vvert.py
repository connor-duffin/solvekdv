import logging

import numpy as np
import scipy.linalg as la


class VVerticalMode(object):
    def __init__(self, dx, x_start, x_end, dz0, z0_start, z0_end,
                 n_eigen, rho_0):
        self.dz0 = dz0
        self.z0_grid = np.arange(z0_start, z0_end, dz0)
        self.n_z0 = len(self.z0_grid)
        self.dx = dx
        self.x_grid = np.arange(x_start, x_end, dx)
        self.n_x = len(self.x_grid)
        self.n_eigen = n_eigen

        self.rho_0 = rho_0
        self.bathymetry = None
        self.density = None
        self.density_grad = None
        self.density_func = None

        self.phi0 = np.zeros(self.n_z0)
        self.phi0_grad = np.zeros(self.n_z0)
        self.c = np.zeros(self.n_x)

        self.alpha = np.zeros(self.n_x)
        self.beta = np.zeros(self.n_x)
        self.q = np.zeros(self.n_x)
        self.q_grad = np.zeros(self.n_x)

    def initialize_dht_density(self, params=None):
        z0_grid = self.z0_grid

        if params is None:
            params = np.array(
                [1023.66, -1.15, 72.58, 49.12, 153.44, 49.98]
            )
            logging.warning("Density profile initialized with defaults.")

        self.density = (
            params[0] + params[1] * (
                np.tanh((z0_grid + params[2]) / params[3])
                + np.tanh((z0_grid + params[4]) / params[5])
            )
        )
        self.density_func = lambda z: (
            params[0] + params[1] * (
                np.tanh((z + params[2]) / params[3])
                + np.tanh((z + params[4]) / params[5])
            )
        )
        self.density_grad_func = lambda z: (
            params[1] * (
                (1 / np.cosh((z + params[2]) / params[3]))**2 / params[3]
                + (1 / np.cosh((z + params[4]) / params[5]))**2 / params[5]
            )
        )

    def initialize_lamb_density(self):
        z0_grid = self.z0_grid
        self.density = (
            1027.31 - 3.3955 * np.exp((z0_grid - 300) / 50)
        )
        self.density_func = lambda z: (
            1027.31 - 3.3955 * np.exp((z - 300) / 50)
        )

    def compute_parameters(self):
        rho_0 = self.rho_0
        n_eigen = self.n_eigen

        bathymetry = self.bathymetry

        c_temp = np.zeros(self.n_x)
        q_temp = np.zeros(self.n_x)
        beta_temp = np.zeros(self.n_x)
        alpha_temp = np.zeros(self.n_x)

        for i in range(self.n_x):
            # set the z-grid of interest
            z_grid_temp = np.linspace(0, bathymetry[i], n_eigen)
            dz = z_grid_temp[1] - z_grid_temp[0]

            # compute the density on the z-grid
            density_grad_temp = self.density_grad_func(z_grid_temp)

            # lhs of eigenvalue problem
            second_diff_temp = -1 / (dz**2) * (
                np.diag(np.full(n_eigen - 1, 1), k=-1)
                + np.diag(np.full(n_eigen, -2), k=0)
                + np.diag(np.full(n_eigen - 1, 1), k=1)
            )
            # rhs of eigenvalue problem
            scale = np.diag(
                -(9.81 / rho_0) * density_grad_temp
            )

            # solve eigenvalue problem using dense matrices, get first
            eigenvalue, phi = la.eigh(
                second_diff_temp, b=scale, eigvals=(0, 0)
            )
            phi = np.ndarray.flatten(phi)
            phi_max = phi[np.argmax(np.abs(phi))]  # get largest absolute value
            phi /= phi_max

            # change this part
            phi_grad = np.gradient(phi, z_grid_temp[1] - z_grid_temp[0])

            # set the initial element (to be used in computing q)
            if (i == 0):
                self.phi0 = phi
                self.phi0_grad = phi_grad
                phi0_grad = phi_grad

            # compute c, alpha, beta, q
            c_temp[i] = np.sqrt(1 / eigenvalue[0])
            alpha_temp[i] = (
                (3 * c_temp[i] / 2)
                * (np.trapz(np.power(phi_grad, 3), dx=dz))
                / np.trapz(np.power(phi_grad, 2), dx=dz)
            )
            beta_temp[i] = (
                (c_temp[i] / 2)
                * np.trapz(np.power(phi, 2), dx=dz)
                / np.trapz(np.power(phi_grad, 2), dx=dz)
            )
            q_temp[i] = (
                (c_temp[i]**3 / c_temp[0]**3)
                * np.trapz(np.power(phi_grad, 2), dx=dz)
                / np.trapz(np.power(phi0_grad, 2), dx=dz)
            )

        self.c = c_temp
        self.beta = beta_temp
        self.alpha = alpha_temp
        self.q = q_temp
        self.q_grad = np.gradient(q_temp)
