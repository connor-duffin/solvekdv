import logging

import numpy as np
import scipy.linalg as la


class VVerticalMode(object):
    def __init__(self, dx, start_x, end_x, dz0, start_z0, end_z0,
                 n_eigen, rho_0):
        self.dz0 = dz0
        self.z0_grid = np.arange(start_z0, end_z0, dz0)
        self.n_z0 = len(self.z0_grid)
        self.dx = dx
        self.x_grid = np.arange(start_x, end_x, dx)
        self.n_x = len(self.x_grid)
        self.n_eigen = n_eigen

        self.rho_0 = rho_0
        self.bathymetry = None
        self.density = None
        self.density_grad = None
        self.density_func = None

        self.phi0 = np.zeros((self.n_x, self.n_z0))
        self.phi0_grad = np.zeros((self.n_x, self.n_z0))
        self.c = np.zeros((self.n_x, 1))

        self.alpha = np.zeros((self.n_x, 1))
        self.beta = np.zeros((self.n_x, 1))
        self.q = np.zeros((self.n_x, 1))
        self.q_grad = np.zeros((self.n_x, 1))

    def compute_bathymetry(self, ocean_floor):
        self.bathymetry = self.z0_grid[-1] - ocean_floor

    def compute_density(self, density="tanh"):
        z0_grid = self.z0_grid
        if density == "lamb-yan-1":
            self.density = (
                1027.31 - 3.3955 * np.exp((z0_grid - 300) / 50)
            )
            self.density_func = lambda z: (
                1027.31 - 3.3955 * np.exp((z - 300) / 50)
            )
        elif density == "lamb-yan-2":
            self.density = (
                1027.31 - 10**(-4) * z0_grid / 9.81
                - 1.696 * (1 + np.tanh((z0_grid - 200)/40))
            )
            self.density_func = lambda z: (
                1027.31 - 10**(-4) * z / 9.81
                - 1.696 * (1 + np.tanh((z - 200)/40))
            )
        elif density == "tanh":
            self.density = (
                (np.exp(z0_grid) - np.exp(-z0_grid))
                / (np.exp(z0_grid) + np.exp(-z0_grid))
            )
            self.density_func = lambda z: (
                (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))
            )
        else:
            logging.ERROR("Density not initialized.")
        self.density_grad = np.gradient(self.density, self.dz0)

    def compute_parameters(self):
        dz = self.dz0
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

            # compute the density on the z-grid
            density_temp = self.density_func(z_grid_temp)
            density_grad_temp = np.gradient(density_temp)

            # lhs of eigenvalue problem
            second_diff_temp = - 1 / (dz**2) * (
                np.diag(np.full(n_eigen - 1, 1), k=-1)
                + np.diag(np.full(n_eigen, -2), k=0)
                + np.diag(np.full(n_eigen - 1, 1), k=1)
            )
            # rhs of eigenvalue problem
            scale = np.diag(
                (- 9.81 / rho_0) * density_grad_temp
            )

            # solve eigenvalue problem using dense matrices, get first
            eigenvalue, phi = la.eigh(
                second_diff_temp, b=scale, eigvals=(0, 0)
            )
            phi = np.ndarray.flatten(phi)
            phi /= np.max(phi)

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
                (3 * c_temp[i] / 2) * (np.trapz(np.power(phi_grad, 3), dx=dz))
                / np.trapz(np.power(phi_grad, 2), dx=dz)
            )
            beta_temp[i] = (
                (c_temp[i] / 2) * np.trapz(np.power(phi, 2), dx=dz)
                / np.trapz(np.power(phi_grad, 2), dx=dz)
            )
            q_temp[i] = (
                c_temp[0]**3 * np.trapz(np.power(phi0_grad, 2), dx=dz)
                / (c_temp[i]**3 * np.trapz(np.power(phi_grad, 2), dx=dz))
            )

        self.c = c_temp
        self.beta = beta_temp
        self.alpha = alpha_temp
        self.q = q_temp
        self.q_grad = np.gradient(q_temp)
