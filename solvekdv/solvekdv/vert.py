import logging

import numpy as np
import scipy.sparse as sparse
import scipy.linalg as la
import scipy.sparse.linalg as spla


class VerticalMode(object):
    def __init__(self, z_start, z_end, n_z, rho_0):
        self.z_grid = np.linspace(z_start, z_end, n_z)
        self.dz = self.z_grid[1] - self.z_grid[0]
        self.n_z = n_z
        self.rho_0 = rho_0
        self.density = None
        self.density_grad = None
        self.phi = None
        self.phi_grad = None

        self.alpha = None
        self.beta = None
        self.c = None

    def initialize_dht_density(self, params=None):
        if (
            isinstance(self.density, np.ndarray)
            and (
                self.density.shape == (self.n_z, )
                or self.density.shape == (self.n_z, 1)
            )
        ):
            logging.warning("Using pre-defined density.")
        else:
            if params is None:
                logging.warning("Using default parameters (post. mean).")
                params = np.array(
                    [1023.66, -1.15, 72.58, 49.12, 153.44, 49.98]
                )
            z_grid = self.z_grid
            self.density = (
                params[0] + params[1] * (
                    np.tanh((z_grid + params[2]) / params[3])
                    + np.tanh((z_grid + params[4]) / params[5])
                )
            )
            self.density_grad = (
                params[1] * (
                    (1 / np.cosh((z_grid + params[2]) / params[3]))**2 / params[3]
                    + (1 / np.cosh((z_grid + params[4]) / params[5]))**2 / params[5]
                )
            )

    def initialize_lamb_density(self):
        if (
            isinstance(self.density, np.ndarray)
            and (
                self.density.shape == (self.n_z, )
                or self.density.shape == (self.n_z, 1)
            )
        ):
            logging.warning("Using pre-defined density.")
        else:
            z_grid = self.z_grid
            self.density = (
                1027.31 - 3.3955 * np.exp(z_grid / 50)
            )
            self.density_grad = (
                -3.3955 / 50 * np.exp(z_grid / 50)
            )

    def find_vertical_mode(self):
        dz = self.dz
        n_z = self.n_z
        rho_0 = self.rho_0
        density_grad = self.density_grad

        second_diff = - 1 / (dz**2) * (
            np.diag(np.full(n_z - 1, 1), k=-1)
            + np.diag(np.full(n_z, -2), k=0)
            + np.diag(np.full(n_z - 1, 1), k=1)
        )
        scale = np.diag(np.full(n_z, (-9.81 / rho_0) * density_grad))

        eigenvalue, phi = la.eigh(
            second_diff, b=scale, eigvals=(0, 0)
        )
        self.phi = np.ndarray.flatten(phi)
        phi_max = self.phi[np.argmax(np.abs(phi))]
        self.phi /= phi_max
        self.phi_grad = np.gradient(self.phi, self.dz)
        self.c = np.sqrt(1 / eigenvalue[0])

    def find_vertical_mode_sparse(self):
        dz = self.dz
        n_z = self.n_z
        rho_0 = self.rho_0
        density_grad = self.density_grad

        second_diff = - 1 / (dz**2) * sparse.diags(
            diagonals=[
                np.full(n_z - 1, 1),
                np.full(n_z, -2),
                np.full(n_z - 1, 1)
            ],
            offsets=[-1, 0, 1],
            format="csc"
        )
        scale = sparse.diags(
            np.full(n_z, (9.81 / rho_0) * density_grad), 0, format="csc"
        )

        eigenvalue, phi = spla.eigs(second_diff, b=scale, k=1, which="LM")
        self.phi = np.ndarray.flatten(phi)
        self.phi /= np.max(self.phi)
        self.phi_grad = np.gradient(self.phi, self.dz)
        self.c = np.sqrt(1 / eigenvalue[0])

    def compute_alpha(self):
        phi_grad = self.phi_grad
        self.alpha = (
            (3 * self.c / 2)
            * np.trapz(np.power(phi_grad, 3), dx=self.dz)
            / np.trapz(np.power(phi_grad, 2), dx=self.dz)
        )

    def compute_beta(self):
        phi = self.phi
        phi_grad = self.phi_grad
        self.beta = (
            (self.c / 2)
            * np.trapz(np.power(phi, 2), dx=self.dz)
            / np.trapz(np.power(phi_grad, 2), dx=self.dz)
        )
