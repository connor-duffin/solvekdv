import numpy as np
import scipy.sparse as sparse

from scipy.linalg import eigh
from scipy.sparse.linalg import eigs


class VerticalMode(object):

    def __init__(self, dz, start_z, end_z, rho_0):
        self.dz = dz
        self.z_grid = np.arange(start_z, end_z, dz)
        self.n_z = len(self.z_grid)
        self.rho_0 = rho_0

        self.density = None
        self.grad_density = None
        self.phi = None
        self.phi_grad = None
        self.c = None

        self.r10 = None
        self.r01 = None

    def compute_density(self, density="sech"):
        z_grid = self.z_grid
        if density == "lamb-yan-1":
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
                (np.exp(z_grid) - np.exp(-z_grid)) / (np.exp(z_grid) + np.exp(-z_grid))
            )
        else:
            print("Please try another density form (e.g. 'lamb-yan-1'")
        self.grad_density = np.gradient(self.density, self.dz)

    def find_vertical_mode(self):
        dz = self.dz
        n_z = self.n_z
        rho_0 = self.rho_0
        grad_density = self.grad_density

        second_diff = - 1 / (dz**2) * (
            np.diag(np.full(n_z - 1, 1), k=-1)
            + np.diag(np.full(n_z, -2), k=0)
            + np.diag(np.full(n_z - 1, 1), k=1)
        )
        scale = np.diag(np.full(n_z, (- 9.81 / rho_0) * grad_density))

        eigenvalue, phi = eigh(
            second_diff, b=scale, eigvals=(0, 0)
        )
        self.phi = np.ndarray.flatten(phi)
        self.phi /= np.max(self.phi)
        self.phi_grad = np.gradient(self.phi, self.dz)
        self.c = np.sqrt(1 / eigenvalue[0])

    def find_vertical_mode_sparse(self):
        dz = self.dz
        n_z = self.n_z
        rho_0 = self.rho_0
        grad_density = self.grad_density

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
            np.full(n_z, (9.81 / rho_0) * grad_density), 0, format="csc"
        )

        eigenvalue, phi = eigs(second_diff, b=scale, k=1, which="LM")
        self.phi = np.ndarray.flatten(phi)
        self.phi /= np.max(self.phi)
        self.phi_grad = np.gradient(self.phi, self.dz)
        self.c = np.sqrt(1 / eigenvalue[0])

    def compute_r10(self):
        phi = self.phi
        phi_grad = self.phi_grad
        self.r10 = (
            (3 / 2) * (np.trapz(np.power(phi, 3), dx=self.dz)
            / np.trapz(np.power(phi_grad, 2), dx=self.dz))
        )

    def compute_r01(self):
        phi = self.phi
        phi_grad = self.phi_grad
        self.r01 = (
            (self.c / 2) * (np.trapz(np.power(phi, 2), dx=self.dz)
            / np.trapz(np.power(phi_grad, 2), dx=self.dz))
        )
