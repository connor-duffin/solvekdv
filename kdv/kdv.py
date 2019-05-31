import numpy as np
import scipy.sparse as sparse

from scipy.sparse.linalg import spsolve


class Kdv(object):

    def __init__(self, dt, dx, start_x, end_x, start_t, end_t):
        self.dt = dt
        self.dx = dx
        self.x_grid = np.arange(start_x, end_x, dx)
        self.t_grid = np.arange(start_t, end_t, dt)
        self.n_x = len(self.x_grid)
        self.n_t = len(self.t_grid)
        self.u0 = np.zeros(self.n_x)
        self.u1 = np.zeros(self.n_x)
        self.u2 = np.zeros(self.n_x)
        self.a = 0
        self.b = 0

        self.first_order_matrix = np.zeros([self.n_x, self.n_x])
        self.third_order_matrix = np.zeros([self.n_x, self.n_x])
        self.lhs_matrix = np.zeros([self.n_x, self.n_x])

    def set_initial_condition(self, initial):
        self.u0[:] = initial
        self.u1[:] = initial
        self.u2[:] = initial

    def set_kdv_parameters(self, a=1, b=0.022**2):
        self.a, self.b = a, b

    def set_first_order_matrix(self):
        dx, n_x = self.dx, self.n_x
        output = (1 / (2*dx)) * sparse.diags(
            diagonals=[
                1,
                np.full(self.n_x - 1, -1),
                np.full(self.n_x - 1, 1),
                -1
            ],
            offsets=[-(n_x - 1), -1, 1, (n_x - 1)],
            format="csr"
        )
        self.first_order_matrix = output

    def set_third_order_matrix(self):
        dx, n_x = self.dx, self.n_x
        output = (1 / (2*dx**3)) * sparse.diags(
            diagonals=[
                -2,
                [1, 1],
                np.full(n_x - 2, -1),
                np.full(n_x - 1, 2),
                np.full(n_x - 1, - 2),
                np.full(n_x - 2, 1),
                [-1, -1],
                2
            ],
            offsets=[-(n_x - 1), -(n_x - 2), -2, -1, 1, 2, n_x - 2, n_x - 1],
            format="csr"
        )
        self.third_order_matrix = output

    def set_lhs_matrix(self):
        output = (sparse.identity(self.n_x, format="csr")
                  + self.b * (3 * self.dt / 4) * self.third_order_matrix)
        self.lhs_matrix = output

    def solve_step(self):
        a, b, dt = self.a, self.b, self.dt
        rhs_vector = (
            self.u0
            - a * (7 * dt / 4) * self.u0 * (self.first_order_matrix @ self.u0)
            + a * dt * self.u1 * (self.first_order_matrix @ self.u1)
            - a * (dt / 4) * self.u2 * (self.first_order_matrix @ self.u2)
            - b * (dt / 4) * self.third_order_matrix @ self.u1
        )
        output = spsolve(self.lhs_matrix, rhs_vector)
        self.u2[:] = self.u1
        self.u1[:] = self.u0
        self.u0[:] = output
        return output
