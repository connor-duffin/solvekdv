import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as spla


class Kdv(object):
    def __init__(self, dt, dx, start_x, end_x, start_t, end_t):
        self.dt = dt
        self.dx = dx
        self.x_grid = np.arange(start_x, end_x + dx, dx)
        self.t_grid = np.arange(start_t, end_t + dt, dt)
        self.n_x = len(self.x_grid)
        self.n_t = len(self.t_grid)
        self.u0 = np.zeros((self.n_x, 1))
        self.u1 = np.zeros((self.n_x, 1))
        self.u2 = np.zeros((self.n_x, 1))

        self.a = None
        self.b = None
        self.c = None

        self.q = None
        self.q_grad = None

        self.first_order_matrix = np.zeros((self.n_x, self.n_x))
        self.third_order_matrix = np.zeros((self.n_x, self.n_x))
        self.lhs_matrix = np.zeros((self.n_x, self.n_x))

        self.a_first_order_matrix = None
        self.b_third_order_matrix = None
        self.c_first_order_matrix = None
        self.bathymetry_term = None

    def set_initial_condition(self, initial):
        self.u0 = initial.copy()
        self.u1 = initial.copy()
        self.u2 = initial.copy()

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
        dt = self.dt
        diag = np.array(
            1 - (3 * dt / 4) * self.q_grad * (self.c / self.q)
        ).flatten()
        output = (
            sparse.diags(diag, format="csr")
            + (3 * dt / 4) * (
                self.first_order_matrix.multiply(self.c)
            )
            + (3 * dt / 4) * (
                self.third_order_matrix.multiply(self.b)
            )
        )
        self.lhs_matrix = output

    def solve_step(self):
        dt = self.dt
        rhs_vector = (
            self.u0
            - (7 * dt / 4) * self.a * self.u0 * (self.first_order_matrix @ self.u0)
            + dt * self.a * self.u1 * (self.first_order_matrix @ self.u1)
            - (dt / 4) * self.a * self.u2 * (self.first_order_matrix @ self.u2)
            + (dt / 4) * self.c * (self.first_order_matrix @ self.u1)
            + (dt / 4) * self.b * (self.third_order_matrix @ self.u1)
            - (dt / 4) * self.bathymetry_term * self.u1
        )
        output = spla.spsolve(self.lhs_matrix, rhs_vector)
        self.u2 = self.u1.copy()
        self.u1 = self.u0.copy()
        self.u0 = output.reshape(self.n_x, 1).copy()
        return(output)
