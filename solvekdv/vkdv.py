import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as spla


# shape of u is incorrect
class Kdv(object):
    def __init__(self, dt, dx, start_x, end_x, start_t, end_t):
        self.dt = dt
        self.dx = dx
        self.x_grid = np.arange(start_x, end_x, dx)
        self.t_grid = np.arange(start_t, end_t, dt)
        self.n_x = len(self.x_grid)
        self.n_t = len(self.t_grid)
        self.u0 = np.zeros((self.n_x, 1))
        self.u1 = np.zeros((self.n_x, 1))
        self.u2 = np.zeros((self.n_x, 1))

        self.a = 0
        self.b = 0
        self.c = 0
        self.q = np.zeros(self.n_x)
        self.q_grad = np.zeros(self.n_x)

        self.first_order_matrix = np.zeros([self.n_x, self.n_x])
        self.third_order_matrix = np.zeros([self.n_x, self.n_x])
        self.lhs_matrix = np.zeros([self.n_x, self.n_x])

        self.a_first_order_matrix = None
        self.b_third_order_matrix = None
        self.c_first_order_matrix = None
        self.bathymetry_term = None

    def set_initial_condition(self, initial):
        self.u0[:] = initial
        self.u1[:] = initial
        self.u2[:] = initial

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
                sparse.diags(self.c, format="csr") @ self.first_order_matrix
            )
            + (3 * dt / 4) * (
                sparse.diags(self.b, format="csr") @ self.third_order_matrix
            )
        )
        self.lhs_matrix = output

    def set_solve_matrices(self):
        a_sp = sparse.diags(self.a, format="csr")
        b_sp = sparse.diags(self.b, format="csr")
        c_sp = sparse.diags(self.c, format="csr")
        self.a_first_order_matrix = a_sp @ self.first_order_matrix
        self.b_third_order_matrix = b_sp @ self.third_order_matrix
        self.c_first_order_matrix = c_sp @ self.first_order_matrix
        self.bathymetry_term = (self.c / self.q) * self.q_grad

    # is a diag matrix for each parameter really necessary???
    # would work the exact same if there was just an element-wise mult.
    # tomorrow: check dimensions on these so that they line up nicely.
    def solve_step(self):
        dt = self.dt
        rhs_vector = (
            self.u0
            - (7 * dt / 4) * self.u0 * (self.a_first_order_matrix @ self.u0)
            + dt * self.u1 * (self.a_first_order_matrix @ self.u1)
            - (dt / 4) * self.u2 * (self.a_first_order_matrix @ self.u2)
            + (dt / 4) * (self.c_first_order_matrix) @ self.u1
            + (dt / 4) * (self.b_third_order_matrix) @ self.u1
            - (dt / 4) * self.bathymetry_term.reshape(self.n_x, 1) * self.u1
        )
        print(rhs_vector.shape)
        output = spla.spsolve(self.lhs_matrix, rhs_vector).reshape(self.n_x)
        self.u2 = self.u1
        self.u1 = self.u0
        self.u0 = output
        return output
