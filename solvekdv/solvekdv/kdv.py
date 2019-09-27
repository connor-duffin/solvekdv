import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as spla


class Kdv(object):
    def __init__(self, dt, dx, x_start, x_end, t_start, t_end):
        self.dt = dt
        self.dx = dx
        self.x_grid = np.arange(x_start, x_end, dx)
        self.t_grid = np.arange(t_start, t_end, dt)
        self.n_x = len(self.x_grid)
        self.n_t = len(self.t_grid)

        self.u0 = np.zeros((self.n_x, 1))
        self.u1 = np.zeros((self.n_x, 1))
        self.u2 = np.zeros((self.n_x, 1))

        self.alpha = None
        self.beta = None
        self.c = None

        self.first_order_matrix = np.zeros((self.n_x, self.n_x))
        self.third_order_matrix = np.zeros((self.n_x, self.n_x))
        self.lhs_matrix = np.zeros((self.n_x, self.n_x))

    def set_initial_condition(self, initial):
        self.u0 = initial.copy()
        self.u1 = initial.copy()
        self.u2 = initial.copy()

    def set_first_order_matrix(self):
        dx, n_x = self.dx, self.n_x
        output = (1 / (2*dx)) * sparse.diags(
            diagonals=[
                1,
                np.full(n_x - 1, -1),
                np.full(n_x - 1, 1),
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

    def set_imex_lhs_matrix(self):
        output = (
            sparse.identity(self.n_x, format="csr")
            + self.beta * (3 * self.dt / 4) * self.third_order_matrix
            + self.c * (3 * self.dt / 4) * self.first_order_matrix
        )
        self.lhs_matrix = output

    def solve_step_imex(self):
        alpha, beta, c, dt = self.alpha, self.beta, self.c, self.dt
        rhs_vector = (
            self.u0
            - alpha * (7 * dt / 4) * self.u0 * (self.first_order_matrix @ self.u0)
            + alpha * dt * self.u1 * (self.first_order_matrix @ self.u1)
            - alpha * (dt / 4) * self.u2 * (self.first_order_matrix @ self.u2)
            - beta * (dt / 4) * self.third_order_matrix @ self.u1
            - c * (dt / 4) * self.first_order_matrix @ self.u1
        )
        output = spla.spsolve(self.lhs_matrix, rhs_vector)
        self.u2 = self.u1.copy()
        self.u1 = self.u0.copy()
        self.u0 = output.reshape(self.n_x, 1).copy()
        return(output)

    def _im_euler_lhs(self, u, u_prev):
        dt = self.dt
        alpha = self.alpha
        beta = self.beta
        c = self.c
        D_first = self.first_order_matrix
        D_third = self.third_order_matrix
        return(
            u - u_prev
            + dt * alpha * np.multiply(u, (D_first @ u).T).T
            + dt * beta * D_third @ u
            + dt * c * D_first @ u
        )

    def _im_euler_jacobian_lhs(self, u):
        D_first = self.first_order_matrix
        D_third = self.third_order_matrix
        dt = self.dt
        dx = self.dx
        n_x = self.n_x
        alpha = self.alpha
        beta = self.beta
        c = self.c
        return(
            sparse.eye(n_x)
            + 1e-8 * sparse.eye(n_x)
            + alpha * (dt / (2 * dx)) * sparse.diags(
                diagonals=[
                    u[-1],
                    -u[1:],
                    np.roll(u, -1) - np.roll(u, 1),
                    u[:-1],
                    -u[0]
                ],
                offsets=[-(n_x - 1), -1, 0, 1, n_x - 1],
                format="csc"
            )
            + beta * dt * D_third
            + c * dt * D_first
        )

    def _newton_im_euler(self, tol=1e-10, max_iter=10_000):
        u_prev = self.u0.copy()
        u = self.u0.copy()
        epsilon = 1e6
        i = 0
        # print(u_prev)
        while (epsilon >= tol and i < max_iter):
            jacobian = self._im_euler_jacobian_lhs(u.flatten())
            f = self._im_euler_lhs(u, u_prev)
            s = spla.spsolve(jacobian, -f)
            u += s
            i += 1
            epsilon = np.sqrt(np.sum(s ** 2))

        return(u)

    def solve_step_im_euler(self):
        u = self._newton_im_euler()
        self.u2 = self.u1.copy()
        self.u1 = self.u0.copy()
        self.u0 = u.copy()
        return(u)
