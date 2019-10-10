import numpy as np

from solvekdv import kdv


def solve_kdv(dt, dx):
    test = kdv.Kdv(
        dt=dt, dx=dt, x_start=0, x_end=2, t_start=0, t_end=2
    )
    test.set_initial_condition(
        np.array(np.cos(np.pi * test.x_grid), ndmin=2).T
    )
    test.alpha = 1
    test.beta = 0.022**2
    test.c = 0
    test.set_first_order_matrix()
    test.set_third_order_matrix()
    test.set_imex_lhs_matrix()
    indices = np.linspace(0, test.n_x, 200, dtype=np.int32, endpoint=False)
    xgrid, tgrid = np.meshgrid(indices, indices)

    u = np.zeros((test.n_x, test.n_t))
    for i in range(test.n_t):
        print(f"\rIteration {i + 1:5} / {test.n_t}", end="")
        u[:, i] = test.solve_step_imex()

    print()
    return(u[xgrid, tgrid])


grids = [0.0005, 0.000625, 0.00125, 0.0025, 0.005, 0.0075]
u_all = np.zeros((6, 200, 200))
diff = np.zeros(5)

for i in range(6):
    u_all[i, :, :] = solve_kdv(dt=grids[i], dx=grids[i])

for i in range(5):
    diff[i] = np.sqrt(np.sum(np.abs(u_all[i + 1, :, :] - u_all[i, :, :])**2))
