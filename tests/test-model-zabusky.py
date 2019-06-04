import numpy as np
import matplotlib.pyplot as plt

from context import kdv


test = kdv.Kdv(
    dt=0.001, dx=0.005, start_x=0, end_x=2, start_t=0, end_t=10
)

test.set_initial_condition(np.cos(np.pi * test.x_grid))
test.set_kdv_parameters(a=1, b=0.022**2, c=0)
test.set_first_order_matrix()
test.set_third_order_matrix()
test.set_lhs_matrix()

u = np.zeros([test.n_x, test.n_t])
for i in range(test.n_t):
    if (i % int(0.1 * test.n_t)) == 0:
        print(f"Simulation {100 * i / test.n_t:.1f} % complete.")
    u[:, i] = test.solve_step()

xmesh, ymesh = np.meshgrid(test.x_grid, test.t_grid)

plt.figure()
plt.pcolormesh(xmesh, ymesh, u.transpose(), cmap='viridis')
plt.colorbar()
plt.xlabel('x')
plt.ylabel('t')
plt.title(
    'KdV, ($a$, $b$, $c$)=(%g, %g, %g)' % (
        test.a, test.b, test.c
    )
)
plt.show()
