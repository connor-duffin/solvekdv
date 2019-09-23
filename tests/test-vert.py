import matplotlib.pyplot as plt

from context import vert


vertical_test = vert.VerticalMode(0.1, 0, 200, 1000)
vertical_test.compute_density("lamb-yan-1")

vertical_test.find_vertical_mode()
vertical_test.compute_alpha()
vertical_test.compute_beta()

print(
    f"r10: {vertical_test.alpha:.4f}\n"
    + f"r01: {vertical_test.beta:.4f}\n"
    + f"c:   {vertical_test.c:.4f}\n"
)

plt.subplot(2, 1, 1)
plt.plot(vertical_test.z_grid, vertical_test.phi)
plt.subplot(2, 1, 2)
plt.plot(vertical_test.z_grid, vertical_test.phi_grad)
plt.show()
plt.plot(vertical_test.density, vertical_test.z_grid)
plt.show()
