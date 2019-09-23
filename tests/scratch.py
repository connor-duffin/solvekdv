dt = test.dt
diag = np.array(
    1 - (3 * dt / 4) * test.bathymetry_term
).flatten()
output = (
    sparse.diags(diag, format="csr")
    + (3 * dt / 4) * (
        test.first_order_matrix.multiply(test.c)
    )
    + (3 * dt / 4) * (
        test.third_order_matrix.multiply(test.b)
    )
)
test.lhs_matrix = output

