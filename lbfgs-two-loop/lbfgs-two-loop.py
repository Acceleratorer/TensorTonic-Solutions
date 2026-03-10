def _dot(a, b):
    """Dot product of two vectors."""
    return sum(x * y for x, y in zip(a, b))

def lbfgs_direction(grad, s_list, y_list):
    """
    Compute the L-BFGS search direction using the two-loop recursion.
    """
    m = len(s_list)
    q = grad[:]  # copy

    rho = [0.0] * m
    alpha = [0.0] * m

    for i in range(m):
        rho[i] = 1.0 / _dot(y_list[i], s_list[i])

    # First loop: newest to oldest
    for i in range(m - 1, -1, -1):
        alpha[i] = rho[i] * _dot(s_list[i], q)
        q = [qj - alpha[i] * yj for qj, yj in zip(q, y_list[i])]

    # Initial scaling with most recent pair
    sy = _dot(s_list[-1], y_list[-1])
    yy = _dot(y_list[-1], y_list[-1])
    gamma = sy / yy
    r = [gamma * qj for qj in q]

    # Second loop: oldest to newest
    for i in range(m):
        beta = rho[i] * _dot(y_list[i], r)
        r = [rj + s_list[i][j] * (alpha[i] - beta) for j, rj in enumerate(r)]

    # Return descent direction
    return [-rj for rj in r]