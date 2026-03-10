import numpy as np

def matrix_transpose(A):
    """
    Return the transpose of matrix A (swap rows and columns).
    """
    A = np.asarray(A)
    n, m = A.shape
    result = np.empty((m, n), dtype=A.dtype)

    for i in range(n):
        for j in range(m):
            result[j, i] = A[i, j]

    return result