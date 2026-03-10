import numpy as np

def pca_projection(X, k):
    """
    Project data onto the top-k principal components.
    """
    X = np.asarray(X, dtype=float)
    n, d = X.shape

    # Center data
    mean = np.mean(X, axis=0)
    X_centered = X - mean

    # Sample covariance
    C = (X_centered.T @ X_centered) / (n - 1)

    # Eigen-decomposition of symmetric covariance matrix
    eigvals, eigvecs = np.linalg.eigh(C)

    # Sort by descending eigenvalue
    idx = np.argsort(eigvals)[::-1]
    W = eigvecs[:, idx[:k]]

    # Project onto top-k components
    X_proj = X_centered @ W

    return X_proj.tolist()