import numpy as np

def batch_generator(X, y, batch_size, rng=None, drop_last=False):
    """
    Randomly shuffle a dataset and yield mini-batches (X_batch, y_batch).
    """
    X = np.asarray(X)
    y = np.asarray(y)

    n = len(X)
    indices = np.arange(n)

    if rng is not None:
        rng.shuffle(indices)
    else:
        np.random.shuffle(indices)

    X_shuffled = X[indices]
    y_shuffled = y[indices]

    for start in range(0, n, batch_size):
        end = start + batch_size
        if end > n and drop_last:
            break
        yield X_shuffled[start:end], y_shuffled[start:end]