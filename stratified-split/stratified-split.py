import numpy as np

def stratified_split(X, y, test_size=0.2, rng=None):
    """
    Return X_train, X_test, y_train, y_test with stratified class proportions.
    """
    X = np.asarray(X)
    y = np.asarray(y)

    train_indices = []
    test_indices = []

    for cls in np.unique(y):
        cls_idx = np.where(y == cls)[0].copy()

        if rng is not None:
            rng.shuffle(cls_idx)
        else:
            np.random.shuffle(cls_idx)

        n_cls = len(cls_idx)
        n_test = int(round(n_cls * test_size))

        if n_cls > 1:
            n_test = min(n_test, n_cls - 1)

        test_indices.append(cls_idx[:n_test])
        train_indices.append(cls_idx[n_test:])

    train_indices = np.sort(np.concatenate(train_indices).astype(int))
    test_indices = np.sort(np.concatenate(test_indices).astype(int))

    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]