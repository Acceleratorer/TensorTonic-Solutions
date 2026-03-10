import numpy as np

def kfold_split(N, k, shuffle=True, rng=None):
    """
    Returns: list of length k with tuples (train_idx, val_idx)
    """
    indices = np.arange(N, dtype=int)

    if shuffle:
        if rng is not None:
            indices = rng.permutation(indices)
        else:
            indices = indices.copy()
            np.random.shuffle(indices)

    folds = np.array_split(indices, k)
    splits = []

    for i in range(k):
        val_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(k) if j != i]).astype(int)
        splits.append((train_idx, val_idx.astype(int)))

    return splits