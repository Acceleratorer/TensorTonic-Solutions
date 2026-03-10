import numpy as np

def impute_missing(X, strategy='mean'):
    """
    Fill NaN values in each feature column using column mean or median.
    """
    X = np.asarray(X, dtype=float)
    out = X.copy()

    if out.ndim == 1:
        nan_mask = np.isnan(out)
        valid = ~nan_mask

        if np.any(valid):
            fill_value = np.mean(out[valid]) if strategy == 'mean' else np.median(out[valid])
        else:
            fill_value = 0.0

        out[nan_mask] = fill_value
        return out

    for j in range(out.shape[1]):
        col = out[:, j]
        nan_mask = np.isnan(col)
        valid = ~nan_mask

        if np.any(valid):
            fill_value = np.mean(col[valid]) if strategy == 'mean' else np.median(col[valid])
        else:
            fill_value = 0.0

        col[nan_mask] = fill_value

    return out