import numpy as np

def confusion_matrix_norm(y_true, y_pred, num_classes=None, normalize='none'):
    """
    Compute confusion matrix with optional normalization.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")

    if num_classes is None:
        if y_true.size == 0 and y_pred.size == 0:
            num_classes = 0
        else:
            num_classes = int(max(y_true.max(initial=0), y_pred.max(initial=0)) + 1)

    K = num_classes

    if np.any(y_true < 0) or np.any(y_true >= K) or np.any(y_pred < 0) or np.any(y_pred >= K):
        raise ValueError("labels must be in range [0, num_classes-1]")

    cm = np.bincount(y_true * K + y_pred, minlength=K * K).reshape(K, K)

    if normalize == 'none':
        return cm

    cm = cm.astype(float)

    if normalize == 'true':
        denom = cm.sum(axis=1, keepdims=True)
        denom[denom == 0] = 1.0
        return cm / denom

    if normalize == 'pred':
        denom = cm.sum(axis=0, keepdims=True)
        denom[denom == 0] = 1.0
        return cm / denom

    if normalize == 'all':
        denom = cm.sum()
        if denom == 0:
            denom = 1.0
        return cm / denom

    raise ValueError("normalize must be one of: 'none', 'true', 'pred', 'all'")