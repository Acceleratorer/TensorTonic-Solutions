import numpy as np

def roc_curve(y_true, y_score):
    """
    Compute ROC curve from binary labels and scores.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)

    # Sort by descending score
    order = np.argsort(-y_score, kind="mergesort")
    y_true_sorted = y_true[order]
    y_score_sorted = y_score[order]

    # Cumulative TP/FP
    tps = np.cumsum(y_true_sorted == 1)
    fps = np.cumsum(y_true_sorted == 0)

    # Keep only last index of each score group
    distinct = np.where(np.diff(y_score_sorted) != 0)[0]
    threshold_idxs = np.r_[distinct, len(y_score_sorted) - 1]

    tps = tps[threshold_idxs]
    fps = fps[threshold_idxs]
    thresholds = y_score_sorted[threshold_idxs]

    P = np.sum(y_true == 1)
    N = np.sum(y_true == 0)

    tpr = tps / P
    fpr = fps / N

    # Add starting point
    tpr = np.r_[0.0, tpr]
    fpr = np.r_[0.0, fpr]
    thresholds = np.r_[np.inf, thresholds]

    return fpr.tolist(), tpr.tolist(), thresholds.tolist()