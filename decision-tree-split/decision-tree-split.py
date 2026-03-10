import numpy as np

def decision_tree_split(X, y):
    """
    Find the best feature and threshold to split the data.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    n, d = X.shape

    def gini(labels):
        if len(labels) == 0:
            return 0.0
        _, counts = np.unique(labels, return_counts=True)
        p = counts / len(labels)
        return 1.0 - np.sum(p ** 2)

    parent_gini = gini(y)

    best_gain = -1.0
    best_feature = None
    best_threshold = None

    for f in range(d):
        values = np.unique(X[:, f])
        if len(values) < 2:
            continue

        thresholds = (values[:-1] + values[1:]) / 2.0

        for threshold in thresholds:
            left_mask = X[:, f] <= threshold
            right_mask = ~left_mask

            if not np.any(left_mask) or not np.any(right_mask):
                continue

            y_left = y[left_mask]
            y_right = y[right_mask]

            left_gini = gini(y_left)
            right_gini = gini(y_right)

            weighted_gini = (len(y_left) / n) * left_gini + (len(y_right) / n) * right_gini
            gain = parent_gini - weighted_gini

            if (
                gain > best_gain
                or (gain == best_gain and (best_feature is None or f < best_feature))
                or (gain == best_gain and f == best_feature and threshold < best_threshold)
            ):
                best_gain = gain
                best_feature = f
                best_threshold = float(threshold)

    return [best_feature, best_threshold]