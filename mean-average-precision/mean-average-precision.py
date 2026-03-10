import numpy as np

def mean_average_precision(y_true_list, y_score_list, k=None):
    """
    Compute Mean Average Precision (mAP) for multiple retrieval queries.
    Returns: (map_value, ap_per_query)
    """
    ap_per_query = []

    for y_true, y_score in zip(y_true_list, y_score_list):
        y_true = np.asarray(y_true, dtype=int)
        y_score = np.asarray(y_score, dtype=float)

        order = np.argsort(-y_score)
        y_true_sorted = y_true[order]

        total_relevant = np.sum(y_true_sorted)   # full query, before cutoff

        if k is not None:
            y_true_sorted = y_true_sorted[:k]

        if total_relevant == 0:
            ap_per_query.append(0.0)
            continue

        cumsum_relevant = np.cumsum(y_true_sorted)
        ranks = np.arange(1, len(y_true_sorted) + 1)
        precision_at_k = cumsum_relevant / ranks

        ap = np.sum(precision_at_k * y_true_sorted) / total_relevant
        ap_per_query.append(float(ap))

    map_value = float(np.mean(ap_per_query)) if ap_per_query else 0.0
    return map_value, ap_per_query