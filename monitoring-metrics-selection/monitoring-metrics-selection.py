import math

def compute_monitoring_metrics(system_type, y_true, y_pred):
    """
    Compute the appropriate monitoring metrics for the given system type.
    """
    n = len(y_true)

    if system_type == "classification":
        tp = fp = tn = fn = 0

        for yt, yp in zip(y_true, y_pred):
            if yt == 1 and yp == 1:
                tp += 1
            elif yt == 0 and yp == 1:
                fp += 1
            elif yt == 0 and yp == 0:
                tn += 1
            elif yt == 1 and yp == 0:
                fn += 1

        accuracy = (tp + tn) / n if n > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        metrics = [
            ("accuracy", accuracy),
            ("f1", f1),
            ("precision", precision),
            ("recall", recall),
        ]
        return sorted(metrics, key=lambda x: x[0])

    elif system_type == "regression":
        abs_errors = [abs(yt - yp) for yt, yp in zip(y_true, y_pred)]
        sq_errors = [(yt - yp) ** 2 for yt, yp in zip(y_true, y_pred)]

        mae = sum(abs_errors) / n if n > 0 else 0.0
        rmse = math.sqrt(sum(sq_errors) / n) if n > 0 else 0.0

        metrics = [
            ("mae", mae),
            ("rmse", rmse),
        ]
        return sorted(metrics, key=lambda x: x[0])

    elif system_type == "ranking":
        paired = sorted(zip(y_pred, y_true), reverse=True)
        top3 = paired[:3]

        relevant_in_top3 = sum(label for _, label in top3)
        total_relevant = sum(y_true)

        precision_at_3 = relevant_in_top3 / 3 if len(top3) > 0 else 0.0
        recall_at_3 = relevant_in_top3 / total_relevant if total_relevant > 0 else 0.0

        metrics = [
            ("precision_at_3", precision_at_3),
            ("recall_at_3", recall_at_3),
        ]
        return sorted(metrics, key=lambda x: x[0])

    else:
        raise ValueError("system_type must be 'classification', 'regression', or 'ranking'")