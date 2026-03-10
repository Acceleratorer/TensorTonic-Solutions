def baseline_predict(ratings_matrix, target_pairs):
    """
    Compute baseline predictions using global mean and user/item biases.
    """
    # Global mean over all non-zero ratings
    nonzero = [r for row in ratings_matrix for r in row if r != 0]
    mu = sum(nonzero) / len(nonzero)

    # User biases
    user_bias = []
    for row in ratings_matrix:
        vals = [r for r in row if r != 0]
        user_mean = sum(vals) / len(vals) if vals else mu
        user_bias.append(user_mean - mu)

    # Item biases
    n_items = len(ratings_matrix[0])
    item_bias = []
    for j in range(n_items):
        vals = [ratings_matrix[i][j] for i in range(len(ratings_matrix)) if ratings_matrix[i][j] != 0]
        item_mean = sum(vals) / len(vals) if vals else mu
        item_bias.append(item_mean - mu)

    # Predictions
    preds = []
    for u, i in target_pairs:
        preds.append(mu + user_bias[u] + item_bias[i])

    return preds