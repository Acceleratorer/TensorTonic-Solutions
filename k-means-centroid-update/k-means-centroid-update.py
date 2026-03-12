def k_means_centroid_update(points, assignments, k):
    """
    Compute new centroids as the mean of assigned points.
    """
    if not points:
        return []

    dim = len(points[0])
    sums = [[0.0] * dim for _ in range(k)]
    counts = [0] * k

    for point, cluster in zip(points, assignments):
        counts[cluster] += 1
        for j in range(dim):
            sums[cluster][j] += point[j]

    centroids = []
    for cluster in range(k):
        if counts[cluster] == 0:
            centroids.append([0.0] * dim)
        else:
            centroids.append([sums[cluster][j] / counts[cluster] for j in range(dim)])

    return centroids