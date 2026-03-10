import numpy as np

def naive_bayes_bernoulli(X_train, y_train, X_test):
    """
    Compute log-likelihood P(y|x) for Bernoulli Naive Bayes.
    """
    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train)
    X_test = np.asarray(X_test, dtype=float)

    classes = np.unique(y_train)
    n_classes = len(classes)
    n_train, d = X_train.shape

    log_posteriors = np.zeros((X_test.shape[0], n_classes), dtype=float)

    for j, c in enumerate(classes):
        X_c = X_train[y_train == c]
        n_c = X_c.shape[0]

        prior = n_c / n_train
        theta = (X_c.sum(axis=0) + 1.0) / (n_c + 2.0)   # Laplace smoothing alpha=1

        log_prior = np.log(prior)
        log_theta = np.log(theta)
        log_one_minus_theta = np.log(1.0 - theta)

        log_posteriors[:, j] = (
            log_prior
            + X_test @ log_theta
            + (1.0 - X_test) @ log_one_minus_theta
        )

    return log_posteriors