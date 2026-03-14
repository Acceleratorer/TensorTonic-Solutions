import numpy as np

def t_test_one_sample(x, mu0):
    """
    Compute one-sample t-statistic.
    """
    x = np.asarray(x, dtype=float)
    n = x.size

    x_bar = np.mean(x)
    s = np.sqrt(np.sum((x - x_bar) ** 2) / (n - 1))  # Bessel correction
    se = s / np.sqrt(n)

    return float((x_bar - mu0) / se)