import numpy as np

def chi2_independence(C):
    """
    Compute chi-square test statistic and expected frequencies.
    """
    C = np.asarray(C, dtype=float)

    row_totals = np.sum(C, axis=1)
    col_totals = np.sum(C, axis=0)
    total = np.sum(C)

    expected = np.outer(row_totals, col_totals) / total
    chi2 = np.sum((C - expected) ** 2 / expected)

    return float(chi2), expected