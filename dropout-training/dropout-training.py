import numpy as np

def dropout(x, p=0.5, rng=None):
    """
    Apply dropout to input x with probability p.
    Return (output, dropout_pattern).
    """
    x = np.asarray(x, dtype=float)

    if p == 0.0:
        pattern = np.ones_like(x, dtype=float)
        return x.copy(), pattern

    keep_prob = 1.0 - p
    rand = rng.random(x.shape) if rng is not None else np.random.random(x.shape)
    pattern = np.where(rand < keep_prob, 1.0 / keep_prob, 0.0)

    output = x * pattern
    return output, pattern