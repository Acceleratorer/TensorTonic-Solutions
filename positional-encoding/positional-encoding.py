import numpy as np

def positional_encoding(seq_len, d_model, base=10000.0):
    """
    Return PE of shape (seq_len, d_model) using sin/cos formulation.
    Odd d_model -> last column is sin.
    """
    positions = np.arange(seq_len, dtype=float)[:, None]              # (T, 1)
    div_terms = base ** (2 * np.arange((d_model + 1) // 2, dtype=float) / d_model)  # (ceil(d/2),)

    angles = positions / div_terms[None, :]                          # (T, ceil(d/2))

    pe = np.empty((seq_len, d_model), dtype=float)
    pe[:, 0::2] = np.sin(angles)
    pe[:, 1::2] = np.cos(angles[:, :d_model // 2])

    return pe