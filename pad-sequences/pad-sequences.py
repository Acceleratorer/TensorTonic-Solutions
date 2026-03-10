import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):
    """
    Pad or truncate a list of integer sequences to the same length.
    Returns a NumPy array of dtype int.
    """
    if len(seqs) == 0:
        return np.empty((0, 0), dtype=int)

    if max_len is None:
        max_len = max(len(seq) for seq in seqs) if seqs else 0

    result = np.full((len(seqs), max_len), pad_value, dtype=int)

    for i, seq in enumerate(seqs):
        seq_arr = np.asarray(seq, dtype=int)
        trunc = seq_arr[:max_len]
        result[i, :len(trunc)] = trunc

    return result