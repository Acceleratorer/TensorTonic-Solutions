import numpy as np

def info_nce_loss(Z1, Z2, temperature=0.1):
    """
    Compute InfoNCE Loss for contrastive learning.
    """
    Z1 = np.asarray(Z1, dtype=float)
    Z2 = np.asarray(Z2, dtype=float)

    S = (Z1 @ Z2.T) / temperature                      # (N, N)

    # Stable log-softmax over each row
    S_max = np.max(S, axis=1, keepdims=True)
    S_stable = S - S_max
    logsumexp = np.log(np.sum(np.exp(S_stable), axis=1)) + S_max[:, 0]

    positive = np.diag(S)
    loss = -np.mean(positive - logsumexp)

    return float(loss)