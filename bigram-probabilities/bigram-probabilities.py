def bigram_probabilities(tokens):
    """
    Returns: (counts, probs)
      counts: dict mapping (w1, w2) -> integer count
      probs: dict mapping (w1, w2) -> float P(w2 | w1) with add-1 smoothing
    """
    vocab = sorted(set(tokens))
    V = len(vocab)

    counts = {}
    context_counts = {w: 0 for w in vocab}

    # Count bigrams
    for i in range(len(tokens) - 1):
        w1, w2 = tokens[i], tokens[i + 1]
        counts[(w1, w2)] = counts.get((w1, w2), 0) + 1
        context_counts[w1] += 1

    # Add-1 smoothed probabilities for every pair in V x V
    probs = {}
    for w1 in vocab:
        denom = context_counts[w1] + V
        for w2 in vocab:
            c = counts.get((w1, w2), 0)
            probs[(w1, w2)] = (c + 1) / denom

    return counts, probs