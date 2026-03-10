import math
from collections import Counter

def bleu_score(candidate, reference, max_n):
    """
    Compute the BLEU score for a candidate translation.
    """
    c_len = len(candidate)
    r_len = len(reference)

    if c_len == 0:
        return 0.0

    precisions = []

    for n in range(1, max_n + 1):
        cand_total = c_len - n + 1
        ref_total = r_len - n + 1

        if cand_total <= 0:
            return 0.0

        cand_ngrams = Counter(tuple(candidate[i:i+n]) for i in range(cand_total))
        ref_ngrams = Counter(tuple(reference[i:i+n]) for i in range(max(0, ref_total)))

        clipped = 0
        for ng, count in cand_ngrams.items():
            clipped += min(count, ref_ngrams.get(ng, 0))

        p_n = clipped / cand_total
        if p_n == 0:
            return 0.0

        precisions.append(p_n)

    if c_len < r_len:
        bp = math.exp(1 - r_len / c_len)
    else:
        bp = 1.0

    bleu = bp * math.exp(sum(math.log(p) for p in precisions) / max_n)
    return float(bleu)