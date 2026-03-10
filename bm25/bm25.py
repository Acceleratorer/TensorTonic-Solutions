import numpy as np
from collections import Counter
import math

def bm25_score(query_tokens, docs, k1=1.2, b=0.75):
    """
    Returns numpy array of BM25 scores for each document.
    """
    n_docs = len(docs)
    if n_docs == 0:
        return np.array([], dtype=float)

    doc_lens = np.array([len(doc) for doc in docs], dtype=float)
    avgdl = np.mean(doc_lens) if n_docs > 0 else 0.0

    doc_counters = [Counter(doc) for doc in docs]

    # Deduplicate query terms while preserving order
    query_terms = list(dict.fromkeys(query_tokens))

    # Document frequency for query terms only
    df = {}
    for term in query_terms:
        df[term] = sum(1 for doc in docs if term in set(doc))

    scores = np.zeros(n_docs, dtype=float)

    for i, tf_counter in enumerate(doc_counters):
        dl = doc_lens[i]

        for term in query_terms:
            if df.get(term, 0) == 0:
                continue

            tf = tf_counter.get(term, 0)
            if tf == 0:
                continue

            idf = math.log((n_docs - df[term] + 0.5) / (df[term] + 0.5) + 1.0)
            denom = tf + k1 * (1.0 - b + b * dl / avgdl)
            scores[i] += idf * (tf * (k1 + 1.0)) / denom

    return scores