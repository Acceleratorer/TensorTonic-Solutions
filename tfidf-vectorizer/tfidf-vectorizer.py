import numpy as np
from collections import Counter
import math

def tfidf_vectorizer(documents):
    """
    Build TF-IDF matrix from a list of text documents.
    Returns tuple of (tfidf_matrix, vocabulary).
    """
    n_docs = len(documents)
    if n_docs == 0:
        return np.zeros((0, 0), dtype=float), []

    tokenized = [doc.lower().split() for doc in documents]

    vocab = sorted({token for doc in tokenized for token in doc})
    if len(vocab) == 0:
        return np.zeros((n_docs, 0), dtype=float), []

    word_to_idx = {word: i for i, word in enumerate(vocab)}

    # Document frequency
    df = Counter()
    for doc in tokenized:
        for word in set(doc):
            df[word] += 1

    # IDF
    idf = np.zeros(len(vocab), dtype=float)
    for word, idx in word_to_idx.items():
        idf[idx] = math.log(n_docs / df[word])

    # TF-IDF matrix
    tfidf = np.zeros((n_docs, len(vocab)), dtype=float)

    for i, doc in enumerate(tokenized):
        if len(doc) == 0:
            continue

        counts = Counter(doc)
        total_terms = len(doc)

        for word, count in counts.items():
            j = word_to_idx[word]
            tf = count / total_terms
            tfidf[i, j] = tf * idf[j]

    return tfidf, vocab