import numpy as np
from typing import List, Tuple
import random

def create_nsp_examples(
    documents: List[List[str]], 
    num_examples: int,
    seed: int = None
) -> List[Tuple[str, str, int]]:
    """
    Create NSP training examples.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    examples = []

    # collect all valid positive pairs
    positive_pairs = []
    for doc in documents:
        for i in range(len(doc) - 1):
            positive_pairs.append((doc[i], doc[i + 1], 1))

    all_sentences = []
    sent_to_doc = []
    for doc_idx, doc in enumerate(documents):
        for sent in doc:
            all_sentences.append(sent)
            sent_to_doc.append(doc_idx)

    for i in range(num_examples):
        make_positive = (i < num_examples / 2)

        if make_positive and positive_pairs:
            examples.append(random.choice(positive_pairs))
        else:
            # negative pair
            doc_a_idx = random.randrange(len(documents))
            doc_a = documents[doc_a_idx]

            sent_a = random.choice(doc_a)

            # prefer different document for negative B
            candidate_docs = [j for j in range(len(documents)) if j != doc_a_idx and len(documents[j]) > 0]
            if candidate_docs:
                doc_b_idx = random.choice(candidate_docs)
                sent_b = random.choice(documents[doc_b_idx])
            else:
                # fallback if only one document exists
                sent_b = random.choice(all_sentences)

            examples.append((sent_a, sent_b, 0))

    random.shuffle(examples)
    return examples

class NSPHead:
    """Next Sentence Prediction classification head."""
    
    def __init__(self, hidden_size: int):
        self.W = np.random.randn(hidden_size, 2) * 0.02
        self.b = np.zeros(2)
    
    def forward(self, cls_hidden: np.ndarray) -> np.ndarray:
        """
        Predict IsNext probability.
        """
        return cls_hidden @ self.W + self.b

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)