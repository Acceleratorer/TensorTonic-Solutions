import numpy as np
from typing import Tuple

def apply_mlm_mask(
    token_ids: np.ndarray,
    vocab_size: int,
    mask_token_id: int = 103,
    mask_prob: float = 0.15,
    seed: int = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply BERT's MLM masking strategy.
    """
    if seed is not None:
        np.random.seed(seed)

    token_ids = np.asarray(token_ids, dtype=int).copy()

    labels = np.full_like(token_ids, -100)
    masked_positions = np.zeros_like(token_ids, dtype=bool)

    # Don't mask special tokens
    special_tokens = {0, 101, 102}  # [PAD], [CLS], [SEP]
    candidate_mask = ~np.isin(token_ids, list(special_tokens))

    random_mask = np.random.rand(*token_ids.shape) < mask_prob
    masked_positions = candidate_mask & random_mask

    labels[masked_positions] = token_ids[masked_positions]

    # 80 / 10 / 10 strategy
    probs = np.random.rand(*token_ids.shape)

    mask_mask = masked_positions & (probs < 0.8)
    random_token_mask = masked_positions & (probs >= 0.8) & (probs < 0.9)
    # remaining 10% unchanged automatically

    token_ids[mask_mask] = mask_token_id
    token_ids[random_token_mask] = np.random.randint(0, vocab_size, size=np.sum(random_token_mask))

    return token_ids, labels, masked_positions

class MLMHead:
    """Masked LM prediction head."""
    
    def __init__(self, hidden_size: int, vocab_size: int):
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.W = np.random.randn(hidden_size, vocab_size) * 0.02
        self.b = np.zeros(vocab_size)
    
    def forward(self, hidden_states: np.ndarray) -> np.ndarray:
        """
        Predict token probabilities.
        """
        logits = hidden_states @ self.W + self.b
        return logits