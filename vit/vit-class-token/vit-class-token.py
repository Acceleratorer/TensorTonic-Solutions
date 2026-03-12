import numpy as np

def prepend_class_token(patches: np.ndarray, embed_dim: int) -> np.ndarray:
    """
    Prepend learnable [CLS] token to patch sequence.
    """
    B = patches.shape[0]
    cls_token = np.random.randn(1, 1, embed_dim) * 0.02
    cls_tokens = np.tile(cls_token, (B, 1, 1))
    return np.concatenate([cls_tokens, patches], axis=1)