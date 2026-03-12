import numpy as np

def classification_head(encoder_output: np.ndarray, num_classes: int) -> np.ndarray:
    """
    Classification head for ViT.
    """
    cls_token = encoder_output[:, 0, :]  # (B, D)

    mean = np.mean(cls_token, axis=-1, keepdims=True)
    var = np.var(cls_token, axis=-1, keepdims=True)
    cls_norm = (cls_token - mean) / np.sqrt(var + 1e-5)

    embed_dim = cls_norm.shape[1]
    W = np.random.randn(embed_dim, num_classes) * 0.02
    b = np.zeros(num_classes)

    logits = cls_norm @ W + b
    return logits