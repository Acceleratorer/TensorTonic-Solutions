import numpy as np

def add_position_embedding(patches: np.ndarray, num_patches: int, embed_dim: int) -> np.ndarray:
    """
    Add learnable position embeddings to patch embeddings.
    """
    pos_embed = np.random.randn(1, num_patches, embed_dim) * 0.02
    return patches + pos_embed