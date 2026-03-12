import numpy as np

def patch_embed(image: np.ndarray, patch_size: int, embed_dim: int) -> np.ndarray:
    """
    Convert image to patch embeddings.
    Input: image of shape (B, H, W, C)
    Output: embeddings of shape (B, N, embed_dim)
    """
    B, H, W, C = image.shape
    P = patch_size

    num_h = H // P
    num_w = W // P
    N = num_h * num_w
    patch_dim = P * P * C

    # Extract non-overlapping patches
    patches = image.reshape(B, num_h, P, num_w, P, C)
    patches = patches.transpose(0, 1, 3, 2, 4, 5)
    patches = patches.reshape(B, N, patch_dim)

    # Linear projection
    W_proj = np.random.randn(patch_dim, embed_dim) * 0.02
    embeddings = patches @ W_proj

    return embeddings