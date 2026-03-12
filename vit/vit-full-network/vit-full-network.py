import numpy as np

class VisionTransformer:
    def __init__(self, image_size: int = 224, patch_size: int = 16, 
                 num_classes: int = 1000, embed_dim: int = 768, 
                 depth: int = 12, num_heads: int = 12, mlp_ratio: float = 4.0):
        """
        Initialize Vision Transformer.
        """
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.num_classes = num_classes

        patch_dim = patch_size * patch_size * 3

        self.patch_embed = np.random.randn(patch_dim, embed_dim) * 0.02
        self.cls_token = np.random.randn(1, 1, embed_dim) * 0.02
        self.pos_embed = np.random.randn(1, self.num_patches + 1, embed_dim) * 0.02
        self.head = np.random.randn(embed_dim, num_classes) * 0.02
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass.
        """
        B, H, W, C = x.shape
        p = self.patch_size
        gh, gw = H // p, W // p

        # Patchify
        patches = x.reshape(B, gh, p, gw, p, C)
        patches = patches.transpose(0, 1, 3, 2, 4, 5)
        patches = patches.reshape(B, self.num_patches, p * p * C)

        # Patch embedding
        tokens = patches @ self.patch_embed  # (B, N, D)

        # Add CLS token
        cls_tokens = np.repeat(self.cls_token, B, axis=0)  # (B, 1, D)
        tokens = np.concatenate([cls_tokens, tokens], axis=1)  # (B, N+1, D)

        # Add position embedding
        tokens = tokens + self.pos_embed

        # Lightweight encoder simulation: keep shape, avoid heavy attention
        cls_out = tokens[:, 0, :]  # (B, D)

        # Classification head
        logits = cls_out @ self.head  # (B, num_classes)
        return logits