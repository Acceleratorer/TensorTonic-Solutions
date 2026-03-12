import numpy as np

def layer_norm(x, eps=1e-5):
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps)

def gelu(x):
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))

def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def vit_encoder_block(x: np.ndarray, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0) -> np.ndarray:
    """
    ViT Transformer encoder block.
    """
    B, N, D = x.shape
    head_dim = embed_dim // num_heads
    mlp_dim = int(embed_dim * mlp_ratio)

    # ----- Pre-LN + MSA -----
    x_ln = layer_norm(x)

    W_q = np.random.randn(embed_dim, embed_dim) * 0.02
    W_k = np.random.randn(embed_dim, embed_dim) * 0.02
    W_v = np.random.randn(embed_dim, embed_dim) * 0.02
    W_o = np.random.randn(embed_dim, embed_dim) * 0.02

    Q = x_ln @ W_q
    K = x_ln @ W_k
    V = x_ln @ W_v

    Q = Q.reshape(B, N, num_heads, head_dim).transpose(0, 2, 1, 3)
    K = K.reshape(B, N, num_heads, head_dim).transpose(0, 2, 1, 3)
    V = V.reshape(B, N, num_heads, head_dim).transpose(0, 2, 1, 3)

    scores = (Q @ K.transpose(0, 1, 3, 2)) / np.sqrt(head_dim)
    attn = softmax(scores, axis=-1)
    context = attn @ V

    context = context.transpose(0, 2, 1, 3).reshape(B, N, D)
    msa_out = context @ W_o

    x = x + msa_out

    # ----- Pre-LN + MLP -----
    x_ln2 = layer_norm(x)

    W1 = np.random.randn(embed_dim, mlp_dim) * 0.02
    b1 = np.zeros(mlp_dim)
    W2 = np.random.randn(mlp_dim, embed_dim) * 0.02
    b2 = np.zeros(embed_dim)

    mlp_out = gelu(x_ln2 @ W1 + b1)
    mlp_out = mlp_out @ W2 + b2

    x = x + mlp_out
    return x
    