import numpy as np

class BertEmbeddings:
    """
    BERT Embeddings = Token + Position + Segment
    """
    
    def __init__(self, vocab_size: int, max_position: int, hidden_size: int):
        self.hidden_size = hidden_size
        
        # Token embeddings
        self.token_embeddings = np.random.randn(vocab_size, hidden_size) * 0.02
        
        # Position embeddings (learned, not sinusoidal)
        self.position_embeddings = np.random.randn(max_position, hidden_size) * 0.02
        
        # Segment embeddings (just 2 segments: A and B)
        self.segment_embeddings = np.random.randn(2, hidden_size) * 0.02
    
    def forward(self, token_ids: np.ndarray, segment_ids: np.ndarray) -> np.ndarray:
        """
        Compute BERT embeddings.
        """
        batch_size, seq_len = token_ids.shape

        tok_emb = self.token_embeddings[token_ids]  # (B, L, H)

        positions = np.arange(seq_len)
        pos_emb = self.position_embeddings[positions]          # (L, H)
        pos_emb = np.broadcast_to(pos_emb, (batch_size, seq_len, self.hidden_size))

        seg_emb = self.segment_embeddings[segment_ids]         # (B, L, H)

        return tok_emb + pos_emb + seg_emb