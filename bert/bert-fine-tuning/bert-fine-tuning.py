import numpy as np
from typing import List, Optional

class MockBertEncoder:
    """Simulated BERT encoder with 12 layers."""
    
    def __init__(self, hidden_size: int = 768, num_layers: int = 12):
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # Each layer just adds a small transformation
        self.layers = [np.random.randn(hidden_size, hidden_size) * 0.01 for _ in range(num_layers)]
        self.layer_frozen = [False] * num_layers
    
    def freeze_layers(self, layer_indices: List[int]):
        """Freeze specified layers (no gradient updates)."""
        for idx in layer_indices:
            if 0 <= idx < self.num_layers:
                self.layer_frozen[idx] = True
    
    def unfreeze_all(self):
        """Unfreeze all layers."""
        self.layer_frozen = [False] * self.num_layers
    
    def forward(self, embeddings: np.ndarray) -> np.ndarray:
        """Forward pass through all layers."""
        x = embeddings
        for i, layer in enumerate(self.layers):
            if not self.layer_frozen[i]:
                x = x @ layer + x  # Simplified residual
            else:
                # Frozen: still compute but mark as no-grad
                x = x @ layer + x
        return x

class BertForSequenceClassification:
    """BERT with classification head."""
    
    def __init__(self, hidden_size: int, num_labels: int, freeze_bert: bool = False):
        self.encoder = MockBertEncoder(hidden_size)
        self.classifier = np.random.randn(hidden_size, num_labels) * 0.02
        self.freeze_bert = freeze_bert
        
        if freeze_bert:
            self.encoder.freeze_layers(list(range(12)))
    
    def forward(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Forward pass for classification.
        """
        hidden_states = self.encoder.forward(embeddings)   # (B, L, H)
        cls_state = hidden_states[:, 0, :]                 # (B, H)
        logits = cls_state @ self.classifier               # (B, num_labels)
        return logits

class BertForTokenClassification:
    """BERT with token-level classification (NER, POS tagging)."""
    
    def __init__(self, hidden_size: int, num_labels: int):
        self.encoder = MockBertEncoder(hidden_size)
        self.classifier = np.random.randn(hidden_size, num_labels) * 0.02
    
    def forward(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Forward pass for token classification.
        """
        hidden_states = self.encoder.forward(embeddings)   # (B, L, H)
        logits = hidden_states @ self.classifier           # (B, L, num_labels)
        return logits