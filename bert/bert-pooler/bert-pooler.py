import numpy as np

def tanh(x):
    return np.tanh(x)

class BertPooler:
    """
    BERT Pooler: Extracts [CLS] and applies dense + tanh.
    """
    
    def __init__(self, hidden_size: int):
        self.hidden_size = hidden_size
        self.W = np.random.randn(hidden_size, hidden_size) * 0.02
        self.b = np.zeros(hidden_size)
    
    def forward(self, hidden_states: np.ndarray) -> np.ndarray:
        """
        Pool the [CLS] token representation.
        """
        cls_hidden = hidden_states[:, 0, :]          # (batch, hidden_size)
        pooled = tanh(cls_hidden @ self.W + self.b)  # (batch, hidden_size)
        return pooled

class SequenceClassifier:
    """
    Sequence classification head on top of BERT.
    """
    
    def __init__(self, hidden_size: int, num_classes: int, dropout_prob: float = 0.1):
        self.pooler = BertPooler(hidden_size)
        self.dropout_prob = dropout_prob
        self.classifier = np.random.randn(hidden_size, num_classes) * 0.02
    
    def forward(self, hidden_states: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Classify sequences.
        """
        pooled = self.pooler.forward(hidden_states)

        if training and self.dropout_prob > 0:
            keep_prob = 1.0 - self.dropout_prob
            mask = (np.random.rand(*pooled.shape) < keep_prob).astype(float) / keep_prob
            pooled = pooled * mask

        logits = pooled @ self.classifier
        return logits