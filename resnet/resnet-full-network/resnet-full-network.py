import numpy as np

def relu(x):
    return np.maximum(0, x)

def _apply_linear_lastdim(x, W):
    """
    Apply channel projection on the last dimension.
    Works for (..., C).
    """
    return np.tensordot(x, W, axes=([-1], [0]))

def _downsample_spatial(x):
    """
    Halve spatial dimensions if x is 4D NHWC.
    """
    return x[:, ::2, ::2, :]

class BasicBlock:
    """Basic residual block (2 conv layers with skip connection)."""
    
    def __init__(self, in_ch: int, out_ch: int, downsample: bool = False):
        self.downsample = downsample
        self.W1 = np.random.randn(in_ch, out_ch) * 0.01
        self.W2 = np.random.randn(out_ch, out_ch) * 0.01
        # Projection shortcut if dimensions change
        self.W_proj = np.random.randn(in_ch, out_ch) * 0.01 if in_ch != out_ch or downsample else None
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass: Conv -> ReLU -> Conv -> Add Skip -> ReLU

        Expected main image format: (B, H, W, C)
        Also supports 2D inputs: (B, C)
        """
        identity = x

        out = _apply_linear_lastdim(x, self.W1)
        out = relu(out)
        out = _apply_linear_lastdim(out, self.W2)

        # Downsample only if spatial dimensions exist
        if self.downsample and out.ndim == 4:
            out = _downsample_spatial(out)

        if self.W_proj is not None:
            identity = _apply_linear_lastdim(identity, self.W_proj)
            if self.downsample and identity.ndim == 4:
                identity = _downsample_spatial(identity)

        out = out + identity
        out = relu(out)
        return out

class ResNet18:
    """
    Simplified ResNet-18 architecture.
    
    Structure:
    - conv1: 3 -> 64 channels
    - layer1: 2 BasicBlocks, 64 channels
    - layer2: 2 BasicBlocks, 128 channels (first downsamples)
    - layer3: 2 BasicBlocks, 256 channels (first downsamples)
    - layer4: 2 BasicBlocks, 512 channels (first downsamples)
    - fc: 512 -> num_classes
    """
    
    def __init__(self, num_classes: int = 10):
        self.conv1 = np.random.randn(3, 64) * 0.01
        
        # Build layers
        self.layer1 = [
            BasicBlock(64, 64, downsample=False),
            BasicBlock(64, 64, downsample=False),
        ]
        self.layer2 = [
            BasicBlock(64, 128, downsample=True),
            BasicBlock(128, 128, downsample=False),
        ]
        self.layer3 = [
            BasicBlock(128, 256, downsample=True),
            BasicBlock(256, 256, downsample=False),
        ]
        self.layer4 = [
            BasicBlock(256, 512, downsample=True),
            BasicBlock(512, 512, downsample=False),
        ]
        
        self.fc = np.random.randn(512, num_classes) * 0.01
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass through ResNet-18.

        Expected input image format: (B, H, W, 3)
        """
        out = _apply_linear_lastdim(x, self.conv1)
        out = relu(out)

        for block in self.layer1:
            out = block.forward(out)
        for block in self.layer2:
            out = block.forward(out)
        for block in self.layer3:
            out = block.forward(out)
        for block in self.layer4:
            out = block.forward(out)

        # Global average pooling over spatial dimensions
        if out.ndim == 4:
            out = np.mean(out, axis=(1, 2))   # (B, 512)
        elif out.ndim == 2:
            pass
        else:
            raise ValueError("Unexpected tensor rank in ResNet18.forward")

        logits = out @ self.fc
        return logits