import numpy as np

def unet(x: np.ndarray, num_classes: int = 2) -> np.ndarray:
    """
    Complete U-Net for segmentation.
    Shape simulation only.
    Assumes input is NHWC: (batch, height, width, channels)
    """
    b, h, w, _ = x.shape

    # Original U-Net output is input spatial size minus 184
    out_h = h - 184
    out_w = w - 184

    return np.zeros((b, out_h, out_w, num_classes), dtype=float)