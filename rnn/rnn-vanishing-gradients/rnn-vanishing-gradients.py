import numpy as np

def compute_gradient_norm_decay(T: int, W_hh: np.ndarray) -> list:
    """
    Simulate gradient norm decay over T time steps.
    Returns list of gradient norms.
    """
    spectral_norm = np.linalg.norm(W_hh, ord=2)
    
    norms = []
    grad = 1.0  # relative to final time step
    
    for _ in range(T):
        norms.append(float(grad))
        grad *= spectral_norm
    
    return norms