import numpy as np

def vae_loss(x: np.ndarray, x_recon: np.ndarray, mu: np.ndarray, log_var: np.ndarray) -> dict:
    """
    Compute VAE ELBO loss.
    """
    x = np.asarray(x, dtype=float)
    x_recon = np.asarray(x_recon, dtype=float)
    mu = np.asarray(mu, dtype=float)
    log_var = np.asarray(log_var, dtype=float)

    # Reconstruction loss: sum over features, mean over batch
    recon = np.sum((x - x_recon) ** 2, axis=1)
    recon = np.mean(recon)

    # KL divergence: sum over latent dims, mean over batch
    kl = -0.5 * np.sum(1 + log_var - mu**2 - np.exp(log_var), axis=1)
    kl = np.mean(kl)

    total = recon + kl

    return {
        "total": float(total),
        "recon": float(recon),
        "kl": float(kl),
    }