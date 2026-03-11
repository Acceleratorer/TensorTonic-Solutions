import numpy as np

class VAE:
    def __init__(self, input_dim: int, latent_dim: int):
        """
        Initialize VAE.
        """
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Encoder: x -> hidden -> (mu, log_var)
        self.W_enc = np.random.randn(input_dim, latent_dim) * 0.01
        self.b_enc = np.zeros(latent_dim)

        self.W_mu = np.random.randn(latent_dim, latent_dim) * 0.01
        self.b_mu = np.zeros(latent_dim)

        self.W_logvar = np.random.randn(latent_dim, latent_dim) * 0.01
        self.b_logvar = np.zeros(latent_dim)

        # Decoder: z -> x_recon
        self.W_dec = np.random.randn(latent_dim, input_dim) * 0.01
        self.b_dec = np.zeros(input_dim)
    
    def forward(self, x: np.ndarray) -> tuple:
        """
        Full forward pass through VAE.
        Returns: (x_recon, mu, log_var)
        """
        x = np.asarray(x, dtype=float)

        h = x @ self.W_enc + self.b_enc
        mu = h @ self.W_mu + self.b_mu
        log_var = h @ self.W_logvar + self.b_logvar

        std = np.exp(0.5 * log_var)
        eps = np.random.randn(*std.shape)
        z = mu + std * eps

        x_recon = z @ self.W_dec + self.b_dec
        return x_recon, mu, log_var
    
    def generate(self, n_samples: int) -> np.ndarray:
        """
        Generate new samples from prior.
        """
        z = np.random.randn(n_samples, self.latent_dim)
        samples = z @ self.W_dec + self.b_dec
        return samples