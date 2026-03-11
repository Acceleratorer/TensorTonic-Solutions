import numpy as np

class GAN:
    def __init__(self, data_dim: int, noise_dim: int):
        """
        Initialize GAN.
        """
        self.data_dim = data_dim
        self.noise_dim = noise_dim

        # Generator: z -> x_fake
        self.Wg = np.random.randn(noise_dim, data_dim) * 0.02
        self.bg = np.zeros(data_dim)

        # Discriminator: x -> score
        self.Wd = np.random.randn(data_dim, 1) * 0.02
        self.bd = np.zeros(1)
    
    def _sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))
    
    def generate(self, n_samples: int) -> np.ndarray:
        """Generate fake samples."""
        z = np.random.randn(n_samples, self.noise_dim)
        samples = z @ self.Wg + self.bg
        return samples
    
    def discriminate(self, x: np.ndarray) -> np.ndarray:
        """Classify samples as real/fake."""
        logits = x @ self.Wd + self.bd
        probs = self._sigmoid(logits)
        return probs.squeeze(-1)
    
    def train_step(self, real_data: np.ndarray) -> dict:
        """Perform one training step."""
        batch_size = real_data.shape[0]
        eps = 1e-8

        # Discriminator step
        fake_data_d = self.generate(batch_size)
        real_scores = self.discriminate(real_data)
        fake_scores = self.discriminate(fake_data_d)

        d_loss_real = -np.mean(np.log(real_scores + eps))
        d_loss_fake = -np.mean(np.log(1.0 - fake_scores + eps))
        d_loss = d_loss_real + d_loss_fake

        # Generator step
        fake_data_g = self.generate(batch_size)
        fake_scores_g = self.discriminate(fake_data_g)
        g_loss = -np.mean(np.log(fake_scores_g + eps))

        return {
            "d_loss": float(d_loss),
            "g_loss": float(g_loss),
        }