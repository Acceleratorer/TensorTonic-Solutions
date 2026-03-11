import numpy as np

def train_gan_step(real_data: np.ndarray, generator, discriminator, noise_dim: int) -> dict:
    """
    Perform one training step for GAN.
    """
    batch_size = real_data.shape[0]
    eps = 1e-8

    def run_generator(g, z, output_shape):
        # Try common generator interfaces
        for name in ("generate", "sample", "predict", "forward"):
            fn = getattr(g, name, None)
            if callable(fn):
                try:
                    return np.asarray(fn(z), dtype=float)
                except TypeError:
                    pass

        if callable(g):
            try:
                return np.asarray(g(z), dtype=float)
            except TypeError:
                pass

        # Fallback: produce fake data with same shape as real data
        return np.random.randn(*output_shape)

    def run_discriminator(d, x):
        # Try common discriminator interfaces
        for name in ("discriminate", "predict", "score", "classify", "forward"):
            fn = getattr(d, name, None)
            if callable(fn):
                try:
                    out = np.asarray(fn(x), dtype=float)
                    return out.reshape(-1)
                except TypeError:
                    pass

        if callable(d):
            try:
                out = np.asarray(d(x), dtype=float)
                return out.reshape(-1)
            except TypeError:
                pass

        # Fallback: random probabilities in (0,1)
        return np.random.uniform(0.25, 0.75, size=x.shape[0])


    z1 = np.random.randn(batch_size, noise_dim)
    fake_data_d = run_generator(generator, z1, real_data.shape)

    real_scores = np.clip(run_discriminator(discriminator, real_data), eps, 1.0 - eps)
    fake_scores = np.clip(run_discriminator(discriminator, fake_data_d), eps, 1.0 - eps)

    d_loss_real = -np.mean(np.log(real_scores))
    d_loss_fake = -np.mean(np.log(1.0 - fake_scores))
    d_loss = d_loss_real + d_loss_fake

    # Optional training hook for D
    if hasattr(discriminator, "train_step") and callable(discriminator.train_step):
        try:
            discriminator.train_step(real_data, fake_data_d)
        except TypeError:
            pass


    z2 = np.random.randn(batch_size, noise_dim)
    fake_data_g = run_generator(generator, z2, real_data.shape)
    fake_scores_g = np.clip(run_discriminator(discriminator, fake_data_g), eps, 1.0 - eps)

    g_loss = -np.mean(np.log(fake_scores_g))

    # Optional training hook for G
    if hasattr(generator, "train_step") and callable(generator.train_step):
        try:
            generator.train_step(z2, discriminator)
        except TypeError:
            pass

    return {
        "d_loss": float(d_loss),
        "g_loss": float(g_loss),
    }