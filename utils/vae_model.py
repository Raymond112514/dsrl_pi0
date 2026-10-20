"""Flax MLP autoencoder / VAE (copied from serl/autoencoder/model.py)."""

from typing import Optional, Sequence, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp


class MLP(nn.Module):
    dims: Sequence[int]
    activate_final: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for i, dim in enumerate(self.dims):
            x = nn.Dense(dim, kernel_init=nn.initializers.xavier_uniform())(x)
            if i + 1 < len(self.dims) or self.activate_final:
                x = nn.relu(x)
        return x


class MLPAutoencoder(nn.Module):
    input_dim: int
    latent_dim: int
    hidden_dims: Sequence[int] = (256, 256)
    vae: bool = False

    def setup(self):
        enc_out = 2 * self.latent_dim if self.vae else self.latent_dim
        self.encoder = MLP(list(self.hidden_dims) + [enc_out], activate_final=False)
        self.decoder = MLP(
            list(reversed(self.hidden_dims)) + [self.input_dim], activate_final=False
        )

    def decode(self, z: jnp.ndarray) -> jnp.ndarray:
        return self.decoder(z)

    def __call__(
        self,
        x: jnp.ndarray,
        latent_rng: Optional[jax.Array] = None,
        sample: bool = False,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        h = self.encoder(x)
        if self.vae:
            mu, logvar = jnp.split(h, 2, axis=-1)
            logvar = jnp.clip(logvar, -20.0, 10.0)
            if sample:
                eps = jax.random.normal(latent_rng, mu.shape)
                z = mu + jnp.exp(0.5 * logvar) * eps
            else:
                z = mu
        else:
            mu = h
            logvar = jnp.zeros_like(mu)
            z = mu
        return self.decode(z), mu, logvar


def decode_latents_jax(
    params,
    z,
    mean,
    std,
    input_dim: int,
    latent_dim: int,
    hidden_dims=(256, 256),
):
    """JAX decode for SAC updates: z -> std * decoder(z) + mean."""
    module = MLPAutoencoder(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        vae=True,
    )
    recon = module.apply({"params": params}, z, method=module.decode)
    return recon * std + mean
