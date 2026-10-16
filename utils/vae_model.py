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
    cond_dim: int = 0

    def setup(self):
        enc_out = 2 * self.latent_dim if self.vae else self.latent_dim
        self.encoder = MLP(list(self.hidden_dims) + [enc_out], activate_final=False)
        self.decoder = MLP(
            list(reversed(self.hidden_dims)) + [self.input_dim], activate_final=False
        )

    def _concat_cond(self, x: jnp.ndarray, cond: Optional[jnp.ndarray]) -> jnp.ndarray:
        if not self.cond_dim:
            return x
        if cond is None:
            raise ValueError("State conditioning is enabled but no state was passed")
        return jnp.concatenate([x, cond], axis=-1)

    def decode(self, z: jnp.ndarray, cond: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        return self.decoder(self._concat_cond(z, cond))

    def __call__(
        self,
        x: jnp.ndarray,
        latent_rng: Optional[jax.Array] = None,
        sample: bool = False,
        cond: Optional[jnp.ndarray] = None,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        h = self.encoder(self._concat_cond(x, cond))
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
        return self.decode(z, cond), mu, logvar


def decode_latents_jax(
    params,
    z,
    mean,
    std,
    input_dim: int,
    latent_dim: int,
    hidden_dims=(256, 256),
    cond=None,
    cond_dim: int = 0,
    state_mean=None,
    state_std=None,
):
    """JAX decode for SAC updates: z -> unnormalized residual flat.

    `cond_dim > 0` concatenates the normalized proprio vector onto z.
    """
    module = MLPAutoencoder(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        vae=True,
        cond_dim=int(cond_dim),
    )
    if int(cond_dim) > 0:
        cond = jnp.reshape(cond, (z.shape[0], int(cond_dim)))
        cond = (cond - state_mean) / state_std
        recon = module.apply({"params": params}, z, cond, method=module.decode)
    else:
        recon = module.apply({"params": params}, z, method=module.decode)
    return recon * std + mean
