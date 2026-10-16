"""Conditional flow matching on action chunks (noise → data).

Matches serl/autoencoder/flow_matching.py: OT-CFM path
x_t = (1-t) ε + t x_1, target velocity u = x_1 - ε.
Decode is an Euler ODE from x_0 = ε to x_1.
"""

from typing import Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp

from utils.vae_model import MLP


class FlowMatchingMLP(nn.Module):
    """Velocity field v_θ(x_t, t) for OT-CFM on R^{dim}."""

    dim: int
    hidden_dims: Sequence[int] = (256, 256, 256)
    time_embed_dim: int = 64
    n_steps: int = 10

    def setup(self):
        self.time_mlp = MLP((self.time_embed_dim, self.time_embed_dim), activate_final=True)
        self.velocity_mlp = MLP(
            list(self.hidden_dims) + [self.dim],
            activate_final=False,
        )

    def _time_embed(self, t: jnp.ndarray) -> jnp.ndarray:
        t = jnp.asarray(t, dtype=jnp.float32).reshape(-1)
        half = self.time_embed_dim // 2
        freqs = jnp.exp(
            -jnp.log(10000.0) * jnp.arange(half, dtype=jnp.float32) / max(half - 1, 1)
        )
        ang = t[:, None] * freqs[None, :]
        emb = jnp.concatenate([jnp.sin(ang), jnp.cos(ang)], axis=-1)
        if emb.shape[-1] < self.time_embed_dim:
            emb = jnp.pad(emb, ((0, 0), (0, self.time_embed_dim - emb.shape[-1])))
        return self.time_mlp(emb)

    def velocity(self, x_t: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        x_t = jnp.asarray(x_t, dtype=jnp.float32)
        if x_t.ndim == 1:
            x_t = x_t[None, :]
            squeeze = True
        else:
            squeeze = False
        t = jnp.asarray(t, dtype=jnp.float32)
        if t.ndim == 0:
            t = jnp.full((x_t.shape[0],), t)
        temb = self._time_embed(t)
        out = self.velocity_mlp(jnp.concatenate([x_t, temb], axis=-1))
        return out[0] if squeeze else out

    def decode(self, z: jnp.ndarray) -> jnp.ndarray:
        """Map noise z=x_0 to data x_1 via Euler ODE. Accepts (D,) or (B, D)."""
        z = jnp.asarray(z, dtype=jnp.float32)
        squeeze = z.ndim == 1
        if squeeze:
            z = z[None, :]
        dt = jnp.float32(1.0 / float(self.n_steps))

        def body(i, x):
            t = jnp.float32(i) * dt
            t_b = jnp.full((x.shape[0],), t)
            temb = self._time_embed(t_b)
            v = self.velocity_mlp(jnp.concatenate([x, temb], axis=-1))
            return x + dt * v

        x = jax.lax.fori_loop(0, int(self.n_steps), body, z)
        return x[0] if squeeze else x

    def __call__(self, x_t: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        return self.velocity(x_t, t)


def decode_flow_jax(params, z, mean, std, dim: int, hidden_dims=(256, 256, 256), n_steps: int = 10):
    """JAX decode for SAC updates: noise z -> unnormalized residual flat."""
    module = FlowMatchingMLP(dim=int(dim), hidden_dims=tuple(hidden_dims), n_steps=int(n_steps))
    recon = module.apply(
        {"params": jax.lax.stop_gradient(params)},
        z,
        method=module.decode,
    )
    return recon * std + mean
