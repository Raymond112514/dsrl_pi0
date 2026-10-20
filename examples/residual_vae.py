"""VAE on pairwise action-chunk differences.

Fits δ = a - a', with a and a' drawn from the warmup chunks. Those chunks
can come from different states. Training uses (δ - mean) / std.
Execution is a_base + λ (std * decoder(z) + mean).
"""

from __future__ import annotations

import json
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from utils.vae_model import MLPAutoencoder


def gaussian_kl(mu, logvar) -> jnp.ndarray:
    return -0.5 * jnp.mean(
        jnp.sum(1.0 + logvar - mu**2 - jnp.exp(logvar), axis=-1)
    )


def sample_pairwise_diffs(
    chunks: np.ndarray,
    num_pairs: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample δ = a - a' from (possibly different) chunk indices."""
    data = np.asarray(chunks, dtype=np.float32)
    n = data.shape[0]
    if n < 2:
        raise ValueError(f"Need >=2 action chunks for VAE diffs, got {n}")
    i = rng.integers(0, n, size=num_pairs)
    j = rng.integers(0, n, size=num_pairs)
    same = i == j
    while np.any(same):
        j[same] = rng.integers(0, n, size=int(same.sum()))
        same = i == j
    return data[i] - data[j]


@dataclass
class ResidualVAE:
    """Frozen VAE decoder for K-dim residual latents."""

    params: dict
    mean: np.ndarray  # (D,)
    std: np.ndarray  # (D,)
    query_freq: int
    action_dim: int
    latent_dim: int
    hidden_dims: Tuple[int, ...] = (256, 256)
    fit_metrics: Optional[Dict[str, float]] = None

    @property
    def feature_dim(self) -> int:
        return int(self.mean.shape[0])

    @property
    def num_basis(self) -> int:
        return int(self.latent_dim)

    def _module(self) -> MLPAutoencoder:
        return MLPAutoencoder(
            input_dim=self.feature_dim,
            latent_dim=self.latent_dim,
            hidden_dims=self.hidden_dims,
            vae=True,
        )

    def decode_latents(self, z) -> np.ndarray:
        """Map latents z -> flat residual std * decoder(z) + mean."""
        z = np.asarray(z, dtype=np.float32)
        single = z.ndim == 1
        if single:
            z = z.reshape(1, -1)
        module = self._module()
        recon = module.apply({"params": self.params}, jnp.asarray(z), method=module.decode)
        recon = np.asarray(recon, dtype=np.float32)
        out = recon * self.std[None, :] + self.mean[None, :]
        return out[0] if single else out

    def latent_to_chunk(self, z, scale: float = 1.0) -> np.ndarray:
        """Map K SAC latents to a (query_freq, action_dim) chunk.

        The flat vector is std * decoder(z) + mean. `scale` is λ in
        a_base + λ * D(z).
        """
        flat = self.decode_latents(z)
        return (scale * flat).reshape(self.query_freq, self.action_dim).astype(np.float32)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "params": self.params,
                    "mean": self.mean,
                    "std": self.std,
                    "query_freq": self.query_freq,
                    "action_dim": self.action_dim,
                    "latent_dim": self.latent_dim,
                    "hidden_dims": self.hidden_dims,
                    "fit_residual": True,
                    "decode_add_mean": True,
                    "condition_on_state": False,
                    "fit_metrics": self.fit_metrics,
                },
                f,
            )
        meta = {
            "query_freq": self.query_freq,
            "action_dim": self.action_dim,
            "latent_dim": self.latent_dim,
            "hidden_dims": list(self.hidden_dims),
            "feature_dim": self.feature_dim,
            "fit_residual": True,
            "decode_add_mean": True,
            "condition_on_state": False,
            "fit_metrics": self.fit_metrics,
        }
        path.with_suffix(".json").write_text(json.dumps(meta, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> ResidualVAE:
        path = Path(path)
        with open(path, "rb") as f:
            data = pickle.load(f)
        if not _loaded_fit_residual(data):
            raise ValueError(f"{path} was fit on action chunks, not pairwise differences")
        if not bool(data.get("decode_add_mean", True)):
            raise ValueError(f"{path} was saved with the mean dropped at decode")
        if bool(data.get("condition_on_state", False)):
            raise ValueError(f"{path} is a state-conditioned VAE")
        return cls(
            params=data["params"],
            mean=np.asarray(data["mean"], dtype=np.float32),
            std=np.asarray(data["std"], dtype=np.float32),
            query_freq=int(data["query_freq"]),
            action_dim=int(data["action_dim"]),
            latent_dim=int(data["latent_dim"]),
            hidden_dims=tuple(data.get("hidden_dims", (256, 256))),
            fit_metrics=data.get("fit_metrics"),
        )


def _batch_metrics(x, recon, mu, kl, mean, std) -> Dict[str, jnp.ndarray]:
    x_raw = x * std + mean
    recon_raw = recon * std + mean
    err = recon_raw - x_raw
    recon_loss = jnp.mean(err**2)
    var_x = jnp.var(x_raw)
    return {
        "loss": recon_loss,
        "recon_loss": recon_loss,
        "latent_std": jnp.mean(jnp.std(mu, axis=0)),
        "explained_var": 1.0 - jnp.var(err) / jnp.maximum(var_x, 1e-12),
        "kl": kl,
    }


def _loaded_fit_residual(data: dict) -> bool:
    if "fit_residual" in data:
        return bool(data["fit_residual"])
    if data.get("target") == "actions":
        return False
    return True


def prepare_fit_arrays(
    chunks: np.ndarray,
    *,
    pairs_per_chunk: int,
    batch_size: int,
    val_frac: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, Optional[np.ndarray], Dict]:
    """Sample δ = a - a' and split into train and val.

    μ and std are computed by the caller on the train split.
    """
    data = np.asarray(chunks, dtype=np.float32)
    n = int(data.shape[0])
    if n < 2:
        raise ValueError(f"Need >=2 action chunks for pairwise diffs, got {n}")
    num_pairs = max(int(pairs_per_chunk) * n, int(batch_size))
    data = sample_pairwise_diffs(data, num_pairs, rng)

    n_val = max(1, int(round(val_frac * len(data)))) if val_frac > 0 else 0
    n_val = min(n_val, len(data) - 1) if len(data) > 1 else 0
    perm = rng.permutation(len(data))
    val_idx = perm[:n_val]
    train_idx = perm[n_val:] if n_val > 0 else perm
    train = data[train_idx]
    val = data[val_idx] if n_val > 0 else None
    info = {
        "num_chunks": float(n),
        "num_samples": float(len(data)),
    }
    if num_pairs is not None:
        info["num_pairs"] = float(num_pairs)
    return train, val, info


def fit_residual_vae(
    chunks: np.ndarray,
    *,
    latent_dim: int,
    query_freq: int,
    action_dim: int,
    beta: float = 1e-3,
    epochs: int = 100,
    batch_size: int = 256,
    lr: float = 1e-3,
    pairs_per_chunk: int = 8,
    val_frac: float = 0.1,
    hidden_dims: Sequence[int] = (256, 256),
    seed: int = 0,
    normalize: bool = True,
) -> ResidualVAE:
    """Fit a β-VAE on pairwise differences δ = a - a'."""
    chunks = np.asarray(chunks, dtype=np.float32)
    if chunks.ndim != 2:
        raise ValueError(f"Expected (N, D) chunks, got {chunks.shape}")

    rng_np = np.random.default_rng(seed)
    train, val, split_info = prepare_fit_arrays(
        chunks,
        pairs_per_chunk=pairs_per_chunk,
        batch_size=batch_size,
        val_frac=val_frac,
        rng=rng_np,
    )
    n = int(split_info["num_chunks"])

    if normalize:
        mean = train.mean(axis=0).astype(np.float32)
        std = train.std(axis=0).astype(np.float32)
        std = np.maximum(std, 1e-6)
    else:
        mean = np.zeros(train.shape[1], dtype=np.float32)
        std = np.ones(train.shape[1], dtype=np.float32)

    train_n = (train - mean) / std
    val_n = (val - mean) / std if val is not None else None

    input_dim = int(train_n.shape[1])
    hidden = tuple(int(h) for h in hidden_dims)
    module = MLPAutoencoder(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden,
        vae=True,
    )

    rng = jax.random.PRNGKey(seed)
    rng, init_rng = jax.random.split(rng)
    dummy = jnp.zeros((1, input_dim), dtype=jnp.float32)
    params = module.init(init_rng, dummy, sample=False)["params"]
    state = TrainState.create(
        apply_fn=module.apply,
        params=params,
        tx=optax.adam(lr),
    )

    mean_j = jnp.asarray(mean)
    std_j = jnp.asarray(std)
    beta_f = float(beta)

    @jax.jit
    def train_step(state: TrainState, batch: jnp.ndarray, step_rng: jax.Array):
        def loss_fn(params):
            recon, mu, logvar = state.apply_fn(
                {"params": params}, batch, latent_rng=step_rng, sample=True
            )
            mse = jnp.mean((recon - batch) ** 2)
            kl = gaussian_kl(mu, logvar)
            return mse + beta_f * kl, (recon, mu, kl)

        (loss, (recon, mu, kl)), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            state.params
        )
        state = state.apply_gradients(grads=grads)
        metrics = _batch_metrics(batch, recon, mu, kl, mean_j, std_j)
        metrics["loss"] = loss
        return state, metrics

    @jax.jit
    def eval_step(state: TrainState, batch: jnp.ndarray):
        recon, mu, logvar = state.apply_fn({"params": state.params}, batch, sample=False)
        mse = jnp.mean((recon - batch) ** 2)
        kl = gaussian_kl(mu, logvar)
        metrics = _batch_metrics(batch, recon, mu, kl, mean_j, std_j)
        metrics["loss"] = mse + beta_f * kl
        return metrics

    def _iter_batches(data: np.ndarray, bs: int, shuffle: bool):
        idx = rng_np.permutation(len(data)) if shuffle else np.arange(len(data))
        for start in range(0, len(data), bs):
            sel = idx[start : start + bs]
            yield jnp.asarray(data[sel])

    def _mean_metrics(metric_list):
        if not metric_list:
            return {}
        keys = metric_list[0].keys()
        return {k: float(np.mean([float(np.asarray(m[k])) for m in metric_list])) for k in keys}

    last_train, last_val = {}, {}
    for epoch in range(1, epochs + 1):
        train_metrics = []
        for batch in _iter_batches(train_n, batch_size, shuffle=True):
            rng, step_rng = jax.random.split(rng)
            state, metrics = train_step(state, batch, step_rng)
            train_metrics.append(metrics)
        last_train = _mean_metrics(train_metrics)

        if val_n is not None and len(val_n) > 0:
            val_metrics = [
                eval_step(state, batch)
                for batch in _iter_batches(val_n, batch_size, shuffle=False)
            ]
            last_val = _mean_metrics(val_metrics)
        else:
            last_val = {}

        if epoch == 1 or epoch == epochs or epoch % max(1, epochs // 5) == 0:
            tag = "residual_vae"
            print(
                f"[{tag}] epoch {epoch}/{epochs}  "
                f"train_loss={last_train.get('loss', float('nan')):.5f}  "
                f"val_recon={last_val.get('recon_loss', float('nan')):.5f}  "
                f"val_kl={last_val.get('kl', float('nan')):.4f}  "
                f"val_ev={last_val.get('explained_var', float('nan')):.4f}"
            )

    fit_metrics = {
        "train_loss": last_train.get("loss"),
        "train_recon_loss": last_train.get("recon_loss"),
        "train_kl": last_train.get("kl"),
        "train_explained_var": last_train.get("explained_var"),
        "val_loss": last_val.get("loss"),
        "val_recon_loss": last_val.get("recon_loss"),
        "val_kl": last_val.get("kl"),
        "val_explained_var": last_val.get("explained_var"),
        "num_chunks": float(n),
        "num_samples": float(split_info["num_samples"]),
        "num_pairs": split_info.get("num_pairs"),
        "beta": float(beta),
        "epochs": float(epochs),
    }
    # Drop Nones for JSON friendliness
    fit_metrics = {k: float(v) for k, v in fit_metrics.items() if v is not None}

    return ResidualVAE(
        params=state.params,
        mean=mean,
        std=std,
        query_freq=query_freq,
        action_dim=action_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden,
        fit_metrics=fit_metrics,
    )


def decode_latents_jax(
    params,
    z,
    mean,
    std,
    input_dim: int,
    latent_dim: int,
    hidden_dims=(256, 256),
):
    """JAX decode for SAC updates: z -> unnormalized residual flat."""
    from utils.vae_model import decode_latents_jax as _decode
    return _decode(params, z, mean, std, input_dim, latent_dim, hidden_dims)
