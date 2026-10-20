"""Flow-matching on pairwise action-chunk differences.

Fits δ = a - a'. Training uses (δ - mean) / std.
Execution is a_base + λ (std * ODE(z) + mean).
"""

from __future__ import annotations

import json
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from examples.residual_vae import _loaded_fit_residual, prepare_fit_arrays
from utils.flow_model import FlowMatchingMLP


@dataclass
class ResidualFlow:
    """Frozen flow decoder. Latent dim equals the residual chunk dim."""

    params: dict
    mean: np.ndarray  # (D,)
    std: np.ndarray  # (D,)
    query_freq: int
    action_dim: int
    hidden_dims: tuple = (256, 256, 256)
    n_steps: int = 10
    fit_metrics: Optional[Dict[str, float]] = None

    @property
    def feature_dim(self) -> int:
        return int(self.mean.shape[0])

    @property
    def latent_dim(self) -> int:
        return self.feature_dim

    @property
    def num_basis(self) -> int:
        return self.feature_dim

    def _module(self) -> FlowMatchingMLP:
        return FlowMatchingMLP(
            dim=self.feature_dim,
            hidden_dims=self.hidden_dims,
            n_steps=int(self.n_steps),
        )

    def decode_latents(self, z) -> np.ndarray:
        """Map noise z to a flat residual: std * ODE(z) + mean."""
        z = np.asarray(z, dtype=np.float32)
        single = z.ndim == 1
        if single:
            z = z.reshape(1, -1)
        module = self._module()
        recon = module.apply({"params": self.params}, z, method=module.decode)
        recon = np.asarray(recon, dtype=np.float32)
        out = recon * self.std[None, :] + self.mean[None, :]
        return out[0] if single else out

    def latent_to_chunk(self, z, scale: float = 1.0) -> np.ndarray:
        """Map a full-dim SAC latent to a (query_freq, action_dim) chunk.

        The flat vector is std * ODE(z) + μ. `scale` is λ in
        a_base + λ * D(z). Fit target does not change this.
        """
        flat = self.decode_latents(z)
        return (scale * flat).reshape(self.query_freq, self.action_dim).astype(np.float32)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        params = jax.tree_util.tree_map(
            lambda x: np.asarray(jax.device_get(x)),
            self.params,
        )
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "params": params,
                    "mean": self.mean,
                    "std": self.std,
                    "query_freq": self.query_freq,
                    "action_dim": self.action_dim,
                    "hidden_dims": self.hidden_dims,
                    "n_steps": self.n_steps,
                    "fit_residual": True,
                    "decode_add_mean": True,
                    "fit_metrics": self.fit_metrics,
                },
                f,
            )
        meta = {
            "query_freq": self.query_freq,
            "action_dim": self.action_dim,
            "latent_dim": self.latent_dim,
            "hidden_dims": list(self.hidden_dims),
            "n_steps": int(self.n_steps),
            "feature_dim": self.feature_dim,
            "fit_residual": True,
            "decode_add_mean": True,
            "fit_metrics": self.fit_metrics,
            "kind": "flow",
        }
        path.with_suffix(".json").write_text(json.dumps(meta, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> ResidualFlow:
        path = Path(path)
        with open(path, "rb") as f:
            data = pickle.load(f)
        if not _loaded_fit_residual(data):
            raise ValueError(f"{path} was fit on action chunks, not pairwise differences")
        if not bool(data.get("decode_add_mean", True)):
            raise ValueError(f"{path} was saved with the mean dropped at decode")
        return cls(
            params=data["params"],
            mean=np.asarray(data["mean"], dtype=np.float32),
            std=np.asarray(data["std"], dtype=np.float32),
            query_freq=int(data["query_freq"]),
            action_dim=int(data["action_dim"]),
            hidden_dims=tuple(data.get("hidden_dims", (256, 256, 256))),
            n_steps=int(data.get("n_steps", 10)),
            fit_metrics=data.get("fit_metrics"),
        )


def fit_residual_flow(
    chunks: np.ndarray,
    *,
    query_freq: int,
    action_dim: int,
    epochs: int = 100,
    batch_size: int = 256,
    lr: float = 1e-3,
    pairs_per_chunk: int = 8,
    val_frac: float = 0.1,
    hidden_dims: Sequence[int] = (256, 256, 256),
    n_steps: int = 10,
    seed: int = 0,
    normalize: bool = True,
) -> ResidualFlow:
    """Fit OT-CFM on pairwise differences δ = a - a'."""
    chunks = np.asarray(chunks, dtype=np.float32)
    if chunks.ndim != 2:
        raise ValueError(f"Expected (N, D) chunks, got {chunks.shape}")
    expected = int(query_freq) * int(action_dim)
    if chunks.shape[1] != expected:
        raise ValueError(
            f"Chunk dim {chunks.shape[1]} != query_freq * action_dim = {expected}"
        )

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

    train_n = ((train - mean) / std).astype(np.float32)
    val_n = ((val - mean) / std).astype(np.float32) if val is not None else None

    dim = int(train_n.shape[1])
    hidden = tuple(int(h) for h in hidden_dims)
    n_steps = int(n_steps)
    module = FlowMatchingMLP(dim=dim, hidden_dims=hidden, n_steps=n_steps)

    rng = jax.random.PRNGKey(seed)
    rng, init_rng = jax.random.split(rng)
    params = module.init(
        init_rng,
        jnp.zeros((1, dim), dtype=jnp.float32),
        jnp.zeros((1,), dtype=jnp.float32),
    )["params"]
    state = TrainState.create(
        apply_fn=module.apply,
        params=params,
        tx=optax.adam(lr),
    )

    mean_j = jnp.asarray(mean)
    std_j = jnp.asarray(std)

    @jax.jit
    def train_step(state: TrainState, batch: jnp.ndarray, step_rng: jax.Array):
        eps_rng, t_rng = jax.random.split(step_rng)

        def loss_fn(params):
            eps = jax.random.normal(eps_rng, batch.shape)
            t = jax.random.uniform(t_rng, (batch.shape[0],), dtype=jnp.float32)
            t_ = t[:, None]
            x_t = (1.0 - t_) * eps + t_ * batch
            u = batch - eps
            v = state.apply_fn({"params": params}, x_t, t)
            return jnp.mean((v - u) ** 2)

        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        state = state.apply_gradients(grads=grads)
        return state, {"loss": loss}

    @jax.jit
    def eval_step(state: TrainState, batch: jnp.ndarray, step_rng: jax.Array):
        eps_rng, t_rng = jax.random.split(step_rng)
        eps = jax.random.normal(eps_rng, batch.shape)
        t = jax.random.uniform(t_rng, (batch.shape[0],), dtype=jnp.float32)
        t_ = t[:, None]
        x_t = (1.0 - t_) * eps + t_ * batch
        u = batch - eps
        v = state.apply_fn({"params": state.params}, x_t, t)
        loss = jnp.mean((v - u) ** 2)
        recon = state.apply_fn({"params": state.params}, eps, method=module.decode)
        recon_raw = recon * std_j + mean_j
        x_raw = batch * std_j + mean_j
        err = recon_raw - x_raw
        recon_loss = jnp.mean(err ** 2)
        var_x = jnp.var(x_raw)
        explained = 1.0 - jnp.var(err) / jnp.maximum(var_x, 1e-12)
        return {"loss": loss, "recon_loss": recon_loss, "explained_var": explained}

    def _iter_batches(data: np.ndarray, bs: int, shuffle: bool):
        idx = rng_np.permutation(len(data)) if shuffle else np.arange(len(data))
        for start in range(0, len(data), bs):
            yield jnp.asarray(data[idx[start : start + bs]])

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
            val_metrics = []
            for batch in _iter_batches(val_n, batch_size, shuffle=False):
                rng, step_rng = jax.random.split(rng)
                val_metrics.append(eval_step(state, batch, step_rng))
            last_val = _mean_metrics(val_metrics)
        else:
            last_val = {}

        if epoch == 1 or epoch == epochs or epoch % max(1, epochs // 5) == 0:
            tag = "residual_flow"
            print(
                f"[{tag}] epoch {epoch}/{epochs}  "
                f"train_loss={last_train.get('loss', float('nan')):.5f}  "
                f"val_loss={last_val.get('loss', float('nan')):.5f}  "
                f"val_recon={last_val.get('recon_loss', float('nan')):.5f}  "
                f"val_ev={last_val.get('explained_var', float('nan')):.4f}"
            )

    fit_metrics = {
        "train_loss": last_train.get("loss"),
        "val_loss": last_val.get("loss"),
        "val_recon_loss": last_val.get("recon_loss"),
        "val_explained_var": last_val.get("explained_var"),
        "num_chunks": float(n),
        "num_samples": float(split_info["num_samples"]),
        "num_pairs": split_info.get("num_pairs"),
        "epochs": float(epochs),
        "n_steps": float(n_steps),
        "latent_dim": float(dim),
    }
    fit_metrics = {k: float(v) for k, v in fit_metrics.items() if v is not None}

    return ResidualFlow(
        params=state.params,
        mean=mean,
        std=std,
        query_freq=int(query_freq),
        action_dim=int(action_dim),
        hidden_dims=hidden,
        n_steps=n_steps,
        fit_metrics=fit_metrics,
    )
