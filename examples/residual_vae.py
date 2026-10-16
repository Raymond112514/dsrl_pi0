"""VAE on action chunks.

fit_residual=True (default) fits pairwise differences δ = a - a'.
fit_residual=False fits the action chunks.
Execution is always a_base + λ (std * decoder(z) + μ).

condition_on_state concatenates the query-time proprio vector into the
encoder and decoder. Those targets are already aligned with that state:
within-state chunk differences when fit_residual is set, otherwise the
chunk itself. Cross-state pairs are not conditioned on one state.
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
    fit_residual: bool = True
    decode_add_mean: bool = True
    condition_on_state: bool = False
    state_mean: Optional[np.ndarray] = None
    state_std: Optional[np.ndarray] = None
    fit_metrics: Optional[Dict[str, float]] = None

    @property
    def feature_dim(self) -> int:
        return int(self.mean.shape[0])

    @property
    def num_basis(self) -> int:
        return int(self.latent_dim)

    @property
    def cond_dim(self) -> int:
        if not self.condition_on_state:
            return 0
        return int(np.asarray(self.state_mean).shape[0])

    def _module(self) -> MLPAutoencoder:
        return MLPAutoencoder(
            input_dim=self.feature_dim,
            latent_dim=self.latent_dim,
            hidden_dims=self.hidden_dims,
            vae=True,
            cond_dim=self.cond_dim,
        )

    def _normalized_state(self, state, batch: int) -> np.ndarray:
        if not self.condition_on_state:
            raise ValueError("This VAE is unconditional")
        s = np.asarray(state, dtype=np.float32).reshape(batch, -1)
        if s.shape[-1] != self.cond_dim:
            raise ValueError(f"State dim {s.shape[-1]} != VAE cond_dim {self.cond_dim}")
        return (s - self.state_mean) / self.state_std

    def decode_latents(self, z, state=None) -> np.ndarray:
        """Map latents z -> flat residual: std * decoder(z), plus μ unless disabled.

        `state` is the query-time proprio vector when `condition_on_state` is set.
        """
        z = np.asarray(z, dtype=np.float32)
        single = z.ndim == 1
        if single:
            z = z.reshape(1, -1)
        module = self._module()
        if self.condition_on_state:
            cond = jnp.asarray(self._normalized_state(state, z.shape[0]))
            recon = module.apply({"params": self.params}, jnp.asarray(z), cond, method=module.decode)
        else:
            recon = module.apply({"params": self.params}, jnp.asarray(z), method=module.decode)
        recon = np.asarray(recon, dtype=np.float32)
        out = recon * self.std[None, :]
        if self.decode_add_mean:
            out = out + self.mean[None, :]
        return out[0] if single else out

    def latent_to_chunk(self, z, scale: float = 1.0, state=None) -> np.ndarray:
        """Map K SAC latents to a (query_freq, action_dim) chunk.

        The flat vector is std * decoder(z) + μ. `scale` is λ in
        a_base + λ * D(z). Fit target does not change this.
        """
        flat = self.decode_latents(z, state=state)
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
                    "fit_residual": self.fit_residual,
                    "decode_add_mean": self.decode_add_mean,
                    "condition_on_state": self.condition_on_state,
                    "state_mean": self.state_mean,
                    "state_std": self.state_std,
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
            "fit_residual": self.fit_residual,
            "decode_add_mean": self.decode_add_mean,
            "condition_on_state": self.condition_on_state,
            "state_dim": self.cond_dim,
            "fit_metrics": self.fit_metrics,
        }
        path.with_suffix(".json").write_text(json.dumps(meta, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> ResidualVAE:
        path = Path(path)
        with open(path, "rb") as f:
            data = pickle.load(f)
        return cls(
            params=data["params"],
            mean=np.asarray(data["mean"], dtype=np.float32),
            std=np.asarray(data["std"], dtype=np.float32),
            query_freq=int(data["query_freq"]),
            action_dim=int(data["action_dim"]),
            latent_dim=int(data["latent_dim"]),
            hidden_dims=tuple(data.get("hidden_dims", (256, 256))),
            fit_residual=_loaded_fit_residual(data),
            decode_add_mean=bool(data.get("decode_add_mean", True)),
            condition_on_state=bool(data.get("condition_on_state", False)),
            state_mean=_loaded_state_stat(data, "state_mean"),
            state_std=_loaded_state_stat(data, "state_std", ones=True),
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


def _loaded_state_stat(data: dict, key: str, ones: bool = False) -> np.ndarray:
    value = data.get(key)
    if value is None:
        return np.ones((0,), dtype=np.float32) if ones else np.zeros((0,), dtype=np.float32)
    return np.asarray(value, dtype=np.float32).reshape(-1)


def _loaded_fit_residual(data: dict) -> bool:
    if "fit_residual" in data:
        return bool(data["fit_residual"])
    if data.get("target") == "actions":
        return False
    return True


def prepare_fit_arrays(
    chunks: np.ndarray,
    *,
    fit_residual: bool,
    pairs_per_chunk: int,
    batch_size: int,
    val_frac: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, Optional[np.ndarray], Dict]:
    """Build the train/val arrays for a residual or full-action fit.

    `fit_residual=True` samples δ = a - a'. `fit_residual=False` uses the
    chunks themselves. μ and std are computed by the caller on the train split.
    """
    fit_residual = bool(fit_residual)
    data = np.asarray(chunks, dtype=np.float32)
    n = int(data.shape[0])
    num_pairs = None
    if not fit_residual:
        if n < 1:
            raise ValueError(f"Need >=1 action chunk to fit the action distribution, got {n}")
    else:
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
        "fit_residual": fit_residual,
        "num_chunks": float(n),
        "num_samples": float(len(data)),
    }
    if num_pairs is not None:
        info["num_pairs"] = float(num_pairs)
    return train, val, info


def _split_aligned(data, states, val_frac, rng):
    """Train/val split that keeps each row aligned with its state."""
    n = int(data.shape[0])
    n_val = max(1, int(round(val_frac * n))) if val_frac > 0 else 0
    n_val = min(n_val, n - 1) if n > 1 else 0
    perm = rng.permutation(n)
    val_idx = perm[:n_val]
    train_idx = perm[n_val:] if n_val > 0 else perm
    train = data[train_idx]
    val = data[val_idx] if n_val > 0 else None
    train_s = states[train_idx]
    val_s = states[val_idx] if n_val > 0 else None
    return train, val, train_s, val_s


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
    fit_residual: bool = True,
    states: Optional[np.ndarray] = None,
    condition_on_state: bool = False,
) -> ResidualVAE:
    """Fit a β-VAE on residuals (`fit_residual=True`) or action chunks.

    `condition_on_state` requires `states` with one row per chunk. Those rows
    are the fit targets already (within-state diffs, or the chunks themselves).
    This does not build cross-state pairs.
    """
    chunks = np.asarray(chunks, dtype=np.float32)
    if chunks.ndim != 2:
        raise ValueError(f"Expected (N, D) chunks, got {chunks.shape}")
    condition_on_state = bool(condition_on_state)
    if condition_on_state and states is None:
        raise ValueError("condition_on_state requires a state per chunk")
    if states is not None and not condition_on_state:
        raise ValueError("states were passed but condition_on_state is false")

    rng_np = np.random.default_rng(seed)
    train_s = val_s = None
    if condition_on_state:
        states = np.asarray(states, dtype=np.float32)
        if states.ndim == 1:
            states = states.reshape(-1, 1)
        if states.shape[0] != chunks.shape[0]:
            raise ValueError(
                f"Got {states.shape[0]} states for {chunks.shape[0]} chunks"
            )
        train, val, train_s, val_s = _split_aligned(chunks, states, val_frac, rng_np)
        split_info = {
            "fit_residual": bool(fit_residual),
            "num_chunks": float(chunks.shape[0]),
            "num_samples": float(chunks.shape[0]),
        }
    else:
        train, val, split_info = prepare_fit_arrays(
            chunks,
            fit_residual=fit_residual,
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

    state_mean = np.zeros((0,), dtype=np.float32)
    state_std = np.ones((0,), dtype=np.float32)
    train_s_n = val_s_n = None
    cond_dim = 0
    if condition_on_state:
        state_mean = train_s.mean(axis=0).astype(np.float32)
        state_std = np.maximum(train_s.std(axis=0).astype(np.float32), 1e-6)
        train_s_n = (train_s - state_mean) / state_std
        val_s_n = (val_s - state_mean) / state_std if val_s is not None else None
        cond_dim = int(state_mean.shape[0])

    input_dim = int(train_n.shape[1])
    hidden = tuple(int(h) for h in hidden_dims)
    module = MLPAutoencoder(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden,
        vae=True,
        cond_dim=cond_dim,
    )

    rng = jax.random.PRNGKey(seed)
    rng, init_rng = jax.random.split(rng)
    dummy = jnp.zeros((1, input_dim), dtype=jnp.float32)
    if cond_dim:
        params = module.init(
            init_rng, dummy, sample=False, cond=jnp.zeros((1, cond_dim), dtype=jnp.float32)
        )["params"]
    else:
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
    def train_step(state: TrainState, batch: jnp.ndarray, cond: jnp.ndarray, step_rng: jax.Array):
        def loss_fn(params):
            if cond_dim:
                recon, mu, logvar = state.apply_fn(
                    {"params": params}, batch, latent_rng=step_rng, sample=True, cond=cond
                )
            else:
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
    def eval_step(state: TrainState, batch: jnp.ndarray, cond: jnp.ndarray):
        if cond_dim:
            recon, mu, logvar = state.apply_fn({"params": state.params}, batch, sample=False, cond=cond)
        else:
            recon, mu, logvar = state.apply_fn({"params": state.params}, batch, sample=False)
        mse = jnp.mean((recon - batch) ** 2)
        kl = gaussian_kl(mu, logvar)
        metrics = _batch_metrics(batch, recon, mu, kl, mean_j, std_j)
        metrics["loss"] = mse + beta_f * kl
        return metrics

    def _iter_batches(data: np.ndarray, cond_data: Optional[np.ndarray], bs: int, shuffle: bool):
        idx = rng_np.permutation(len(data)) if shuffle else np.arange(len(data))
        for start in range(0, len(data), bs):
            sel = idx[start : start + bs]
            batch = jnp.asarray(data[sel])
            if cond_data is None:
                cond = jnp.zeros((batch.shape[0], 1), dtype=jnp.float32)
            else:
                cond = jnp.asarray(cond_data[sel])
            yield batch, cond

    def _mean_metrics(metric_list):
        if not metric_list:
            return {}
        keys = metric_list[0].keys()
        return {k: float(np.mean([float(np.asarray(m[k])) for m in metric_list])) for k in keys}

    last_train, last_val = {}, {}
    for epoch in range(1, epochs + 1):
        train_metrics = []
        for batch, cond in _iter_batches(train_n, train_s_n, batch_size, shuffle=True):
            rng, step_rng = jax.random.split(rng)
            state, metrics = train_step(state, batch, cond, step_rng)
            train_metrics.append(metrics)
        last_train = _mean_metrics(train_metrics)

        if val_n is not None and len(val_n) > 0:
            val_metrics = [
                eval_step(state, batch, cond)
                for batch, cond in _iter_batches(val_n, val_s_n, batch_size, shuffle=False)
            ]
            last_val = _mean_metrics(val_metrics)
        else:
            last_val = {}

        if epoch == 1 or epoch == epochs or epoch % max(1, epochs // 5) == 0:
            tag = "residual_vae" if split_info["fit_residual"] else "action_vae"
            if cond_dim:
                tag = f"{tag}+state"
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
        "condition_on_state": float(condition_on_state),
        "state_dim": float(cond_dim),
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
        fit_residual=bool(split_info["fit_residual"]),
        condition_on_state=condition_on_state,
        state_mean=state_mean,
        state_std=state_std,
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
    cond=None,
    cond_dim: int = 0,
    state_mean=None,
    state_std=None,
):
    """JAX decode for SAC updates: z -> unnormalized residual flat."""
    from utils.vae_model import decode_latents_jax as _decode
    return _decode(
        params, z, mean, std, input_dim, latent_dim, hidden_dims,
        cond=cond, cond_dim=cond_dim, state_mean=state_mean, state_std=state_std,
    )
