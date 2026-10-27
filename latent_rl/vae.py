"""VAE on pairwise action differences.

``fit`` samples δ = a - a' from the raw chunks. Decode is
std * decoder(z) + mean, without residual_scale.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState

from latent_rl import register
from latent_rl.base import LatentModel, flat_chunks
from latent_rl.dataset import loaded_fit_residual, prepare_fit_arrays

_FILENAME = "residual_vae.pkl"


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


def _gaussian_kl(mu, logvar) -> jnp.ndarray:
    return -0.5 * jnp.mean(
        jnp.sum(1.0 + logvar - mu**2 - jnp.exp(logvar), axis=-1)
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


@register("vae")
class VAELatentModel(LatentModel):
    def __init__(self, hparams: Any, seed: int = 0):
        self.latent_dim = int(getattr(hparams, "latent_dim", 8))
        self.beta = float(getattr(hparams, "beta", 1e-3))
        self.epochs = int(getattr(hparams, "epochs", 100))
        self.batch_size = int(getattr(hparams, "batch_size", 256))
        self.lr = float(getattr(hparams, "lr", 1e-3))
        self._hidden_dims = tuple(getattr(hparams, "hidden_dims", (256, 256)))
        self.seed = int(seed)
        self.query_freq: int | None = None
        self.env_action_dim: int | None = None
        self.params = None
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None
        self._fit_metrics: Optional[Dict[str, float]] = None
        self._fitted = False

    @classmethod
    def add_args(cls, parser) -> None:
        parser.add_argument("--vae_latent_dim", type=int, default=8, help="VAE latent width K.")
        parser.add_argument("--vae_beta", type=float, default=1e-3)
        parser.add_argument("--vae_epochs", type=int, default=100)
        parser.add_argument("--vae_batch_size", type=int, default=256)
        parser.add_argument("--vae_lr", type=float, default=1e-3)
        parser.add_argument(
            "--vae_path",
            default="",
            type=str,
            help="Optional pre-fit residual_vae.pkl.",
        )

    @classmethod
    def action_dim(cls, variant: Any) -> int:
        hparams = getattr(variant, "vae", None)
        k = getattr(hparams, "latent_dim", None) if hparams is not None else None
        if k is None:
            k = getattr(variant, "vae_latent_dim", 8)
        return int(k)

    def fit(self, base_chunks: np.ndarray) -> None:
        if self.query_freq is None or self.env_action_dim is None:
            raise RuntimeError("VAELatentModel.fit called before configure")
        chunks = flat_chunks(base_chunks).astype(np.float32)
        if chunks.ndim != 2:
            raise ValueError(f"Expected (N, D) chunks, got {chunks.shape}")

        rng_np = np.random.default_rng(self.seed)
        train, val, split_info = prepare_fit_arrays(
            chunks,
            pairs_per_chunk=8,
            batch_size=self.batch_size,
            val_frac=0.1,
            rng=rng_np,
        )
        n = int(split_info["num_chunks"])
        mean = train.mean(axis=0).astype(np.float32)
        std = np.maximum(train.std(axis=0).astype(np.float32), 1e-6)
        train_n = (train - mean) / std
        val_n = (val - mean) / std if val is not None else None

        input_dim = int(train_n.shape[1])
        hidden = tuple(int(h) for h in self._hidden_dims)
        module = MLPAutoencoder(
            input_dim=input_dim,
            latent_dim=self.latent_dim,
            hidden_dims=hidden,
            vae=True,
        )
        rng = jax.random.PRNGKey(self.seed)
        rng, init_rng = jax.random.split(rng)
        dummy = jnp.zeros((1, input_dim), dtype=jnp.float32)
        params = module.init(init_rng, dummy, sample=False)["params"]
        state = TrainState.create(
            apply_fn=module.apply,
            params=params,
            tx=optax.adam(self.lr),
        )
        mean_j = jnp.asarray(mean)
        std_j = jnp.asarray(std)
        beta_f = float(self.beta)

        @jax.jit
        def train_step(state: TrainState, batch: jnp.ndarray, step_rng: jax.Array):
            def loss_fn(params):
                recon, mu, logvar = state.apply_fn(
                    {"params": params}, batch, latent_rng=step_rng, sample=True
                )
                mse = jnp.mean((recon - batch) ** 2)
                kl = _gaussian_kl(mu, logvar)
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
            kl = _gaussian_kl(mu, logvar)
            metrics = _batch_metrics(batch, recon, mu, kl, mean_j, std_j)
            metrics["loss"] = mse + beta_f * kl
            return metrics

        def _iter_batches(data: np.ndarray, bs: int, shuffle: bool):
            idx = rng_np.permutation(len(data)) if shuffle else np.arange(len(data))
            for start in range(0, len(data), bs):
                yield jnp.asarray(data[idx[start : start + bs]])

        def _mean_metrics(metric_list):
            if not metric_list:
                return {}
            keys = metric_list[0].keys()
            return {
                k: float(np.mean([float(np.asarray(m[k])) for m in metric_list]))
                for k in keys
            }

        last_train, last_val = {}, {}
        for epoch in range(1, self.epochs + 1):
            train_metrics = []
            for batch in _iter_batches(train_n, self.batch_size, shuffle=True):
                rng, step_rng = jax.random.split(rng)
                state, metrics = train_step(state, batch, step_rng)
                train_metrics.append(metrics)
            last_train = _mean_metrics(train_metrics)
            if val_n is not None and len(val_n) > 0:
                last_val = _mean_metrics(
                    [eval_step(state, batch) for batch in _iter_batches(val_n, self.batch_size, False)]
                )
            else:
                last_val = {}
            if epoch == 1 or epoch == self.epochs or epoch % max(1, self.epochs // 5) == 0:
                print(
                    f"[residual_vae] epoch {epoch}/{self.epochs}  "
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
            "beta": float(self.beta),
            "epochs": float(self.epochs),
        }
        self._fit_metrics = {k: float(v) for k, v in fit_metrics.items() if v is not None}
        self.params = state.params
        self.mean = mean
        self.std = std
        self._hidden_dims = hidden
        self._fitted = True

    @property
    def feature_dim(self) -> int:
        return int(self._require_mean().shape[0])

    @property
    def hidden_dims(self):
        return tuple(self._hidden_dims)

    def fit_metrics(self) -> Optional[dict]:
        return self._fit_metrics

    def decode(self, z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, dtype=np.float32)
        single = z.ndim == 1
        if single:
            z = z.reshape(1, -1)
        module = MLPAutoencoder(
            input_dim=self.feature_dim,
            latent_dim=self.latent_dim,
            hidden_dims=self.hidden_dims,
            vae=True,
        )
        recon = module.apply({"params": self.params}, jnp.asarray(z), method=module.decode)
        recon = np.asarray(recon, dtype=np.float32)
        out = recon * self.std[None, :] + self.mean[None, :]
        flat = out[0] if single else out
        if single:
            return flat.reshape(self.query_freq, self.env_action_dim).astype(np.float32)
        return flat.reshape(-1, self.query_freq, self.env_action_dim).astype(np.float32)

    def jax_decode(self, z):
        latents = jnp.asarray(z, dtype=jnp.float32)
        if latents.ndim == 1:
            latents = latents[None, :]
        return decode_latents_jax(
            self.params,
            latents,
            jnp.asarray(self._require_mean()),
            jnp.asarray(self._require_std()),
            int(self.feature_dim),
            int(self.latent_dim),
            tuple(self.hidden_dims),
        )

    def make_jax_decoder(self):
        params = self.params
        mean = jnp.asarray(self._require_mean())
        std = jnp.asarray(self._require_std())
        input_dim = int(self.feature_dim)
        latent_dim = int(self.latent_dim)
        hidden_dims = tuple(self.hidden_dims)

        def decode_flat(z):
            latents = jnp.asarray(z, dtype=jnp.float32)
            if latents.ndim == 1:
                latents = latents[None, :]
            return decode_latents_jax(
                params, latents, mean, std, input_dim, latent_dim, hidden_dims
            )

        return decode_flat

    def save(self, output_dir: str) -> str:
        path = Path(output_dir) / _FILENAME
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "params": self.params,
                    "mean": self.mean,
                    "std": self.std,
                    "query_freq": self.query_freq,
                    "action_dim": self.env_action_dim,
                    "latent_dim": self.latent_dim,
                    "hidden_dims": self.hidden_dims,
                    "fit_residual": True,
                    "decode_add_mean": True,
                    "condition_on_state": False,
                    "fit_metrics": self._fit_metrics,
                },
                f,
            )
        meta = {
            "query_freq": self.query_freq,
            "action_dim": self.env_action_dim,
            "latent_dim": self.latent_dim,
            "hidden_dims": list(self.hidden_dims),
            "feature_dim": self.feature_dim,
            "fit_residual": True,
            "decode_add_mean": True,
            "condition_on_state": False,
            "fit_metrics": self._fit_metrics,
        }
        path.with_suffix(".json").write_text(json.dumps(meta, indent=2))
        return str(path)

    @classmethod
    def load(cls, path: str) -> "VAELatentModel":
        path = Path(path)
        with open(path, "rb") as f:
            data = pickle.load(f)
        if not loaded_fit_residual(data):
            raise ValueError(f"{path} was fit on action chunks, not pairwise differences")
        if not bool(data.get("decode_add_mean", True)):
            raise ValueError(f"{path} was saved with the mean dropped at decode")
        if bool(data.get("condition_on_state", False)):
            raise ValueError(f"{path} is a state-conditioned VAE")
        obj = cls(
            type("H", (), {
                "latent_dim": int(data["latent_dim"]),
                "hidden_dims": tuple(data.get("hidden_dims", (256, 256))),
            })(),
            seed=0,
        )
        obj.configure(int(data["query_freq"]), int(data["action_dim"]))
        obj.params = data["params"]
        obj.mean = np.asarray(data["mean"], dtype=np.float32)
        obj.std = np.asarray(data["std"], dtype=np.float32)
        obj._fit_metrics = data.get("fit_metrics")
        obj._fitted = True
        return obj

    def _require_mean(self) -> np.ndarray:
        if not self._fitted or self.mean is None:
            raise RuntimeError("VAELatentModel used before fit")
        return self.mean

    def _require_std(self) -> np.ndarray:
        if not self._fitted or self.std is None:
            raise RuntimeError("VAELatentModel used before fit")
        return self.std
