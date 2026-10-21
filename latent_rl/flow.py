"""Flow matching on pairwise action differences.

``fit`` samples δ = a - a' from the raw chunks. Decode is
std * ODE(z) + mean, without residual_scale. SAC width is the full chunk.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Dict, Optional

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState

from latent_rl import register
from latent_rl.base import LatentModel, flat_chunks
from latent_rl.dataset import loaded_fit_residual, prepare_fit_arrays
from utils.flow_model import FlowMatchingMLP, decode_flow_jax

_FILENAME = "residual_flow.pkl"


@register("flow")
class FlowLatentModel(LatentModel):
    def __init__(self, hparams: Any, seed: int = 0):
        self.epochs = int(getattr(hparams, "epochs", 100))
        self.batch_size = int(getattr(hparams, "batch_size", 256))
        self.lr = float(getattr(hparams, "lr", 1e-3))
        self._n_steps = int(getattr(hparams, "steps", 10))
        self._hidden_dims = tuple(getattr(hparams, "hidden_dims", (256, 256, 256)))
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
        parser.add_argument("--flow_epochs", type=int, default=100)
        parser.add_argument("--flow_batch_size", type=int, default=256)
        parser.add_argument("--flow_lr", type=float, default=1e-3)
        parser.add_argument("--flow_steps", type=int, default=10, help="Euler steps at decode time.")
        parser.add_argument(
            "--flow_path",
            default="",
            type=str,
            help="Optional pre-fit residual_flow.pkl.",
        )

    @classmethod
    def action_dim(cls, variant: Any) -> int:
        query_freq = int(getattr(variant, "query_freq", 0) or 0)
        action_dim = int(getattr(variant, "env_action_dim", 0) or 0)
        if query_freq < 1 or action_dim < 1:
            raise ValueError("flow action_dim needs variant.query_freq and variant.env_action_dim")
        return query_freq * action_dim

    def fit(self, base_chunks: np.ndarray) -> None:
        if self.query_freq is None or self.env_action_dim is None:
            raise RuntimeError("FlowLatentModel.fit called before configure")
        chunks = flat_chunks(base_chunks).astype(np.float32)
        if chunks.ndim != 2:
            raise ValueError(f"Expected (N, D) chunks, got {chunks.shape}")
        expected = int(self.query_freq) * int(self.env_action_dim)
        if chunks.shape[1] != expected:
            raise ValueError(
                f"Chunk dim {chunks.shape[1]} != query_freq * action_dim = {expected}"
            )

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
        train_n = ((train - mean) / std).astype(np.float32)
        val_n = ((val - mean) / std).astype(np.float32) if val is not None else None

        dim = int(train_n.shape[1])
        hidden = tuple(int(h) for h in self._hidden_dims)
        n_steps = int(self._n_steps)
        module = FlowMatchingMLP(dim=dim, hidden_dims=hidden, n_steps=n_steps)
        rng = jax.random.PRNGKey(self.seed)
        rng, init_rng = jax.random.split(rng)
        params = module.init(
            init_rng,
            jnp.zeros((1, dim), dtype=jnp.float32),
            jnp.zeros((1,), dtype=jnp.float32),
        )["params"]
        state = TrainState.create(
            apply_fn=module.apply,
            params=params,
            tx=optax.adam(self.lr),
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
                val_metrics = []
                for batch in _iter_batches(val_n, self.batch_size, shuffle=False):
                    rng, step_rng = jax.random.split(rng)
                    val_metrics.append(eval_step(state, batch, step_rng))
                last_val = _mean_metrics(val_metrics)
            else:
                last_val = {}
            if epoch == 1 or epoch == self.epochs or epoch % max(1, self.epochs // 5) == 0:
                print(
                    f"[residual_flow] epoch {epoch}/{self.epochs}  "
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
            "epochs": float(self.epochs),
            "n_steps": float(n_steps),
            "latent_dim": float(dim),
        }
        self._fit_metrics = {k: float(v) for k, v in fit_metrics.items() if v is not None}
        self.params = state.params
        self.mean = mean
        self.std = std
        self._hidden_dims = hidden
        self._n_steps = n_steps
        self._fitted = True

    @property
    def feature_dim(self) -> int:
        return int(self._require_mean().shape[0])

    @property
    def latent_dim(self) -> int:
        return self.feature_dim

    @property
    def hidden_dims(self):
        return tuple(self._hidden_dims)

    @property
    def n_steps(self) -> int:
        return int(self._n_steps)

    def fit_metrics(self) -> Optional[dict]:
        return self._fit_metrics

    def decode(self, z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, dtype=np.float32)
        single = z.ndim == 1
        if single:
            z = z.reshape(1, -1)
        module = FlowMatchingMLP(
            dim=self.feature_dim,
            hidden_dims=self.hidden_dims,
            n_steps=int(self.n_steps),
        )
        recon = module.apply({"params": self.params}, z, method=module.decode)
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
        return decode_flow_jax(
            self.params,
            latents,
            jnp.asarray(self._require_mean()),
            jnp.asarray(self._require_std()),
            int(self.feature_dim),
            tuple(self.hidden_dims),
            int(self.n_steps),
        )

    def make_jax_decoder(self):
        params = jax.tree_util.tree_map(lambda x: jnp.asarray(x), self.params)
        mean = jnp.asarray(self._require_mean())
        std = jnp.asarray(self._require_std())
        dim = int(self.feature_dim)
        hidden_dims = tuple(self.hidden_dims)
        n_steps = int(self.n_steps)

        def decode_flat(z):
            latents = jnp.asarray(z, dtype=jnp.float32)
            if latents.ndim == 1:
                latents = latents[None, :]
            return decode_flow_jax(params, latents, mean, std, dim, hidden_dims, n_steps)

        return decode_flat

    def save(self, output_dir: str) -> str:
        path = Path(output_dir) / _FILENAME
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
                    "action_dim": self.env_action_dim,
                    "hidden_dims": self.hidden_dims,
                    "n_steps": self.n_steps,
                    "fit_residual": True,
                    "decode_add_mean": True,
                    "fit_metrics": self._fit_metrics,
                },
                f,
            )
        meta = {
            "query_freq": self.query_freq,
            "action_dim": self.env_action_dim,
            "latent_dim": self.latent_dim,
            "hidden_dims": list(self.hidden_dims),
            "n_steps": int(self.n_steps),
            "feature_dim": self.feature_dim,
            "fit_residual": True,
            "decode_add_mean": True,
            "fit_metrics": self._fit_metrics,
            "kind": "flow",
        }
        path.with_suffix(".json").write_text(json.dumps(meta, indent=2))
        return str(path)

    @classmethod
    def load(cls, path: str) -> "FlowLatentModel":
        path = Path(path)
        with open(path, "rb") as f:
            data = pickle.load(f)
        if not loaded_fit_residual(data):
            raise ValueError(f"{path} was fit on action chunks, not pairwise differences")
        if not bool(data.get("decode_add_mean", True)):
            raise ValueError(f"{path} was saved with the mean dropped at decode")
        obj = cls(
            type("H", (), {
                "steps": int(data.get("n_steps", 10)),
                "hidden_dims": tuple(data.get("hidden_dims", (256, 256, 256))),
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
            raise RuntimeError("FlowLatentModel used before fit")
        return self.mean

    def _require_std(self) -> np.ndarray:
        if not self._fitted or self.std is None:
            raise RuntimeError("FlowLatentModel used before fit")
        return self.std
