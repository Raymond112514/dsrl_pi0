"""Latent residual RL models (PCA, VAE, flow).

New methods: add a module under latent_rl that subclasses LatentModel and
calls register("name"). The trainer only uses the helpers below.
"""

from __future__ import annotations

import importlib
import pkgutil
from typing import Any, Dict, Optional, Type

from jaxrl2.utils.general_utils import AttrDict

from latent_rl.base import LatentModel

_REGISTRY: Dict[str, Type[LatentModel]] = {}
_models_imported = False


def register(name: str):
    """Decorator: register a LatentModel under `--latent name`."""

    def deco(cls: Type[LatentModel]) -> Type[LatentModel]:
        if name in _REGISTRY and _REGISTRY[name] is not cls:
            raise ValueError(f"latent model {name!r} already registered")
        cls.name = name
        _REGISTRY[name] = cls
        return cls

    return deco


def _import_models() -> None:
    global _models_imported
    if _models_imported:
        return
    package = importlib.import_module(__name__)
    for mod in pkgutil.iter_modules(package.__path__):
        if mod.name in ("base", "dataset"):
            continue
        importlib.import_module(f"{__name__}.{mod.name}")
    _models_imported = True


def add_args(parser) -> None:
    """Register --latent and every model's flags."""
    _import_models()
    parser.add_argument(
        "--latent",
        default="none",
        choices=["none", *sorted(_REGISTRY.keys())],
        help="Latent residual model. none keeps the baseline residual SAC path.",
    )
    for cls in _REGISTRY.values():
        cls.add_args(parser)


def nest(variant: Any) -> Any:
    """Group pca_k -> variant.pca.k, vae_latent_dim -> variant.vae.latent_dim, etc."""
    _import_models()
    for name in _REGISTRY:
        nested = AttrDict()
        prefix = f"{name}_"
        for key in list(variant.keys()):
            if key.startswith(prefix):
                nested[key[len(prefix) :]] = variant[key]
        variant[name] = nested
    return variant


def action_dim(variant: Any) -> Optional[int]:
    """SAC action width, or None for baseline.

    The model sees the full variant so flow can use
    ``query_freq * env_action_dim``.
    """
    _import_models()
    kind = getattr(variant, "latent", "none")
    if kind in (None, "none", ""):
        return None
    if kind not in _REGISTRY:
        raise ValueError(f"Unknown --latent={kind!r}; known={sorted(_REGISTRY)}")
    return int(_REGISTRY[kind].action_dim(variant))


def build(variant: Any) -> Optional[LatentModel]:
    """Construct the selected model, or None for baseline.

    A non-empty ``<name>_path`` loads that checkpoint and marks the model fitted.
    Otherwise the model is unfitted until ``fit``. ``query_freq`` and
    ``env_action_dim`` are copied onto it when they are already set.
    """
    _import_models()
    kind = getattr(variant, "latent", "none")
    if kind in (None, "none", ""):
        return None
    if kind not in _REGISTRY:
        raise ValueError(f"Unknown --latent={kind!r}; known={sorted(_REGISTRY)}")
    cls = _REGISTRY[kind]
    hparams = getattr(variant, kind)
    path = str(getattr(hparams, "path", "") or "")
    if path:
        from pathlib import Path

        if not Path(path).is_file():
            raise FileNotFoundError(f"--{kind}_path file not found: {path}")
        model = cls.load(path)
    else:
        model = cls(hparams, seed=int(getattr(variant, "seed", 0)))
        query_freq = int(getattr(variant, "query_freq", 0) or 0)
        env_action_dim = int(getattr(variant, "env_action_dim", 0) or 0)
        if query_freq > 0 and env_action_dim > 0:
            model.configure(query_freq, env_action_dim)
    return model


__all__ = [
    "LatentModel",
    "register",
    "add_args",
    "nest",
    "action_dim",
    "build",
]

_import_models()
