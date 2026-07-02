import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA


def fit_action_basis(data, n_components):
    """Fit PCA on (N, D) actions. Returns V (D, k), per-PC explained variance."""
    centered = data - data.mean(axis=0)
    k = min(n_components, centered.shape[0], centered.shape[1])
    pca = PCA(n_components=k)
    pca.fit(centered)
    return pca.components_.T, pca.explained_variance_ratio_


def fit_action_basis_for_variance(data, variance_threshold=0.99):
    """Fit PCA and keep top-K components until cumulative variance >= threshold."""
    data = np.asarray(data, dtype=np.float64)
    if data.shape[0] < 2:
        raise ValueError(f"Need >=2 action samples for PCA, got {data.shape[0]}")
    mean = data.mean(axis=0)
    centered = data - mean
    pca = PCA()
    pca.fit(centered)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    k = int(np.searchsorted(cumvar, variance_threshold) + 1)
    k = min(k, centered.shape[0], centered.shape[1])
    explained = pca.explained_variance_ratio_[:k]
    V = pca.components_[:k].T.astype(np.float32)
    return mean.astype(np.float32), V, explained, k


def save_action_basis(path, V, explained_variance_ratio, **metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    V = np.asarray(V, dtype=np.float32)
    explained_variance_ratio = np.asarray(explained_variance_ratio, dtype=np.float32)
    np.savez_compressed(
        path,
        V=V,
        explained_variance_ratio=explained_variance_ratio,
        n_components=np.array([V.shape[1]], dtype=np.int32),
        feature_dim=np.array([V.shape[0]], dtype=np.int32),
    )
    meta = {
        "n_components": int(V.shape[1]),
        "feature_dim": int(V.shape[0]),
        "total_explained_variance": float(explained_variance_ratio.sum()),
        "explained_variance_ratio": explained_variance_ratio.tolist(),
        **metadata,
    }
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2))

