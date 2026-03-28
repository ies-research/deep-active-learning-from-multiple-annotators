from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple
from ._cache import sha1_json, sha1_bytes, to_plain
import json

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors


@dataclass(frozen=True)
class AnnotatorTypeConfig:
    """
    Configuration for an annotator archetype.

    Parameters
    ----------
    name:
        Identifier used for logging/debugging.
    proportion:
        Relative weight of this type in the annotator population.
        The weights are normalized internally and do not need to sum to 1.0.
    q_mean, q_std:
        For non-spammers: corrected skill ``q_a`` is sampled from a Beta
        distribution parameterized by mean/std on [0,1].
        Raw accuracy is mapped by class count ``K`` as:
        ``p_a = 1/K + (1 - 1/K) * q_a``.
        ``p_a`` defines the diagonal mass of the base confusion mean.
    s_mean, s_std:
        For non-spammers: global (per-annotator) confusion rows are sampled as
        ``Dirichlet(s_a * mu_row)``, where ``s_a` is drawn lognormally with
        mean ``s_mean`` and log-space sigma ``s_std``.
        Larger ``s_a`` means that the sampled rows stay closer to mu_row
        (more consistent).
    kappa_mean, kappa_std:
        Cluster-conditioned confusion rows are sampled as
        ``Dirichlet(kappa_a * C_base_row)``.
        Larger ``kappa_a`` means that there is less drift across clusters.
    difficulty_beta_mean, difficulty_beta_std:
        For non-spammers: annotator-specific sensitivity to instance
        difficulty. Larger ``beta_a`` means that hard instances cause a
        stronger drop in the probability of the correct label.
    spammer_mode:
        - If "uniform": annotator outputs labels uniformly at random over
          classes.
        - If "single_class": annotator always outputs one fixed class.
        - If None: normal annotator with p/s/kappa.
    single_class:
        Used only for spammer_mode="single_class". If ``None``, a random class
        is chosen.
    """

    name: str
    proportion: float

    q_mean: float = 0.0
    q_std: float = 0.0

    s_mean: float = 150.0
    s_std: float = 0.35

    kappa_mean: float = 200.0
    kappa_std: float = 0.35

    difficulty_beta_mean: float = 3.0
    difficulty_beta_std: float = 0.35

    spammer_mode: Optional[Literal["uniform", "single_class"]] = None
    single_class: Optional[int] = None


@dataclass(frozen=True)
class MultiAnnotatorSimConfig:
    """
    Configuration for multi-annotator simulation and caching.

    Simulation model
    ----------------
    - Assign each sample ``X[i]`` a cluster id ``c_i`` using k-means on ``X``.
    - Optionally derive an instance difficulty score ``d_i`` from local
      class overlap in feature space using k-nearest neighbors.
    - Optionally aggregate the local kNN label distributions within each
      ``(cluster, true-class)`` cell to form a cluster/class ambiguity template
      that steers off-diagonal confusion mass toward plausible alternatives.
    - Each annotator ``a`` has a global confusion matrix ``C_base[a]``.
    - For each cluster ``g``, sample a cluster-specific confusion matrix
      ``C[a,g]`` around a cluster mean that blends ``C_base[a]`` with the
      ambiguity template via a Dirichlet with concentration ``kappa_a``.
    - For each ``(i, a)```, sample label
      ``z[i,a] ~ Categorical(C[a, c_i, y_i, :])`` with optional missingness.
      If difficulty is enabled, the diagonal mass of ``C[a, c_i, y_i, :]`` is
      adjusted based on ``d_i`` while preserving the relative off-diagonal
      mass. The adjustment is centered so that average annotator quality stays
      close to the configured baseline.

    Caching requirement
    -------------------
    This module caches ``z_train``` in a way that is independent of the
    embedding model:
      - The cache key does NOT include the embedder fingerprint.
      - The cache key DOES include:
          * ``dataset_id`` (you provide it; should depend only on the
            dataset spec),
          * a hash of ``y_train`` bytes to bind cache to the exact sample
            ordering,
          * all simulation parameters.

    Parameters
    ----------
    seed:
        Base RNG seed.
    n_annotators:
        Number of annotators.
    allocation:
        How to assign annotator types.
        - "deterministic": Hamilton / largest remainder rounding of
          proportions (stable).
        - "iid": Sample types i.i.d. from normalized proportions.
    missing_rate:
        Fraction of items per annotator that are missing (Bernoulli).
    missing_value:
        Value used for missing labels (e.g., -1).
    use_clusters:
        If False, all samples share one cluster (no instance dependence).
    n_clusters:
        Number of clusters G used in ``k``-means (only if
        ``use_clusters=True``).
    kmeans_seed:
        RNG seed for k-means initialization.
    kmeans_iters:
        Number of Lloyd iterations for k-means.
    feature_preprocess:
        Preprocessing applied once to the feature matrix before any
        geometry-based simulation step. This affects both k-means clustering
        and kNN-derived difficulty / ambiguity estimation.
    use_difficulty:
        Whether to modulate sample-specific noise using a kNN-based difficulty
        score computed from ``X`` and ``y``.
    difficulty_k:
        Number of neighbors used to estimate local class overlap.
    difficulty_metric:
        Difficulty summary computed from neighborhood label frequencies.
    difficulty_alpha:
        Exponent applied after normalization. Values > 1 sharpen difficulty,
        values < 1 flatten it.
    use_knn_ambiguity:
        Whether to blend cluster/class ambiguity templates derived from local
        kNN label distributions into the cluster-conditioned confusion means.
    knn_ambiguity_blend:
        Blend weight between the annotator-specific off-diagonal base pattern
        and the cluster/class ambiguity template. ``0`` keeps the base pattern,
        ``1`` uses only the ambiguity template.
    knn_ambiguity_min_samples:
        Minimum number of samples required in a ``(cluster, true-class)`` cell
        before using its local ambiguity template. Sparser cells fall back to a
        global class-conditional ambiguity template.
    types:
        List of annotator type configs.

    Cache
    -----
    cache_dir:
        Directory to store cached ``z_train``.
    cache_version:
        Integer you can bump to force regeneration without deleting files.
    cache_store_metadata:
        Whether to store a JSON metadata sidecar (recommended).
    """

    seed: int = 0
    n_annotators: int = 20
    allocation: Literal["deterministic", "iid"] = "deterministic"

    missing_rate: float = 0.0
    missing_value: int = -1

    use_clusters: bool = True
    n_clusters: int = 50
    kmeans_seed: int = 0
    k_means_max_iter: int = 100
    k_means_batch_size: int = 1024
    feature_preprocess: Literal[
        "none", "l2_normalize", "standardize"
    ] = "none"

    use_difficulty: bool = False
    difficulty_k: int = 15
    difficulty_metric: Literal["entropy", "one_minus_max"] = "entropy"
    difficulty_alpha: float = 1.0
    use_knn_ambiguity: bool = False
    knn_ambiguity_blend: float = 0.35
    knn_ambiguity_min_samples: int = 5

    types: Sequence[AnnotatorTypeConfig] = ()

    cache_dir: str = ".hf_multi_annotator_cache"
    cache_version: int = 1
    cache_store_metadata: bool = True


def hash_y_train(y_train: np.ndarray) -> str:
    """
    Hash the exact `y_train` byte representation to bind cache to
    sample ordering.

    Parameters
    ----------
    y_train:
        Array of shape (N,). Will be cast to int64 for hashing.

    Returns
    -------
    y_hash:
        SHA1 hex digest of y_train bytes.
    """
    y = np.asarray(y_train)
    if y.dtype != np.int64:
        y = y.astype(np.int64, copy=False)
    return sha1_bytes(y.tobytes())


def make_dataset_id_from_spec_fingerprint(
    spec_fingerprint: Dict[str, Any],
) -> str:
    """
    Create a stable dataset identifier from a dataset spec fingerprint dict.

    This is just a convenience function: you can supply any dataset_id string
    you want, as long as it depends only on the dataset definition
    (NOT the embedding model).

    Parameters
    ----------
    spec_fingerprint:
        Dictionary describing the dataset setup (source, splits, keys,
        revisions, etc.).

    Returns
    -------
    dataset_id:
        SHA1 hash over the fingerprint dict.
    """
    return sha1_json(spec_fingerprint)


def _normalize_weights(w: np.ndarray) -> np.ndarray:
    s = float(w.sum())
    if s <= 0:
        raise ValueError("Sum of type proportions must be > 0.")
    return w / s


def _preprocess_simulation_features(
    X: np.ndarray,
    *,
    mode: Literal["none", "l2_normalize", "standardize"],
    eps: float = 1e-12,
) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError(
            "Simulation features must be a 2D array of shape (N, D), got "
            f"{X.shape}."
        )

    if mode == "none":
        return X

    if mode == "l2_normalize":
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms = np.maximum(norms, np.float32(eps))
        return X / norms

    if mode == "standardize":
        mean = X.mean(axis=0, dtype=np.float64)
        std = X.std(axis=0, dtype=np.float64)
        scale = np.maximum(std, eps)
        return ((X - mean) / scale).astype(np.float32, copy=False)

    raise ValueError(f"Unknown feature_preprocess={mode!r}.")


def allocate_type_ids(
    types: Sequence[AnnotatorTypeConfig],
    n_annotators: int,
    *,
    allocation: Literal["deterministic", "iid"],
    seed: int,
) -> np.ndarray:
    """
    Assign a type index to each annotator.

    Parameters
    ----------
    types:
        List of annotator types.
    n_annotators:
        Number of annotators A.
    allocation:
        "deterministic" or "iid".
    seed:
        RNG seed.

    Returns
    -------
    type_ids:
        Array of shape (A,) with integer type indices.
    """
    rng = np.random.default_rng(seed)
    weights = np.array(
        [max(0.0, t.proportion) for t in types], dtype=np.float64
    )
    probs = _normalize_weights(weights)

    if allocation == "iid":
        return rng.choice(len(types), size=n_annotators, p=probs)

    # deterministic Hamilton / largest remainder
    expected = probs * n_annotators
    counts = np.floor(expected).astype(int)
    remainder = n_annotators - counts.sum()

    frac = expected - np.floor(expected)
    order = np.argsort(-frac)
    for i in range(remainder):
        counts[order[i % len(types)]] += 1

    type_ids = np.repeat(np.arange(len(types)), counts)
    rng.shuffle(type_ids)
    return type_ids


def _clip01(x: float, eps: float = 1e-6) -> float:
    return float(np.clip(x, eps, 1.0 - eps))


def _lognormal(rng: np.random.Generator, mean: float, sigma: float) -> float:
    mean = max(mean, 1e-6)
    sigma = max(sigma, 1e-9)
    return float(rng.lognormal(mean=np.log(mean), sigma=sigma))


def _sample_beta_from_mean_std(
    rng: np.random.Generator, mean: float, std: float, eps: float = 1e-6
) -> float:
    """
    Sample x in (0,1) from a Beta distribution with target mean/std.

    If std <= 0, return deterministic mean (clipped to (eps, 1-eps)).
    """
    m = float(np.clip(mean, eps, 1.0 - eps))
    s = max(float(std), 0.0)
    if s <= 0.0:
        return m

    v = s * s
    max_v = m * (1.0 - m)
    if v >= max_v:
        raise ValueError(
            "Invalid q_std for Beta sampling: need q_std^2 < q_mean*(1-q_mean), "
            f"got q_mean={mean}, q_std={std}."
        )

    conc = (m * (1.0 - m) / v) - 1.0
    alpha = m * conc
    beta = (1.0 - m) * conc
    if alpha <= 0.0 or beta <= 0.0:
        raise ValueError(
            "Invalid Beta parameters derived from q_mean/q_std: "
            f"alpha={alpha}, beta={beta}."
        )
    return float(rng.beta(alpha, beta))


def build_annotator_params(
    types: Sequence[AnnotatorTypeConfig],
    type_ids: np.ndarray,
    *,
    n_classes: int,
    seed: int,
) -> Dict[str, Any]:
    """
    Sample per-annotator parameters from the configured archetypes.

    Parameters
    ----------
    types:
        Annotator archetype configs.
    type_ids:
        Array of shape (A,) assigning a type to each annotator.
    n_classes:
        Number of classes K.
    seed:
        RNG seed.

    Returns
    -------
    params:
        Dictionary containing:
        - q: (A,) float32, corrected skill for normal annotators
        - p: (A,) float32, accuracy for normal annotators
        - s: (A,) float32, base Dirichlet concentration
        - kappa: (A,) float32, cluster Dirichlet concentration
        - beta: (A,) float32, difficulty sensitivity
        - spammer_mode: list[str|None] length A
        - single_class: (A,) int64, class for single-class spammers
        - type_ids: (A,) int64
    """
    rng = np.random.default_rng(seed)
    A = int(type_ids.shape[0])
    chance = 1.0 / float(n_classes)

    q = np.empty(A, dtype=np.float32)
    p = np.empty(A, dtype=np.float32)
    s = np.empty(A, dtype=np.float32)
    kappa = np.empty(A, dtype=np.float32)
    beta = np.empty(A, dtype=np.float32)
    spammer_mode: List[Optional[str]] = [None] * A
    single_class = np.full(A, -1, dtype=np.int64)

    for a in range(A):
        t = types[int(type_ids[a])]

        if t.spammer_mode == "uniform":
            spammer_mode[a] = "uniform"
            q[a] = float(0.0)  # informational only
            p[a] = float(1.0 / n_classes)  # informational only
            beta[a] = 0.0
        elif t.spammer_mode == "single_class":
            spammer_mode[a] = "single_class"
            single_class[a] = int(
                t.single_class
                if t.single_class is not None
                else rng.integers(0, n_classes)
            )
            q[a] = np.nan
            p[a] = np.nan
            beta[a] = 0.0
        else:
            qa = _sample_beta_from_mean_std(rng, t.q_mean, t.q_std)
            q[a] = qa
            p[a] = _clip01(chance + (1.0 - chance) * qa)
            beta[a] = _lognormal(
                rng, t.difficulty_beta_mean, t.difficulty_beta_std
            )

        s[a] = _lognormal(rng, t.s_mean, t.s_std)
        kappa[a] = _lognormal(rng, t.kappa_mean, t.kappa_std)

    return {
        "type_ids": type_ids.astype(np.int64, copy=False),
        "q": q,
        "p": p,
        "s": s,
        "kappa": kappa,
        "beta": beta,
        "spammer_mode": spammer_mode,
        "single_class": single_class,
    }


def compute_knn_label_distribution(
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_classes: int,
    k: int,
) -> np.ndarray:
    """
    Estimate the local class distribution around each sample using kNN.

    For each sample, the returned row contains the empirical class
    distribution among its k nearest neighbors (excluding the sample itself).
    """
    X = np.asarray(X)
    y = np.asarray(y, dtype=np.int64)
    N = int(y.shape[0])

    if X.ndim != 2:
        raise ValueError(
            "X must be a 2D feature matrix for kNN difficulty, got shape "
            f"{X.shape}."
        )
    if N != X.shape[0]:
        raise ValueError(
            "X and y must agree in the number of samples for difficulty "
            f"estimation, got X.shape[0]={X.shape[0]} and len(y)={N}."
        )
    if N <= 1:
        return np.full((N, n_classes), 1.0 / max(n_classes, 1), dtype=np.float32)
    if k <= 0:
        raise ValueError(f"difficulty_k must be > 0, got {k}.")

    k_eff = min(int(k) + 1, N)
    neigh_ind = NearestNeighbors(n_neighbors=k_eff).fit(X).kneighbors(
        return_distance=False
    )
    neigh_ind = neigh_ind[:, 1:]
    if neigh_ind.shape[1] == 0:
        return np.full((N, n_classes), 1.0 / max(n_classes, 1), dtype=np.float32)

    neigh_labels = y[neigh_ind]
    counts = np.zeros((N, n_classes), dtype=np.float64)
    for c in range(n_classes):
        counts[:, c] = np.sum(neigh_labels == c, axis=1)

    return (
        counts / np.clip(counts.sum(axis=1, keepdims=True), 1.0, None)
    ).astype(np.float32)


def compute_knn_difficulty(
    knn_probs: np.ndarray,
    *,
    n_classes: int,
    metric: Literal["entropy", "one_minus_max"],
    alpha: float,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Estimate item difficulty from local label overlap in feature space.

    Parameters
    ----------
    knn_probs:
        Per-sample class distributions estimated from kNN, shape (N, K).
    n_classes:
        Number of classes K.
    metric:
        Difficulty summary computed from each row of ``knn_probs``.
    alpha:
        Exponent applied after normalization.
    eps:
        Numerical floor for log computations.
    """
    probs = np.asarray(knn_probs, dtype=np.float64)
    if probs.ndim != 2 or probs.shape[1] != n_classes:
        raise ValueError(
            "knn_probs must have shape (N, n_classes), got "
            f"{probs.shape} for n_classes={n_classes}."
        )
    if alpha <= 0:
        raise ValueError(f"difficulty_alpha must be > 0, got {alpha}.")

    if metric == "entropy":
        raw = -(probs * np.log(np.clip(probs, eps, 1.0))).sum(axis=1)
        if n_classes > 1:
            raw /= np.log(n_classes)
    elif metric == "one_minus_max":
        raw = 1.0 - probs.max(axis=1)
        if n_classes > 1:
            raw /= 1.0 - (1.0 / n_classes)
    else:
        raise ValueError(f"Unknown difficulty_metric={metric!r}.")

    raw = np.clip(raw, 0.0, 1.0)
    return np.power(raw, alpha).astype(np.float32)


def _normalize_off_diagonal_template(
    vec: np.ndarray,
    *,
    true_class: int,
    n_classes: int,
    smoothing: float = 1e-3,
) -> np.ndarray:
    out = np.asarray(vec, dtype=np.float64).copy()
    out = np.clip(out, 0.0, None)
    out[true_class] = 0.0

    if n_classes <= 1:
        return np.ones(1, dtype=np.float32)

    out += smoothing / max(n_classes - 1, 1)
    out[true_class] = 0.0
    s = out.sum()
    if s <= 0.0:
        out.fill(1.0 / (n_classes - 1))
        out[true_class] = 0.0
        s = out.sum()
    out /= s
    return out.astype(np.float32)


def build_cluster_ambiguity_templates(
    *,
    knn_probs: np.ndarray,  # (N, K)
    cluster_id: np.ndarray,  # (N,)
    y_true: np.ndarray,  # (N,)
    n_classes: int,
    n_clusters: int,
    min_samples: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Aggregate local kNN label distributions into cluster/class ambiguity rows.

    Returns
    -------
    ambiguity:
        Array of shape (G, K, K). Each row has zero diagonal and normalized
        off-diagonal mass.
    counts:
        Integer array of shape (G, K) with the sample count in each
        ``(cluster, true-class)`` cell.
    """
    knn_probs = np.asarray(knn_probs, dtype=np.float32)
    cluster_id = np.asarray(cluster_id, dtype=np.int64)
    y_true = np.asarray(y_true, dtype=np.int64)

    G = int(n_clusters)
    K = int(n_classes)

    if knn_probs.shape != (y_true.shape[0], K):
        raise ValueError(
            "knn_probs must have shape (N, K), got "
            f"{knn_probs.shape} for N={y_true.shape[0]}, K={K}."
        )

    counts = np.zeros((G, K), dtype=np.int64)
    global_raw = np.zeros((K, K), dtype=np.float64)
    global_counts = np.zeros(K, dtype=np.int64)

    for t in range(K):
        mask_t = y_true == t
        global_counts[t] = int(mask_t.sum())
        if global_counts[t] > 0:
            global_raw[t] = knn_probs[mask_t].mean(axis=0)

    global_templates = np.zeros((K, K), dtype=np.float32)
    for t in range(K):
        global_templates[t] = _normalize_off_diagonal_template(
            global_raw[t], true_class=t, n_classes=K
        )

    ambiguity = np.empty((G, K, K), dtype=np.float32)
    for g in range(G):
        mask_g = cluster_id == g
        for t in range(K):
            mask = mask_g & (y_true == t)
            counts[g, t] = int(mask.sum())
            if counts[g, t] >= min_samples:
                raw = knn_probs[mask].mean(axis=0)
                ambiguity[g, t] = _normalize_off_diagonal_template(
                    raw, true_class=t, n_classes=K
                )
            else:
                ambiguity[g, t] = global_templates[t]

    return ambiguity, counts


def _dirichlet_rows(
    rng: np.random.Generator, alpha_rows: np.ndarray
) -> np.ndarray:
    K = alpha_rows.shape[0]
    out = np.empty((K, K), dtype=np.float32)
    for t in range(K):
        out[t] = rng.dirichlet(alpha_rows[t]).astype(np.float32)
    return out


def sample_global_confusions(
    *,
    n_classes: int,
    p: np.ndarray,  # (A,)
    s: np.ndarray,  # (A,)
    spammer_mode: List[Optional[str]],
    single_class: np.ndarray,
    seed: int,
    eps: float = 1e-6,
) -> np.ndarray:
    """
    Create per-annotator global confusion matrices C_base.

    For normal annotators:
      - mean row mu has diag=p_a, off=(1-p_a)/(K-1)
      - sample each row Dirichlet(s_a * mu_row)

    For spammers:
      - uniform: rows are uniform
      - single_class: rows are one-hot at the spammed class

    Parameters
    ----------
    n_classes:
        Number of classes K.
    p, s:
        Arrays of shape (A,) with accuracy and base concentration.
    spammer_mode:
        List length A with spammer modes.
    single_class:
        Array of shape (A,) with the fixed class for single-class spammers.
    seed:
        RNG seed.
    eps:
        Minimum Dirichlet concentration.

    Returns
    -------
    C_base:
        Array of shape (A, K, K). Rows sum to 1.
    """
    rng = np.random.default_rng(seed)
    A = int(p.shape[0])
    K = int(n_classes)

    C_base = np.empty((A, K, K), dtype=np.float32)

    for a in range(A):
        mode = spammer_mode[a]
        if mode == "uniform":
            C_base[a] = np.full((K, K), 1.0 / K, dtype=np.float32)
            continue
        if mode == "single_class":
            j = int(single_class[a])
            M = np.zeros((K, K), dtype=np.float32)
            M[:, j] = 1.0
            C_base[a] = M
            continue

        pa = float(p[a])
        sa = float(s[a])

        off = (1.0 - pa) / (K - 1)
        mu = np.full((K, K), off, dtype=np.float32)
        np.fill_diagonal(mu, pa)

        alpha = np.clip(sa * mu, eps, None)
        C_base[a] = _dirichlet_rows(rng, alpha)

    return C_base


def sample_cluster_confusions(
    *,
    C_base: np.ndarray,  # (A, K, K)
    kappa: np.ndarray,  # (A,)
    n_clusters: int,
    ambiguity_templates: Optional[np.ndarray] = None,  # (G, K, K)
    ambiguity_blend: float = 0.0,
    use_ambiguity: Optional[np.ndarray] = None,  # (A,)
    seed: int,
    eps: float = 1e-6,
) -> np.ndarray:
    """
    Sample cluster-conditioned confusion matrices around C_base.

    Model:
      C[a,g,t,:] ~ Dirichlet(kappa[a] * M[a,g,t,:])
      where ``M`` optionally blends the annotator-specific off-diagonal base
      row with a cluster/class ambiguity template.

    Parameters
    ----------
    C_base:
        Global confusion matrices, shape (A, K, K).
    kappa:
        Cluster concentration per annotator, shape (A,).
    n_clusters:
        Number of clusters G.
    ambiguity_templates:
        Optional ambiguity template array of shape (G, K, K) with zero
        diagonal and normalized off-diagonal mass.
    ambiguity_blend:
        Blend weight in [0, 1] used when ``ambiguity_templates`` is provided.
    use_ambiguity:
        Optional boolean mask of shape (A,) indicating which annotators should
        use the ambiguity blend. Spammers can be excluded this way.
    seed:
        RNG seed.
    eps:
        Minimum Dirichlet concentration.

    Returns
    -------
    C_cluster:
        Array of shape (A, G, K, K).
    """
    rng = np.random.default_rng(seed)
    A, K, _ = C_base.shape
    G = int(n_clusters)
    blend = float(np.clip(ambiguity_blend, 0.0, 1.0))

    base = np.clip(C_base, eps, 1.0)
    C_cluster = np.empty((A, G, K, K), dtype=np.float32)

    if ambiguity_templates is not None:
        ambiguity_templates = np.asarray(ambiguity_templates, dtype=np.float32)
        if ambiguity_templates.shape != (G, K, K):
            raise ValueError(
                "ambiguity_templates must have shape (G, K, K), got "
                f"{ambiguity_templates.shape} for G={G}, K={K}."
            )
    if use_ambiguity is None:
        use_ambiguity = np.ones(A, dtype=bool)
    else:
        use_ambiguity = np.asarray(use_ambiguity, dtype=bool)
        if use_ambiguity.shape != (A,):
            raise ValueError(
                f"use_ambiguity must have shape (A,), got {use_ambiguity.shape}."
            )

    for a in range(A):
        ka = float(kappa[a])
        for g in range(G):
            for t in range(K):
                mean_row = base[a, t]
                if (
                    ambiguity_templates is not None
                    and blend > 0.0
                    and bool(use_ambiguity[a])
                ):
                    diag = float(mean_row[t])
                    off_base = mean_row.copy()
                    off_base[t] = 0.0
                    off_sum = float(off_base.sum())
                    if off_sum > 0.0:
                        off_base /= off_sum
                    else:
                        off_base = np.full(K, 1.0 / max(K - 1, 1), dtype=np.float32)
                        off_base[t] = 0.0

                    off_mix = (
                        (1.0 - blend) * off_base
                        + blend * ambiguity_templates[g, t]
                    )
                    off_mix[t] = 0.0
                    off_mix_sum = float(off_mix.sum())
                    if off_mix_sum > 0.0:
                        off_mix /= off_mix_sum
                    else:
                        off_mix = off_base

                    row = np.zeros(K, dtype=np.float32)
                    row[t] = diag
                    row += (1.0 - diag) * off_mix
                    row[t] = diag
                    mean_row = row

                alpha = np.clip(ka * mean_row, eps, None)
                C_cluster[a, g, t] = rng.dirichlet(alpha).astype(np.float32)

    return C_cluster


def simulate_labels(
    *,
    y_true: np.ndarray,  # (N,)
    cluster_id: np.ndarray,  # (N,) in [0..G-1]
    C_cluster: np.ndarray,  # (A,G,K,K)
    difficulty: np.ndarray,  # (N,)
    beta: np.ndarray,  # (A,)
    missing_rate: float,
    missing_value: int,
    seed: int,
) -> np.ndarray:
    """
    Sample annotator labels given cluster-conditioned confusion matrices.

    Parameters
    ----------
    y_true:
        True labels, shape (N,).
    cluster_id:
        Cluster assignments, shape (N,).
    C_cluster:
        Confusions, shape (A, G, K, K) where C[a,g,t,:] is a
        distribution over labels.
    difficulty:
        Centered difficulty score per sample, shape (N,). Values above zero
        correspond to harder-than-average instances, values below zero to
        easier-than-average instances.
    beta:
        Difficulty sensitivity per annotator, shape (A,). Zero disables the
        difficulty adjustment for that annotator.
    missing_rate:
        Bernoulli missing probability per (i,a).
    missing_value:
        Value used for missing labels.
    seed:
        RNG seed.

    Returns
    -------
    z:
        Noisy labels, shape (N, A), dtype int64.
    """
    rng = np.random.default_rng(seed)

    y_true = np.asarray(y_true, dtype=np.int64)
    cluster_id = np.asarray(cluster_id, dtype=np.int64)
    difficulty = np.asarray(difficulty, dtype=np.float32)
    beta = np.asarray(beta, dtype=np.float32)

    N = y_true.shape[0]
    A, G, K, _ = C_cluster.shape

    if difficulty.shape != (N,):
        raise ValueError(
            "difficulty must have shape (N,), got "
            f"{difficulty.shape} for N={N}."
        )
    if beta.shape != (A,):
        raise ValueError(
            f"beta must have shape (A,), got {beta.shape} for A={A}."
        )

    z = np.full((N, A), missing_value, dtype=np.int64)

    for a in range(A):
        obs = rng.random(N) >= missing_rate
        idx = np.where(obs)[0]
        if idx.size == 0:
            continue

        t = y_true[idx]
        g = cluster_id[idx]
        probs = C_cluster[a, g, t].astype(np.float64, copy=True)  # (n_obs, K)

        beta_a = float(beta[a])
        if beta_a > 0.0:
            diff = difficulty[idx].astype(np.float64, copy=False)
            chance = 1.0 / float(K)
            p0 = probs[np.arange(idx.size), t]

            # Apply difficulty to the accuracy margin above chance level and
            # re-normalize the multiplier so its mean is one over the current
            # sample set. This keeps marginal annotator quality close to the
            # configured baseline while introducing heteroscedastic noise.
            factor = np.exp(-beta_a * diff)
            factor /= np.clip(factor.mean(), 1e-12, None)

            p_eff = chance + (p0 - chance) * factor
            p_eff = np.clip(p_eff, chance, 1.0 - 1e-12)

            wrong = probs.copy()
            wrong[np.arange(idx.size), t] = 0.0
            wrong_sum = wrong.sum(axis=1, keepdims=True)
            fallback = np.full_like(wrong, 1.0 / max(K - 1, 1))
            fallback[np.arange(idx.size), t] = 0.0

            denom = np.where(wrong_sum > 1e-12, wrong_sum, 1.0)
            wrong_norm = np.where(wrong_sum > 1e-12, wrong / denom, fallback)
            probs = wrong_norm * (1.0 - p_eff)[:, None]
            probs[np.arange(idx.size), t] = p_eff

        u = rng.random(idx.size)
        cdf = np.cumsum(probs, axis=1)
        z[idx, a] = (cdf < u[:, None]).sum(axis=1).astype(np.int64)

    return z


def simulate_multi_annotator_labels_from_features(
    X_features: np.ndarray,
    y_true: np.ndarray,
    cfg: MultiAnnotatorSimConfig,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Simulate multi-annotator labels z for a dataset split.

    Parameters
    ----------
    X_features:
        Feature matrix used for instance dependence, shape (N, D).
        Typically this is your `np_arrays["X_train"]` when you run with
        an embedding model.
    y_true:
        True labels, shape (N,).
    cfg:
        Multi-annotator simulation configuration.

    Returns
    -------
    z:
        Simulated noisy labels, shape (N, A).
    info:
        Debug info: type_ids, n_classes, cluster_id, C_base (C_cluster omitted by default).
    """
    if cfg.n_annotators <= 0:
        raise ValueError("cfg.n_annotators must be > 0.")
    if len(cfg.types) == 0:
        raise ValueError("cfg.types must not be empty.")

    X_sim = _preprocess_simulation_features(
        X_features, mode=cfg.feature_preprocess
    )
    y_true = np.asarray(y_true, dtype=np.int64)
    K = int(np.unique(y_true).size)

    if cfg.use_clusters:
        cluster_id = (
            MiniBatchKMeans(
                n_clusters=cfg.n_clusters,
                random_state=cfg.kmeans_seed,
                max_iter=cfg.k_means_max_iter,
                batch_size=cfg.k_means_batch_size,
                compute_labels=True,
            )
            .fit(X_sim)
            .labels_
        )
        G = int(cfg.n_clusters)
    else:
        cluster_id = np.zeros(y_true.shape[0], dtype=np.int64)
        G = 1

    knn_probs = None
    if cfg.use_difficulty or cfg.use_knn_ambiguity:
        knn_probs = compute_knn_label_distribution(
            X_sim, y_true, n_classes=K, k=cfg.difficulty_k
        )

    if cfg.use_difficulty:
        difficulty_raw = compute_knn_difficulty(
            knn_probs,
            n_classes=K,
            metric=cfg.difficulty_metric,
            alpha=cfg.difficulty_alpha,
        )
        difficulty_mean = float(difficulty_raw.mean())
        difficulty = difficulty_raw - difficulty_mean
    else:
        difficulty_raw = np.zeros(y_true.shape[0], dtype=np.float32)
        difficulty_mean = 0.0
        difficulty = difficulty_raw

    if cfg.use_knn_ambiguity:
        ambiguity_templates, ambiguity_counts = build_cluster_ambiguity_templates(
            knn_probs=knn_probs,
            cluster_id=cluster_id,
            y_true=y_true,
            n_classes=K,
            n_clusters=G,
            min_samples=cfg.knn_ambiguity_min_samples,
        )
    else:
        ambiguity_templates = None
        ambiguity_counts = None

    type_ids = allocate_type_ids(
        cfg.types, cfg.n_annotators, allocation=cfg.allocation, seed=cfg.seed
    )
    params = build_annotator_params(
        cfg.types, type_ids, n_classes=K, seed=cfg.seed + 11
    )

    C_base = sample_global_confusions(
        n_classes=K,
        p=params["p"],
        s=params["s"],
        spammer_mode=params["spammer_mode"],
        single_class=params["single_class"],
        seed=cfg.seed + 23,
    )
    C_cluster = sample_cluster_confusions(
        C_base=C_base,
        kappa=params["kappa"],
        n_clusters=G,
        ambiguity_templates=ambiguity_templates,
        ambiguity_blend=cfg.knn_ambiguity_blend,
        use_ambiguity=np.array(
            [mode is None for mode in params["spammer_mode"]], dtype=bool
        ),
        seed=cfg.seed + 37,
    )

    z = simulate_labels(
        y_true=y_true,
        cluster_id=cluster_id,
        C_cluster=C_cluster,
        difficulty=difficulty,
        beta=params["beta"],
        missing_rate=cfg.missing_rate,
        missing_value=cfg.missing_value,
        seed=cfg.seed + 41,
    )

    info = {
        "n_classes": K,
        "n_clusters": G,
        "cluster_id": cluster_id,
        "type_ids": params["type_ids"],
        "q": params["q"],
        "p": params["p"],
        "s": params["s"],
        "kappa": params["kappa"],
        "beta": params["beta"],
        "spammer_mode": params["spammer_mode"],
        "single_class": params["single_class"],
        "feature_preprocess": cfg.feature_preprocess,
        "difficulty": difficulty_raw,
        "difficulty_centered": difficulty,
        "difficulty_mean": difficulty_mean,
        "knn_probs": knn_probs,
        "ambiguity_counts": ambiguity_counts,
        "knn_ambiguity_blend": float(cfg.knn_ambiguity_blend),
        "C_base": C_base,  # global confusions; small enough to keep
    }
    return z, info


def make_z_cache_key(
    *,
    dataset_id: str,
    y_hash: str,
    cfg: MultiAnnotatorSimConfig,
) -> str:
    """
    Create a cache key for z_train.

    IMPORTANT: This key does NOT include any embedding model fingerprint,
    by design. It binds to:
      - dataset_id (caller-supplied; should depend only on dataset spec)
      - y_hash (hash of y_train bytes to bind sample order/content)
      - all simulation parameters in cfg (including type definitions)
      - cache_version (manual invalidation knob)

    Parameters
    ----------
    dataset_id:
        Stable identifier of the dataset setup (not embedder).
    y_hash:
        Hash of y_train bytes.
    cfg:
        Simulation configuration.

    Returns
    -------
    key:
        SHA1 key used for cache filenames.
    """
    payload = {
        "dataset_id": dataset_id,
        "y_hash": y_hash,
        "cache_version": int(cfg.cache_version),
        "sim_cfg": to_plain(cfg),
    }
    return sha1_json(payload)


def ensure_z_train_cached(
    *,
    dataset_id: str,
    X_train_features: Optional[np.ndarray],
    y_train: np.ndarray,
    cfg: MultiAnnotatorSimConfig,
    embedder_fingerprint: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Load z_train from cache or simulate and cache it.

    This fulfills the requirement:
    - `z_train` cache lookup is independent of the embedding model used to
      create X_train.
    - On a cache miss, you *do* need features (X_train_features) to simulate
      instance dependence. After the first run, you can switch embedders freely
      and z_train will still load.

    Parameters
    ----------
    dataset_id:
        Stable dataset identifier (e.g., hash of spec_fingerprint(spec)).
        Must not depend on the embedding model.
    X_train_features:
        Feature matrix (N,D) used for clustering *only if cache miss*.
        If cache exists, this can be None.
    y_train:
        True labels (N,).
    cfg:
        Simulation config including cache_dir.
    embedder_fingerprint:
        Stored only in metadata for traceability; not part of cache key.

    Returns
    -------
    z_train:
        Array (N, A) noisy labels.
    info:
        Metadata dict (either loaded JSON or simulation info).
    """
    cache_dir = Path(cfg.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    y_hash = hash_y_train(y_train)
    key = make_z_cache_key(dataset_id=dataset_id, y_hash=y_hash, cfg=cfg)

    npz_path = cache_dir / f"{key}.npz"
    meta_path = cache_dir / f"{key}.json"

    # Cache hit
    if npz_path.exists():
        d = np.load(npz_path, allow_pickle=False)
        z = d["z_train"]
        info: Dict[str, Any] = {}
        if meta_path.exists():
            try:
                info = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                info = {}
        return z, info

    # Cache miss -> simulate
    if X_train_features is None:
        raise ValueError(
            "z_train cache miss and X_train_features is None. "
            "Provide (N,D) features for the initial simulation run."
        )

    X_train_features = np.asarray(X_train_features)
    if X_train_features.ndim != 2:
        raise ValueError(
            "Need 2D features (N,D) to simulate z_train, got shape "
            f"{X_train_features.shape}. Run once with an embedding model so "
            "X_train is (N,D)."
        )

    z, sim_info = simulate_multi_annotator_labels_from_features(
        X_train_features, y_train, cfg
    )

    np.savez_compressed(npz_path, z_train=z)

    if cfg.cache_store_metadata:
        meta = {
            "dataset_id": dataset_id,
            "y_hash": y_hash,
            "cache_version": int(cfg.cache_version),
            "sim_cfg": to_plain(cfg),
            "embedder_fingerprint_at_creation": to_plain(embedder_fingerprint),
            "sim_info_light": {
                "n_classes": int(sim_info.get("n_classes", -1)),
                "n_clusters": int(sim_info.get("n_clusters", -1)),
            },
        }
        meta_path.write_text(
            json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8"
        )

    return z, sim_info
