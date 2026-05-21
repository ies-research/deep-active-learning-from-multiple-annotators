from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple
from ._cache import sha1_json, sha1_bytes, to_plain
import json

import numpy as np
from sklearn.neighbors import NearestNeighbors


@dataclass(frozen=True)
class AnnotatorTypeConfig:
    """
    Configuration for one annotator archetype.

    Normal annotators follow a local-expertise GLAD/IRT-style model. Their
    corrected global skill ``q_a`` is sampled from ``q_mean/q_std`` and mapped
    to class-count-aware accuracy as ``p = chance + (1 - chance) * q_a``.
    ``difficulty_beta_*`` controls sensitivity to shared item difficulty.
    ``local_variability`` multiplies the regime-level local expertise target.

    Spammers ignore skill, difficulty, and local expertise:
    - ``uniform`` samples labels uniformly from all classes.
    - ``single_class`` always emits one fixed class.
    """

    name: str
    proportion: float

    q_mean: float = 0.0
    q_std: float = 0.0

    difficulty_beta_mean: float = 3.0
    difficulty_beta_std: float = 0.35
    local_variability: float = 1.0

    spammer_mode: Optional[Literal["uniform", "single_class"]] = None
    single_class: Optional[int] = None


@dataclass(frozen=True)
class MultiAnnotatorSimConfig:
    """
    Configuration for multi-annotator simulation and caching.

    The simulator uses a local-expertise GLAD/IRT-style model:

    ``eta_ai = logit(q_a) - beta_a * difficulty_i + local_effect_ai``

    where ``q_a`` is corrected global annotator skill, ``difficulty_i`` is a
    shared kNN class-overlap difficulty score, and ``local_effect_ai`` is an
    optional annotator-specific feature-local expertise term. Wrong labels for
    normal annotators are sampled uniformly over the incorrect classes.

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
    feature_preprocess:
        Preprocessing applied once to the feature matrix before any
        geometry-based simulation step.
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

    feature_preprocess: Literal[
        "none", "l2_normalize", "standardize"
    ] = "none"

    use_difficulty: bool = True
    difficulty_k: int = 15
    difficulty_metric: Literal["entropy", "one_minus_max"] = "entropy"
    difficulty_alpha: float = 1.0

    local_expertise_enabled: bool = False
    local_expertise_kind: Literal["feature", "class"] = "feature"
    local_expertise_target_gap_q: float = 0.0
    local_expertise_n_classes: int = 2
    local_expertise_n_prototypes: int = 3
    local_expertise_bandwidth_quantile: float = 0.10
    local_expertise_prototype_sampling: Literal["class_balanced"] = (
        "class_balanced"
    )
    local_expertise_score: Literal["rbf_max"] = "rbf_max"
    local_expertise_q_min: float = 0.0
    local_expertise_q_max: float = 0.98

    types: Sequence[AnnotatorTypeConfig] = ()

    cache_dir: str = ".hf_multi_annotator_cache"
    cache_version: int = 1
    cache_store_metadata: bool = True


def _as_1d_int_labels(y: np.ndarray, *, name: str = "y") -> np.ndarray:
    y = np.asarray(y, dtype=np.int64)
    if y.ndim == 1:
        return y
    if y.ndim == 2 and y.shape[1] == 1:
        return y.reshape(-1)
    raise ValueError(f"{name} must have shape (N,) or (N, 1), got {y.shape}.")


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
    y = _as_1d_int_labels(y_train, name="y_train")
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


def _logit(p: np.ndarray | float, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x, dtype=np.float64)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    exp_x = np.exp(x[~pos])
    out[~pos] = exp_x / (1.0 + exp_x)
    return out


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
    rng = np.random.default_rng(seed)
    A = int(type_ids.shape[0])
    chance = 1.0 / float(n_classes)

    q = np.empty(A, dtype=np.float32)
    p = np.empty(A, dtype=np.float32)
    beta = np.empty(A, dtype=np.float32)
    local_variability = np.empty(A, dtype=np.float32)
    spammer_mode: List[Optional[str]] = [None] * A
    single_class = np.full(A, -1, dtype=np.int64)
    type_names = []

    for a in range(A):
        t = types[int(type_ids[a])]
        type_names.append(t.name)

        if t.spammer_mode == "uniform":
            spammer_mode[a] = "uniform"
            q[a] = 0.0
            p[a] = chance
            beta[a] = 0.0
            local_variability[a] = 0.0
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
            local_variability[a] = 0.0
        else:
            qa = _sample_beta_from_mean_std(rng, t.q_mean, t.q_std)
            q[a] = qa
            p[a] = _clip01(chance + (1.0 - chance) * qa)
            beta[a] = _lognormal(
                rng, t.difficulty_beta_mean, t.difficulty_beta_std
            )
            local_variability[a] = max(float(t.local_variability), 0.0)

    return {
        "type_ids": type_ids.astype(np.int64, copy=False),
        "type_names": type_names,
        "q": q,
        "p": p,
        "beta": beta,
        "local_variability": local_variability,
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
    y = _as_1d_int_labels(y, name="y")
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


def _select_local_expertise_prototypes(
    *,
    y_true: np.ndarray,
    n_annotators: int,
    n_prototypes: int,
    spammer_mode: Sequence[Optional[str]],
    seed: int,
) -> np.ndarray:
    if n_prototypes <= 0:
        raise ValueError(
            "local_expertise_n_prototypes must be > 0 when local expertise "
            f"is enabled, got {n_prototypes}."
        )

    rng = np.random.default_rng(seed)
    classes = np.unique(y_true)
    by_class = {int(c): np.flatnonzero(y_true == c) for c in classes}
    prototype_indices = np.full((n_annotators, n_prototypes), -1, dtype=np.int64)
    class_order = rng.permutation(classes)
    class_cursor = 0

    for a in range(n_annotators):
        if spammer_mode[a] is not None:
            continue
        for s in range(n_prototypes):
            if class_cursor % len(class_order) == 0:
                class_order = rng.permutation(classes)
            cls = int(class_order[class_cursor % len(class_order)])
            class_cursor += 1
            candidates = by_class.get(cls, np.empty(0, dtype=int))
            if candidates.size == 0:
                candidates = np.arange(y_true.shape[0])
            prototype_indices[a, s] = int(rng.choice(candidates))

    return prototype_indices


def _compute_local_expertise_scores(
    *,
    X: np.ndarray,
    prototype_indices: np.ndarray,
    bandwidth_quantile: float,
) -> np.ndarray:
    if not 0.0 < float(bandwidth_quantile) < 1.0:
        raise ValueError(
            "local_expertise_bandwidth_quantile must be in (0, 1), got "
            f"{bandwidth_quantile}."
        )

    X = np.asarray(X, dtype=np.float32)
    N = X.shape[0]
    A, _ = prototype_indices.shape
    scores = np.zeros((N, A), dtype=np.float32)

    for a in range(A):
        valid = prototype_indices[a][prototype_indices[a] >= 0]
        if valid.size == 0:
            continue
        annotator_score = np.zeros(N, dtype=np.float32)
        for idx in valid:
            dist = np.linalg.norm(X - X[int(idx)], axis=1)
            positive_dist = dist[dist > 1e-12]
            if positive_dist.size == 0:
                sigma = 1.0
            else:
                sigma = float(np.quantile(positive_dist, bandwidth_quantile))
                sigma = max(sigma, 1e-12)
            rbf = np.exp(-0.5 * np.square(dist / sigma)).astype(np.float32)
            annotator_score = np.maximum(annotator_score, rbf)
        scores[:, a] = annotator_score

    return scores


def _select_class_expertise_classes(
    *,
    classes: np.ndarray,
    n_annotators: int,
    n_classes_per_annotator: int,
    spammer_mode: Sequence[Optional[str]],
    seed: int,
) -> np.ndarray:
    if n_classes_per_annotator <= 0:
        raise ValueError(
            "local_expertise_n_classes must be > 0 for class expertise, got "
            f"{n_classes_per_annotator}."
        )

    rng = np.random.default_rng(seed)
    classes = np.asarray(classes, dtype=np.int64)
    selected = np.full(
        (n_annotators, n_classes_per_annotator), -1, dtype=np.int64
    )
    class_order = rng.permutation(classes)
    class_cursor = 0

    for a in range(n_annotators):
        if spammer_mode[a] is not None:
            continue
        for s in range(n_classes_per_annotator):
            if class_cursor % len(class_order) == 0:
                class_order = rng.permutation(classes)
            selected[a, s] = int(class_order[class_cursor % len(class_order)])
            class_cursor += 1

    return selected


def _compute_class_expertise_scores(
    *,
    y_true: np.ndarray,
    class_indices: np.ndarray,
) -> np.ndarray:
    y_true = _as_1d_int_labels(y_true, name="y_true")
    N = y_true.shape[0]
    A = class_indices.shape[0]
    scores = np.zeros((N, A), dtype=np.float32)
    for a in range(A):
        selected = class_indices[a][class_indices[a] >= 0]
        if selected.size:
            scores[:, a] = np.isin(y_true, selected).astype(np.float32)
    return scores


def _quartile_masks(score: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    values = np.unique(score)
    if values.size == 2:
        bottom = score == values[0]
        top = score == values[1]
        if top.sum() > 0 and bottom.sum() > 0:
            return top, bottom

    lo = np.quantile(score, 0.25)
    hi = np.quantile(score, 0.75)
    bottom = score <= lo
    top = score >= hi
    if top.sum() == 0 or bottom.sum() == 0:
        order = np.argsort(score)
        n = max(1, score.shape[0] // 4)
        bottom = np.zeros(score.shape[0], dtype=bool)
        top = np.zeros(score.shape[0], dtype=bool)
        bottom[order[:n]] = True
        top[order[-n:]] = True
    return top, bottom


def _calibrate_local_scale(
    *,
    base_eta: np.ndarray,
    centered_score: np.ndarray,
    target_gap_q: float,
    q_min: float,
    q_max: float,
) -> Tuple[float, float, float]:
    if target_gap_q <= 0.0 or np.allclose(centered_score, 0.0):
        return 0.0, 0.0, 0.0

    top, bottom = _quartile_masks(centered_score)
    base_q = np.clip(_sigmoid(base_eta), q_min, q_max)

    def effect_gap(scale: float) -> float:
        q = np.clip(_sigmoid(base_eta + scale * centered_score), q_min, q_max)
        delta = q - base_q
        return float(delta[top].mean() - delta[bottom].mean())

    high = 1.0
    high_gap = effect_gap(high)
    while high_gap < target_gap_q and high < 256.0:
        high *= 2.0
        high_gap = effect_gap(high)

    low = 0.0
    for _ in range(40):
        mid = 0.5 * (low + high)
        if effect_gap(mid) < target_gap_q:
            low = mid
        else:
            high = mid

    scale = high
    achieved_effect_gap = effect_gap(scale)
    q = np.clip(_sigmoid(base_eta + scale * centered_score), q_min, q_max)
    achieved_total_gap = float(q[top].mean() - q[bottom].mean())
    return float(scale), achieved_effect_gap, achieved_total_gap


def _apply_local_expertise(
    *,
    base_eta: np.ndarray,
    local_scores: np.ndarray,
    params: Dict[str, Any],
    cfg: MultiAnnotatorSimConfig,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    q_min = float(cfg.local_expertise_q_min)
    q_max = float(cfg.local_expertise_q_max)
    if not 0.0 <= q_min < q_max <= 1.0:
        raise ValueError(
            "Require 0 <= local_expertise_q_min < local_expertise_q_max <= 1."
        )

    eta = base_eta.copy()
    A = eta.shape[1]
    scales = np.zeros(A, dtype=np.float32)
    effect_gaps = np.zeros(A, dtype=np.float32)
    total_gaps = np.zeros(A, dtype=np.float32)
    target_gaps = np.zeros(A, dtype=np.float32)

    if not cfg.local_expertise_enabled:
        return eta, {
            "scales": scales,
            "effect_gaps": effect_gaps,
            "total_gaps": total_gaps,
            "target_gaps": target_gaps,
        }

    target_base = max(float(cfg.local_expertise_target_gap_q), 0.0)
    for a in range(A):
        if params["spammer_mode"][a] is not None:
            continue
        target = target_base * float(params["local_variability"][a])
        target_gaps[a] = target
        if target <= 0.0:
            continue
        score = local_scores[:, a].astype(np.float64, copy=False)
        centered = score - float(score.mean())
        scale, effect_gap, total_gap = _calibrate_local_scale(
            base_eta=base_eta[:, a],
            centered_score=centered,
            target_gap_q=target,
            q_min=q_min,
            q_max=q_max,
        )
        eta[:, a] = base_eta[:, a] + scale * centered
        scales[a] = scale
        effect_gaps[a] = effect_gap
        total_gaps[a] = total_gap

    return eta, {
        "scales": scales,
        "effect_gaps": effect_gaps,
        "total_gaps": total_gaps,
        "target_gaps": target_gaps,
    }


def _sample_local_expertise_labels(
    *,
    y_true: np.ndarray,
    classes: np.ndarray,
    q: np.ndarray,
    params: Dict[str, Any],
    missing_rate: float,
    missing_value: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    y_true = _as_1d_int_labels(y_true, name="y_true")
    N, A = q.shape
    K = len(classes)
    chance = 1.0 / float(K)

    class_to_pos = {int(cls): pos for pos, cls in enumerate(classes)}
    y_pos = np.asarray([class_to_pos[int(y)] for y in y_true], dtype=np.int64)
    z = np.full((N, A), missing_value, dtype=np.int64)

    for a in range(A):
        obs = rng.random(N) >= missing_rate
        idx = np.flatnonzero(obs)
        if idx.size == 0:
            continue

        mode = params["spammer_mode"][a]
        if mode == "uniform":
            z[idx, a] = rng.choice(classes, size=idx.size)
            continue
        if mode == "single_class":
            z[idx, a] = classes[int(params["single_class"][a])]
            continue

        p_correct = chance + (1.0 - chance) * q[idx, a]
        is_correct = rng.random(idx.size) < p_correct
        labels = np.empty(idx.size, dtype=np.int64)
        labels[is_correct] = y_true[idx[is_correct]]

        wrong_local = np.flatnonzero(~is_correct)
        if wrong_local.size:
            if K <= 1:
                labels[wrong_local] = y_true[idx[wrong_local]]
            else:
                offsets = rng.integers(1, K, size=wrong_local.size)
                wrong_pos = (y_pos[idx[wrong_local]] + offsets) % K
                labels[wrong_local] = classes[wrong_pos]
        z[idx, a] = labels

    return z


def _format_float(value: float) -> str:
    if not np.isfinite(value):
        return "nan"
    return f"{value:.4f}"


def _print_simulation_diagnostics(
    *,
    z: np.ndarray,
    y_true: np.ndarray,
    classes: np.ndarray,
    params: Dict[str, Any],
    cfg: MultiAnnotatorSimConfig,
    local_scores: Optional[np.ndarray],
    local_diag: Dict[str, Any],
) -> None:
    print(
        "[local_expertise_sim] "
        f"classes={len(classes)} annotators={z.shape[1]} "
        f"difficulty={bool(cfg.use_difficulty)} "
        f"local_enabled={bool(cfg.local_expertise_enabled)} "
        f"kind={cfg.local_expertise_kind} "
        f"target_gap_q={float(cfg.local_expertise_target_gap_q):.4f}"
    )

    if not cfg.local_expertise_enabled or local_scores is None:
        print("[local_expertise_sim] local expertise disabled.")
        return

    chance = 1.0 / float(len(classes))
    normal = np.array([m is None for m in params["spammer_mode"]], dtype=bool)
    sampled_gaps = np.full(z.shape[1], np.nan, dtype=np.float64)
    present = z != cfg.missing_value
    correct = (z == y_true[:, None]) & present

    for a in np.flatnonzero(normal):
        top, bottom = _quartile_masks(local_scores[:, a])
        top_present = top & present[:, a]
        bottom_present = bottom & present[:, a]
        if top_present.sum() == 0 or bottom_present.sum() == 0:
            continue
        acc_top = correct[top_present, a].mean()
        acc_bottom = correct[bottom_present, a].mean()
        sampled_gaps[a] = (acc_top - acc_bottom) / max(1.0 - chance, 1e-12)

    effect_gaps = np.asarray(local_diag["effect_gaps"], dtype=float)
    total_gaps = np.asarray(local_diag["total_gaps"], dtype=float)
    target_gaps = np.asarray(local_diag["target_gaps"], dtype=float)
    norm_idx = np.flatnonzero(normal)
    if norm_idx.size == 0:
        print("[local_expertise_sim] no normal annotators for local diagnostics.")
        return

    print(
        "[local_expertise_sim] gap_q normal annotators "
        f"target_mean={_format_float(np.nanmean(target_gaps[norm_idx]))} "
        f"effect_mean={_format_float(np.nanmean(effect_gaps[norm_idx]))} "
        f"effect_median={_format_float(np.nanmedian(effect_gaps[norm_idx]))} "
        f"total_mean={_format_float(np.nanmean(total_gaps[norm_idx]))} "
        f"sampled_mean={_format_float(np.nanmean(sampled_gaps[norm_idx]))} "
        f"sampled_median={_format_float(np.nanmedian(sampled_gaps[norm_idx]))}"
    )

    type_names = np.asarray(params["type_names"], dtype=object)
    for type_name in sorted({str(t) for t in type_names[norm_idx]}):
        mask = normal & (type_names == type_name)
        print(
            "[local_expertise_sim] "
            f"type={type_name} n={int(mask.sum())} "
            f"target={_format_float(np.nanmean(target_gaps[mask]))} "
            f"effect={_format_float(np.nanmean(effect_gaps[mask]))} "
            f"sampled={_format_float(np.nanmean(sampled_gaps[mask]))}"
        )


def simulate_multi_annotator_labels_from_features(
    X_features: np.ndarray,
    y_true: np.ndarray,
    cfg: MultiAnnotatorSimConfig,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    if cfg.n_annotators <= 0:
        raise ValueError("cfg.n_annotators must be > 0.")
    if len(cfg.types) == 0:
        raise ValueError("cfg.types must not be empty.")
    if cfg.local_expertise_kind not in {"feature", "class"}:
        raise ValueError("local_expertise_kind must be 'feature' or 'class'.")
    if (
        cfg.local_expertise_kind == "feature"
        and cfg.local_expertise_prototype_sampling != "class_balanced"
    ):
        raise ValueError("Only class_balanced prototype sampling is supported.")
    if cfg.local_expertise_kind == "feature" and cfg.local_expertise_score != "rbf_max":
        raise ValueError("Only rbf_max local expertise scoring is supported.")

    X_sim = _preprocess_simulation_features(
        X_features, mode=cfg.feature_preprocess
    )
    y_true = _as_1d_int_labels(y_true, name="y_true")
    classes = np.unique(y_true).astype(np.int64, copy=False)
    K = int(classes.size)
    if K <= 0:
        raise ValueError("y_true must contain at least one class.")

    knn_probs = None
    if cfg.use_difficulty:
        knn_probs = compute_knn_label_distribution(
            X_sim, y_true, n_classes=K, k=cfg.difficulty_k
        )
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

    type_ids = allocate_type_ids(
        cfg.types, cfg.n_annotators, allocation=cfg.allocation, seed=cfg.seed
    )
    params = build_annotator_params(
        cfg.types, type_ids, n_classes=K, seed=cfg.seed + 11
    )

    N = y_true.shape[0]
    A = cfg.n_annotators
    base_eta = np.zeros((N, A), dtype=np.float64)
    for a in range(A):
        if params["spammer_mode"][a] is None:
            base_eta[:, a] = _logit(float(params["q"][a])) - (
                float(params["beta"][a]) * difficulty
            )

    prototype_indices = np.full(
        (A, int(max(cfg.local_expertise_n_prototypes, 1))),
        -1,
        dtype=np.int64,
    )
    class_expertise_classes = np.full(
        (A, int(max(cfg.local_expertise_n_classes, 1))),
        -1,
        dtype=np.int64,
    )
    if cfg.local_expertise_enabled:
        if cfg.local_expertise_kind == "class":
            class_expertise_classes = _select_class_expertise_classes(
                classes=classes,
                n_annotators=A,
                n_classes_per_annotator=int(cfg.local_expertise_n_classes),
                spammer_mode=params["spammer_mode"],
                seed=cfg.seed + 23,
            )
            local_scores = _compute_class_expertise_scores(
                y_true=y_true,
                class_indices=class_expertise_classes,
            )
        else:
            prototype_indices = _select_local_expertise_prototypes(
                y_true=y_true,
                n_annotators=A,
                n_prototypes=int(cfg.local_expertise_n_prototypes),
                spammer_mode=params["spammer_mode"],
                seed=cfg.seed + 23,
            )
            local_scores = _compute_local_expertise_scores(
                X=X_sim,
                prototype_indices=prototype_indices,
                bandwidth_quantile=float(cfg.local_expertise_bandwidth_quantile),
            )
    else:
        local_scores = np.zeros((N, A), dtype=np.float32)

    eta, local_diag = _apply_local_expertise(
        base_eta=base_eta,
        local_scores=local_scores,
        params=params,
        cfg=cfg,
    )
    q = np.clip(
        _sigmoid(eta),
        float(cfg.local_expertise_q_min),
        float(cfg.local_expertise_q_max),
    ).astype(np.float32)

    z = _sample_local_expertise_labels(
        y_true=y_true,
        classes=classes,
        q=q,
        params=params,
        missing_rate=float(cfg.missing_rate),
        missing_value=int(cfg.missing_value),
        seed=cfg.seed + 41,
    )

    _print_simulation_diagnostics(
        z=z,
        y_true=y_true,
        classes=classes,
        params=params,
        cfg=cfg,
        local_scores=local_scores if cfg.local_expertise_enabled else None,
        local_diag=local_diag,
    )

    info = {
        "n_classes": K,
        "classes": classes,
        "type_ids": params["type_ids"],
        "type_names": params["type_names"],
        "q": params["q"],
        "p": params["p"],
        "beta": params["beta"],
        "local_variability": params["local_variability"],
        "spammer_mode": params["spammer_mode"],
        "single_class": params["single_class"],
        "feature_preprocess": cfg.feature_preprocess,
        "difficulty": difficulty_raw,
        "difficulty_centered": difficulty,
        "difficulty_mean": difficulty_mean,
        "knn_probs": knn_probs,
        "local_expertise_enabled": bool(cfg.local_expertise_enabled),
        "local_expertise_kind": cfg.local_expertise_kind,
        "local_expertise_target_gap_q": float(
            cfg.local_expertise_target_gap_q
        ),
        "local_expertise_prototype_indices": prototype_indices,
        "local_expertise_classes": class_expertise_classes,
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
      difficulty and local expertise. After the first run, you can switch
      embedders freely and z_train will still load.

    Parameters
    ----------
    dataset_id:
        Stable dataset identifier (e.g., hash of spec_fingerprint(spec)).
        Must not depend on the embedding model.
    X_train_features:
        Feature matrix (N,D) used for simulation *only if cache miss*. If cache
        exists, this can be None.
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
        print(
            "[local_expertise_sim] loaded cached z_train; diagnostics are "
            "printed only on cache miss."
        )
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
                "local_expertise_enabled": bool(
                    sim_info.get("local_expertise_enabled", False)
                ),
            },
        }
        meta_path.write_text(
            json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8"
        )

    return z, sim_info
