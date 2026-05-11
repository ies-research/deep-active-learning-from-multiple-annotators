from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import check_random_state

from skactiveml.utils import MISSING_LABEL

from ._base import PairScorer


# Preset names follow:
#   <evidence_scope>_<posterior_family>_<prior_source>_prior
# where fixed priors are annotator-independent channels, and global/local
# priors are empirical base estimates regularized by fixed_prior_*.
_PRESETS = {
    "global_full_fixed_prior": {
        "base_channel": "fixed_full",
        "evidence": "global_soft_counts",
        "posterior": "dirichlet_global",
    },
    "global_full_global_full_prior": {
        "base_channel": "global_full",
        "evidence": "global_soft_counts",
        "posterior": "dirichlet_global",
    },
    "global_full_global_mace_prior": {
        "base_channel": "global_mace",
        "evidence": "global_soft_counts",
        "posterior": "dirichlet_global",
    },
    "global_full_global_accuracy_uniform_prior": {
        "base_channel": "global_accuracy_uniform",
        "evidence": "global_soft_counts",
        "posterior": "dirichlet_global",
    },
    "local_full_fixed_prior": {
        "base_channel": "fixed_full",
        "evidence": "local_kernel_soft_counts",
        "posterior": "dirichlet_local",
    },
    "local_full_global_full_prior": {
        "base_channel": "global_full",
        "evidence": "local_kernel_soft_counts",
        "posterior": "dirichlet_local",
    },
    "local_full_global_mace_prior": {
        "base_channel": "global_mace",
        "evidence": "local_kernel_soft_counts",
        "posterior": "dirichlet_local",
    },
    "local_full_global_accuracy_uniform_prior": {
        "base_channel": "global_accuracy_uniform",
        "evidence": "local_kernel_soft_counts",
        "posterior": "dirichlet_local",
    },
    "local_full_local_mace_prior": {
        "base_channel": "local_mace",
        "evidence": "local_kernel_soft_counts",
        "posterior": "dirichlet_local",
    },
    "local_full_local_accuracy_uniform_prior": {
        "base_channel": "local_accuracy_uniform",
        "evidence": "local_kernel_soft_counts",
        "posterior": "dirichlet_local",
    },
    "global_mace_fixed_prior": {
        "base_channel": "fixed_mace",
        "evidence": "global_mace_counts",
        "posterior": "global_mace",
    },
    "global_mace_global_mace_prior": {
        "base_channel": "global_mace",
        "evidence": "global_mace_counts",
        "posterior": "global_mace",
    },
    "global_mace_global_full_prior": {
        "base_channel": "global_full",
        "evidence": "global_mace_counts",
        "posterior": "global_mace",
    },
    "local_mace_fixed_prior": {
        "base_channel": "fixed_mace",
        "evidence": "local_kernel_mace_counts",
        "posterior": "local_mace",
    },
    "local_mace_global_mace_prior": {
        "base_channel": "global_mace",
        "evidence": "local_kernel_mace_counts",
        "posterior": "local_mace",
    },
    "local_mace_global_full_prior": {
        "base_channel": "global_full",
        "evidence": "local_kernel_mace_counts",
        "posterior": "local_mace",
    },
    "global_accuracy_uniform_fixed_prior": {
        "base_channel": "fixed_accuracy_uniform",
        "evidence": "global_accuracy_counts",
        "posterior": "global_accuracy_uniform",
    },
    "global_accuracy_uniform_global_accuracy_uniform_prior": {
        "base_channel": "global_accuracy_uniform",
        "evidence": "global_accuracy_counts",
        "posterior": "global_accuracy_uniform",
    },
    "global_accuracy_uniform_global_full_prior": {
        "base_channel": "global_full",
        "evidence": "global_accuracy_counts",
        "posterior": "global_accuracy_uniform",
    },
    "local_accuracy_uniform_fixed_prior": {
        "base_channel": "fixed_accuracy_uniform",
        "evidence": "local_kernel_accuracy_counts",
        "posterior": "local_accuracy_uniform",
    },
    "local_accuracy_uniform_global_accuracy_uniform_prior": {
        "base_channel": "global_accuracy_uniform",
        "evidence": "local_kernel_accuracy_counts",
        "posterior": "local_accuracy_uniform",
    },
    "local_accuracy_uniform_global_full_prior": {
        "base_channel": "global_full",
        "evidence": "local_kernel_accuracy_counts",
        "posterior": "local_accuracy_uniform",
    },
    "local_balanced_accuracy_global_full_prior": {
        "base_channel": "global_full",
        "evidence": "local_kernel_balanced_accuracy_counts",
        "posterior": "local_balanced_accuracy",
    },
}


@dataclass
class ChannelEstimationResult:
    base_channel: np.ndarray
    evidence_counts: (
        np.ndarray
        | MaceEvidenceCounts
        | AccuracyEvidenceCounts
        | "LocalBalancedAccuracyResult"
        | None
    )
    posterior_alpha: np.ndarray | None
    posterior_channel: np.ndarray
    base_mace_params: "MaceChannelParameters | None"
    evidence_mace_counts: "MaceEvidenceCounts | None"
    posterior_mace_params: "MaceChannelParameters | None"
    base_accuracy_params: "AccuracyUniformChannelParameters | None"
    evidence_accuracy_counts: "AccuracyEvidenceCounts | None"
    posterior_accuracy_params: "AccuracyUniformChannelParameters | None"
    classifier_proba: np.ndarray
    observed_class_proba: np.ndarray
    class_prior: np.ndarray
    class_prior_alpha: np.ndarray | None
    class_prior_laplace_logits: np.ndarray | None
    class_prior_laplace_logit_variance: np.ndarray | None
    observed_laplace_logit_variance: np.ndarray | None
    local_class_accuracy: np.ndarray | None
    local_balanced_accuracy: np.ndarray | None
    local_corrected_balanced_accuracy: np.ndarray | None
    local_balanced_total: np.ndarray | None
    local_balanced_correct: np.ndarray | None
    budget_aware_locality: "BudgetAwareLocalityResult | None"
    prior_mask: np.ndarray
    evidence_mask: np.ndarray
    prior_observations: str
    uses_same_prior_and_evidence: bool
    sample_indices: np.ndarray
    annotator_indices: np.ndarray


@dataclass
class LaplacePredictiveResult:
    proba: np.ndarray
    logit_variance: np.ndarray
    samples: np.ndarray | None = None


@dataclass
class BudgetAwareLocalityResult:
    k_t: int
    k_final: int
    s_local: float
    diagnostics: dict


@dataclass
class LocalBalancedAccuracyResult:
    theta: np.ndarray
    balanced_accuracy: np.ndarray
    corrected_balanced_accuracy: np.ndarray
    total: np.ndarray
    correct: np.ndarray
    prior_diag: np.ndarray
    prior_strength: float


@dataclass
class MaceChannelParameters:
    theta: np.ndarray
    g: np.ndarray
    theta_success_alpha: np.ndarray | None = None
    theta_failure_beta: np.ndarray | None = None
    g_alpha: np.ndarray | None = None


@dataclass
class MaceEvidenceCounts:
    theta_success: np.ndarray
    theta_failure: np.ndarray
    g_counts: np.ndarray


@dataclass
class AccuracyUniformChannelParameters:
    theta: np.ndarray
    alpha: np.ndarray | None = None
    beta: np.ndarray | None = None


@dataclass
class AccuracyEvidenceCounts:
    success: np.ndarray
    failure: np.ndarray


def _normalize_axis(X: np.ndarray, *, axis: int = -1, eps: float = 1e-12):
    X = np.asarray(X, dtype=float)
    return X / np.maximum(X.sum(axis=axis, keepdims=True), eps)


def _softmax_logits(logits: np.ndarray, *, axis: int = -1) -> np.ndarray:
    logits = np.asarray(logits, dtype=float)
    shifted = logits - np.max(logits, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.maximum(exp.sum(axis=axis, keepdims=True), 1e-300)


def _l2_normalize(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(n, eps)


def _is_missing(y: np.ndarray, missing_label) -> np.ndarray:
    y = np.asarray(y)
    try:
        if bool(np.isnan(missing_label)):
            return y != y
    except TypeError:
        pass
    return y == missing_label


def _resolve_gamma_from_embeddings(
    X: np.ndarray,
    gamma,
    *,
    bandwidth_knn_k: int = 10,
    eps: float = 1e-12,
) -> float:
    gamma_name = None if gamma is None else str(gamma).lower()
    if gamma is None or gamma_name in {"median", "minimum", "knn"}:
        X = np.asarray(X, dtype=float)
        if X.shape[0] < 2:
            return 1.0
        diff = X[:, None, :] - X[None, :, :]
        d2 = np.sum(diff * diff, axis=2)
        if gamma_name == "knn":
            k = min(max(int(bandwidth_knn_k), 1), X.shape[0] - 1)
            d2_knn = d2.copy()
            np.fill_diagonal(d2_knn, np.inf)
            kth = np.partition(d2_knn, kth=k - 1, axis=1)[:, k - 1]
            positive = kth[np.isfinite(kth) & (kth > eps)]
            if positive.size == 0:
                return 1.0
            scale = np.median(positive)
        else:
            pairwise = d2[np.triu_indices(X.shape[0], k=1)]
            pairwise = pairwise[pairwise > eps]
            if pairwise.size == 0:
                return 1.0
            scale = np.min(pairwise) if gamma_name == "minimum" else np.median(pairwise)
        return float(1.0 / (2.0 * max(scale, eps)))
    try:
        gamma = float(gamma)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "gamma must be positive, None, 'median', 'minimum', 'knn', "
            "'self_tuning', or 'local_scale'."
        ) from exc
    if gamma <= 0:
        raise ValueError(
            "gamma must be positive, None, 'median', 'minimum', 'knn', "
            "'self_tuning', or 'local_scale'."
        )
    return gamma


def _resolve_local_scales(
    points: np.ndarray,
    reference: np.ndarray,
    *,
    bandwidth_knn_k: int = 10,
    eps: float = 1e-12,
) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    reference = np.asarray(reference, dtype=float)
    if points.ndim != 2 or reference.ndim != 2:
        raise ValueError("points and reference must be 2D arrays.")
    if points.shape[1] != reference.shape[1]:
        raise ValueError("points and reference must have the same feature dimension.")
    if points.shape[0] == 0:
        return np.empty(0, dtype=float)
    if reference.shape[0] == 0:
        return np.ones(points.shape[0], dtype=float)

    k = max(int(bandwidth_knn_k), 1)
    min_positive = float(np.sqrt(eps))
    n_ref = reference.shape[0]
    nn = NearestNeighbors(metric="euclidean")
    nn.fit(reference)

    fallback_scale = 1.0
    if n_ref > 1:
        n_ref_neighbors = min(n_ref, 2)
        while True:
            ref_dist = nn.kneighbors(
                reference,
                n_neighbors=n_ref_neighbors,
                return_distance=True,
            )[0]
            positive = ref_dist[ref_dist > min_positive]
            if positive.size or n_ref_neighbors == n_ref:
                break
            n_ref_neighbors = min(n_ref, max(n_ref_neighbors + 1, 2 * n_ref_neighbors))
        if positive.size:
            fallback_scale = float(np.median(positive))

    scales = np.full(points.shape[0], np.nan, dtype=float)
    n_neighbors = min(n_ref, max(1, min(n_ref, k + 1)))
    while True:
        distances = nn.kneighbors(
            points,
            n_neighbors=n_neighbors,
            return_distance=True,
        )[0]
        positive = distances > min_positive
        for i in np.flatnonzero(np.isnan(scales)):
            row_positive = distances[i, positive[i]]
            if row_positive.size >= k:
                scales[i] = row_positive[k - 1]
            elif n_neighbors == n_ref and row_positive.size > 0:
                scales[i] = row_positive[-1]
            elif n_neighbors == n_ref:
                scales[i] = fallback_scale
        if np.all(np.isfinite(scales)):
            break
        if n_neighbors == n_ref:
            scales[~np.isfinite(scales)] = fallback_scale
            break
        n_neighbors = min(n_ref, max(n_neighbors + 1, 2 * n_neighbors))

    return np.maximum(scales, min_positive)


def _pairwise_kernel(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    kernel: str,
    gamma=None,
    normalize_embeddings: bool = True,
    gamma_reference_embeddings: np.ndarray | None = None,
    bandwidth_knn_k: int = 10,
    eps: float = 1e-12,
) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError("Kernel inputs must be 2D arrays.")
    if X.shape[1] != Y.shape[1]:
        raise ValueError("Kernel inputs must have the same feature dimension.")

    kernel = str(kernel).lower()
    if kernel == "cosine":
        Xn = _l2_normalize(X, eps=eps)
        Yn = _l2_normalize(Y, eps=eps)
        K = 0.5 * (1.0 + Xn @ Yn.T)
        return np.clip(K, 0.0, 1.0)
    if kernel == "rbf":
        Xr = _l2_normalize(X, eps=eps) if normalize_embeddings else X
        Yr = _l2_normalize(Y, eps=eps) if normalize_embeddings else Y
        gamma_name = None if gamma is None else str(gamma).lower()
        gamma_ref = X if X.shape[0] >= 2 else Y
        if gamma_reference_embeddings is not None:
            gamma_ref = np.asarray(gamma_reference_embeddings, dtype=float)
            if gamma_ref.ndim != 2:
                raise ValueError("gamma_reference_embeddings must be a 2D array.")
            if gamma_ref.shape[1] != X.shape[1]:
                raise ValueError(
                    "gamma_reference_embeddings must have the same feature dimension "
                    "as the kernel inputs."
                )
        gamma_ref = _l2_normalize(gamma_ref, eps=eps) if normalize_embeddings else gamma_ref
        x2 = np.sum(Xr * Xr, axis=1)[:, None]
        y2 = np.sum(Yr * Yr, axis=1)[None, :]
        d2 = np.maximum(x2 + y2 - 2.0 * (Xr @ Yr.T), 0.0)
        if gamma_name in {"self_tuning", "local_scale"}:
            scale_x = _resolve_local_scales(
                Xr,
                gamma_ref,
                bandwidth_knn_k=bandwidth_knn_k,
                eps=eps,
            )
            scale_y = _resolve_local_scales(
                Yr,
                gamma_ref,
                bandwidth_knn_k=bandwidth_knn_k,
                eps=eps,
            )
            denom = np.maximum(scale_x[:, None] * scale_y[None, :], eps)
            return np.exp(-d2 / denom)
        gamma_value = _resolve_gamma_from_embeddings(
            gamma_ref,
            gamma,
            bandwidth_knn_k=bandwidth_knn_k,
            eps=eps,
        )
        return np.exp(-float(gamma_value) * d2)
    raise ValueError("kernel must be one of {'rbf', 'cosine'}.")


def _apply_top_k_sample_support(
    Kx: np.ndarray,
    source_indices: np.ndarray,
    top_k: int | None,
    weighting: str = "kernel",
) -> np.ndarray:
    weighting = str(weighting).lower()
    if weighting not in {"kernel", "constant"}:
        raise ValueError("local_kernel_weighting must be one of {'kernel', 'constant'}.")
    Kx = np.asarray(Kx, dtype=float)
    source_indices = np.asarray(source_indices, dtype=int)
    if Kx.ndim != 2:
        raise ValueError("Kx must be a 2D kernel matrix.")
    if source_indices.shape != (Kx.shape[0],):
        raise ValueError("source_indices must have one entry per kernel row.")
    if Kx.shape[0] == 0 or Kx.shape[1] == 0:
        return Kx
    if top_k is None:
        return np.ones_like(Kx, dtype=float) if weighting == "constant" else Kx
    top_k = int(top_k)
    if top_k <= 0:
        raise ValueError("local_kernel_top_k must be > 0.")
    unique_source, inverse = np.unique(source_indices, return_inverse=True)
    n_unique = unique_source.size
    if top_k >= n_unique:
        return np.ones_like(Kx, dtype=float) if weighting == "constant" else Kx
    scores = np.full((n_unique, Kx.shape[1]), -np.inf, dtype=float)
    np.maximum.at(scores, inverse, Kx)
    keep_unique = np.zeros_like(scores, dtype=bool)
    kth = n_unique - top_k
    for j in range(Kx.shape[1]):
        keep = np.argpartition(scores[:, j], kth=kth)[kth:]
        keep_unique[keep, j] = True
    keep_rows = keep_unique[inverse]
    if weighting == "constant":
        return keep_rows.astype(float)
    return Kx * keep_rows


def compute_budget_aware_k_and_prior_strength(
    N,
    M,
    B_total,
    B_t,
    T0=10,
    rho=1.0,
    k_min=20,
    k_max=500,
    s_min=5,
    s_max=50,
    actual_local_evidence=None,
    eps=1e-12,
) -> BudgetAwareLocalityResult:
    N = int(N)
    M = int(M)
    B_total = float(B_total)
    B_t = float(B_t)
    T0 = float(T0)
    rho = float(rho)
    k_min = int(k_min)
    k_max = int(k_max)
    s_min = float(s_min)
    s_max = float(s_max)
    eps = float(eps)
    if N <= 0:
        raise ValueError("N must be > 0.")
    if M <= 0:
        raise ValueError("M must be > 0.")
    if B_total <= 0:
        raise ValueError("B_total must be > 0.")
    if B_t < 0:
        raise ValueError("B_t must be >= 0.")
    if T0 <= 0:
        raise ValueError("T0 must be > 0.")
    if rho < 0:
        raise ValueError("rho must be >= 0.")
    if k_min <= 0 or k_max <= 0:
        raise ValueError("k_min and k_max must be > 0.")
    if k_min > k_max:
        raise ValueError("k_min must be <= k_max.")
    if s_min < 0 or s_max < 0:
        raise ValueError("s_min and s_max must be >= 0.")
    if s_min > s_max:
        raise ValueError("s_min must be <= s_max.")

    evidence_scale = T0 * N * M
    k_final_raw = int(np.ceil(evidence_scale / max(B_total, eps)))
    k_current_raw = int(np.ceil(evidence_scale / max(B_t, eps)))
    k_upper = min(k_max, N)
    lower = min(max(k_final_raw, k_min), k_upper)
    k_t = int(np.clip(k_current_raw, lower, k_upper))
    k_final = int(np.clip(k_final_raw, 1, k_upper))
    T_expected_t = float(k_t * B_t / max(N * M, eps))

    prior_source = "expected"
    T_ref_t = T_expected_t
    if actual_local_evidence is not None:
        evidence = np.asarray(actual_local_evidence, dtype=float)
        valid = evidence[np.isfinite(evidence) & (evidence >= 0.0)]
        if valid.size > 0:
            T_ref_t = float(np.median(valid))
            prior_source = "actual"
    s_local = float(np.clip(rho * T_ref_t, s_min, s_max))
    feasibility_ratio = float(k_final_raw / max(N, eps))
    diagnostics = {
        "T_expected_t": T_expected_t,
        "T_ref_t": float(T_ref_t),
        "prior_source": prior_source,
        "k_final_raw": int(k_final_raw),
        "k_current_raw": int(k_current_raw),
        "k_final_over_N": feasibility_ratio,
        "k_t_over_N": float(k_t / max(N, eps)),
        "k_t_clipped_lower": bool(k_t > k_current_raw),
        "k_t_clipped_upper": bool(k_t < k_current_raw),
        "local_modeling_feasible": bool(feasibility_ratio <= 0.2),
    }
    return BudgetAwareLocalityResult(
        k_t=k_t,
        k_final=k_final,
        s_local=s_local,
        diagnostics=diagnostics,
    )


def _compute_evidence_weights(
    P: np.ndarray,
    method: str,
    *,
    eps: float = 1e-12,
) -> np.ndarray:
    P = _normalize_axis(np.clip(P, eps, 1.0), axis=1, eps=eps)
    method = str(method).lower()
    if method == "none":
        return np.ones(P.shape[0], dtype=float)
    if method == "entropy":
        K = P.shape[1]
        if K < 2:
            raise ValueError("entropy evidence weights require at least 2 classes.")
        H = -np.sum(P * np.log(P), axis=1)
        return np.clip(1.0 - H / np.log(K), 0.0, 1.0)
    if method == "margin":
        if P.shape[1] < 2:
            raise ValueError("margin evidence weights require at least 2 classes.")
        top2 = np.partition(P, kth=-2, axis=1)[:, -2:]
        top2.sort(axis=1)
        return np.clip(top2[:, 1] - top2[:, 0], 0.0, 1.0)
    raise ValueError("evidence_weight must be one of {'none', 'entropy', 'margin'}.")


def _make_observation_mask(
    y: np.ndarray,
    *,
    missing_label,
    indices=None,
    mask=None,
) -> np.ndarray:
    y = np.asarray(y)
    allowed = ~_is_missing(y, missing_label)
    if indices is not None:
        row_allowed = np.zeros(y.shape[0], dtype=bool)
        row_allowed[np.asarray(indices, dtype=int)] = True
        allowed &= row_allowed[:, None]
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != y.shape:
            raise ValueError(f"observation mask must have shape {y.shape}, got {mask.shape}.")
        allowed &= mask
    return allowed


def _resolve_candidates(y: np.ndarray, candidates=None, sample_indices=None) -> np.ndarray:
    if candidates is not None and sample_indices is not None:
        raise ValueError("Use only one of candidates or sample_indices.")
    selector = sample_indices if sample_indices is not None else candidates
    if selector is None:
        return np.arange(np.asarray(y).shape[0], dtype=int)
    selector = np.asarray(selector)
    if selector.dtype == bool:
        if selector.ndim != 1 or selector.shape[0] != np.asarray(y).shape[0]:
            raise ValueError("Boolean candidates must have shape (n_samples,).")
        return np.flatnonzero(selector)
    if selector.ndim != 1:
        raise ValueError("candidates/sample_indices must be a 1D selector.")
    return selector.astype(int, copy=False)


def _resolve_annotators(
    y: np.ndarray,
    *,
    sample_indices: np.ndarray,
    annotators=None,
    annotator_indices=None,
    available_mask=None,
) -> tuple[np.ndarray, np.ndarray | None]:
    if annotators is not None and annotator_indices is not None:
        raise ValueError("Use only one of annotators or annotator_indices.")
    selector = annotator_indices if annotator_indices is not None else annotators
    y = np.asarray(y)
    resolved_available = None if available_mask is None else np.asarray(available_mask, dtype=bool)
    if selector is None:
        resolved = np.arange(y.shape[1], dtype=int)
    else:
        selector = np.asarray(selector)
        if selector.dtype == bool and selector.ndim == 2:
            if selector.shape != y.shape:
                raise ValueError(
                    "2D boolean annotator availability must have shape "
                    f"{y.shape}, got {selector.shape}."
                )
            resolved = np.flatnonzero(selector[np.asarray(sample_indices, dtype=int)].any(axis=0))
            matrix_mask = selector[np.ix_(sample_indices, resolved)]
            resolved_available = (
                matrix_mask
                if resolved_available is None
                else (resolved_available & matrix_mask)
            )
            return resolved, resolved_available
        if selector.dtype == bool:
            if selector.ndim != 1 or selector.shape[0] != y.shape[1]:
                raise ValueError("Boolean annotators must have shape (n_annotators,).")
            resolved = np.flatnonzero(selector)
        else:
            if selector.ndim != 1:
                raise ValueError("annotators/annotator_indices must be a 1D selector or 2D mask.")
            resolved = selector.astype(int, copy=False)
    return resolved, resolved_available


def _labels_to_indices(
    y: np.ndarray,
    classes: np.ndarray,
    *,
    missing_label,
) -> np.ndarray:
    y = np.asarray(y)
    out = np.full(y.shape, -1, dtype=int)
    missing = _is_missing(y, missing_label)
    class_to_idx = {label: idx for idx, label in enumerate(classes)}
    rows, cols = np.where(~missing)
    for i, m in zip(rows, cols):
        label = y[i, m]
        try:
            out[i, m] = class_to_idx[label]
        except KeyError as exc:
            raise ValueError(f"Observed label {label!r} not found in clf.classes_.") from exc
    return out


def _compute_global_soft_counts(
    P: np.ndarray,
    y_idx: np.ndarray,
    observation_mask: np.ndarray,
    weights: np.ndarray,
    *,
    n_annotators: int,
    n_classes: int,
    annotator_similarity: np.ndarray | None = None,
) -> np.ndarray:
    counts = np.zeros((n_annotators, n_classes, n_classes), dtype=float)
    obs_s, obs_a = np.where(observation_mask)
    if obs_s.size == 0:
        return counts
    obs_y = y_idx[obs_s, obs_a]
    if annotator_similarity is not None:
        annotator_similarity = np.asarray(annotator_similarity, dtype=float)
        if annotator_similarity.shape != (n_annotators, n_annotators):
            raise ValueError(
                "annotator_similarity must have shape "
                f"{(n_annotators, n_annotators)}, got {annotator_similarity.shape}."
            )
    for target_m in range(n_annotators):
        if annotator_similarity is None:
            src = obs_a == target_m
            scale = weights[obs_s[src]]
            src_s = obs_s[src]
            src_y = obs_y[src]
        else:
            sim = annotator_similarity[obs_a, target_m]
            src = sim > 0
            scale = weights[obs_s[src]] * sim[src]
            src_s = obs_s[src]
            src_y = obs_y[src]
        if src_s.size == 0:
            continue
        for c in range(n_classes):
            take = src_y == c
            if np.any(take):
                counts[target_m, :, c] = P[src_s[take]].T @ scale[take]
    return counts


def _compute_local_kernel_soft_counts(
    P: np.ndarray,
    y_idx: np.ndarray,
    observation_mask: np.ndarray,
    weights: np.ndarray,
    *,
    sample_embeddings: np.ndarray,
    candidate_indices: np.ndarray,
    annotator_indices: np.ndarray,
    n_annotators: int,
    n_classes: int,
    kernel: str,
    gamma,
    normalize_embeddings: bool,
    gamma_reference_embeddings: np.ndarray | None = None,
    bandwidth_knn_k: int = 10,
    local_kernel_top_k: int | None = None,
    local_kernel_weighting: str = "kernel",
    annotator_similarity: np.ndarray | None = None,
    eps: float = 1e-12,
) -> np.ndarray:
    candidate_indices = np.asarray(candidate_indices, dtype=int)
    annotator_indices = np.asarray(annotator_indices, dtype=int)
    counts = np.zeros(
        (len(candidate_indices), len(annotator_indices), n_classes, n_classes),
        dtype=float,
    )
    obs_s, obs_a = np.where(observation_mask)
    if obs_s.size == 0 or len(candidate_indices) == 0:
        return counts
    obs_y = y_idx[obs_s, obs_a]
    Kx = _pairwise_kernel(
        sample_embeddings[obs_s],
        sample_embeddings[candidate_indices],
        kernel=kernel,
        gamma=gamma,
        normalize_embeddings=normalize_embeddings,
        gamma_reference_embeddings=gamma_reference_embeddings,
        bandwidth_knn_k=bandwidth_knn_k,
        eps=eps,
    )
    Kx = _apply_top_k_sample_support(Kx, obs_s, local_kernel_top_k, local_kernel_weighting)
    if annotator_similarity is not None:
        annotator_similarity = np.asarray(annotator_similarity, dtype=float)
        if annotator_similarity.shape != (n_annotators, n_annotators):
            raise ValueError(
                "annotator_similarity must have shape "
                f"{(n_annotators, n_annotators)}, got {annotator_similarity.shape}."
            )
    for j, target_m in enumerate(annotator_indices):
        if annotator_similarity is None:
            sim = (obs_a == target_m).astype(float)
        else:
            sim = annotator_similarity[obs_a, target_m]
        scale = weights[obs_s] * sim
        if not np.any(scale > 0):
            continue
        V = Kx * scale[:, None]
        for c in range(n_classes):
            take = obs_y == c
            if np.any(take):
                counts[:, j, :, c] = V[take].T @ P[obs_s[take]]
    return counts


def _compute_local_kernel_balanced_accuracy(
    P: np.ndarray,
    y_idx: np.ndarray,
    observation_mask: np.ndarray,
    *,
    sample_embeddings: np.ndarray,
    candidate_indices: np.ndarray,
    annotator_indices: np.ndarray,
    prior_diag: np.ndarray,
    prior_strength: float,
    n_classes: int,
    kernel: str,
    gamma,
    normalize_embeddings: bool,
    gamma_reference_embeddings: np.ndarray | None = None,
    bandwidth_knn_k: int = 10,
    local_kernel_top_k: int | None = None,
    local_kernel_weighting: str = "kernel",
    eps: float = 1e-12,
) -> LocalBalancedAccuracyResult:
    candidate_indices = np.asarray(candidate_indices, dtype=int)
    annotator_indices = np.asarray(annotator_indices, dtype=int)
    prior_diag = np.asarray(prior_diag, dtype=float)
    expected_prior_shape = (len(annotator_indices), n_classes)
    if prior_diag.shape != expected_prior_shape:
        raise ValueError(
            "prior_diag must have shape "
            f"{expected_prior_shape}, got {prior_diag.shape}."
        )
    if prior_strength <= 0:
        raise ValueError("prior_strength must be > 0.")
    shape = (len(candidate_indices), len(annotator_indices), n_classes)
    total = np.zeros(shape, dtype=float)
    correct = np.zeros(shape, dtype=float)
    obs_s, obs_a = np.where(observation_mask)
    if obs_s.size > 0 and len(candidate_indices) > 0:
        obs_y = y_idx[obs_s, obs_a]
        Kx = _pairwise_kernel(
            sample_embeddings[obs_s],
            sample_embeddings[candidate_indices],
            kernel=kernel,
            gamma=gamma,
            normalize_embeddings=normalize_embeddings,
            gamma_reference_embeddings=gamma_reference_embeddings,
            bandwidth_knn_k=bandwidth_knn_k,
            eps=eps,
        )
        Kx = _apply_top_k_sample_support(Kx, obs_s, local_kernel_top_k, local_kernel_weighting)
        for j, target_m in enumerate(annotator_indices):
            take_annotator = obs_a == target_m
            if not np.any(take_annotator):
                continue
            src_s = obs_s[take_annotator]
            src_y = obs_y[take_annotator]
            V = Kx[take_annotator]
            P_src = P[src_s]
            total[:, j, :] = V.T @ P_src
            y_onehot = np.zeros_like(P_src, dtype=float)
            y_onehot[np.arange(src_y.size), src_y] = 1.0
            correct[:, j, :] = V.T @ (P_src * y_onehot)
    theta = (
        prior_strength * prior_diag[None, :, :] + correct
    ) / np.maximum(prior_strength + total, eps)
    balanced_accuracy = np.mean(theta, axis=2)
    chance = 1.0 / n_classes
    corrected = (balanced_accuracy - chance) / np.maximum(1.0 - chance, eps)
    corrected = np.maximum(corrected, 0.0)
    return LocalBalancedAccuracyResult(
        theta=theta,
        balanced_accuracy=balanced_accuracy,
        corrected_balanced_accuracy=corrected,
        total=total,
        correct=correct,
        prior_diag=prior_diag,
        prior_strength=float(prior_strength),
    )


def _class_prior_alpha0_vector(alpha0, n_classes: int) -> np.ndarray:
    alpha0 = np.asarray(alpha0, dtype=float)
    if alpha0.ndim == 0:
        alpha0 = np.full(n_classes, float(alpha0), dtype=float)
    if alpha0.shape != (n_classes,) or np.any(alpha0 <= 0):
        raise ValueError("class_prior_alpha0 must be a positive scalar or length-K vector.")
    return alpha0


def _mace_channel_from_theta_g(
    theta: np.ndarray,
    g: np.ndarray,
    *,
    n_classes: int,
    eps: float = 1e-12,
) -> np.ndarray:
    theta = np.asarray(theta, dtype=float)
    g = _normalize_axis(np.asarray(g, dtype=float), axis=-1, eps=eps)
    eye = np.eye(n_classes, dtype=float)
    C = theta[..., None, None] * eye + (1.0 - theta)[..., None, None] * g[..., None, :]
    return _normalize_axis(C, axis=-1, eps=eps)


def _accuracy_uniform_channel_from_theta(
    theta: np.ndarray,
    *,
    n_classes: int,
    eps: float = 1e-12,
) -> np.ndarray:
    if n_classes < 2:
        raise ValueError("accuracy_uniform channels require at least 2 classes.")
    theta = np.asarray(theta, dtype=float)
    C = np.full((*theta.shape, n_classes, n_classes), 0.0, dtype=float)
    off_diag = (1.0 - theta) / (n_classes - 1)
    C[...] = off_diag[..., None, None]
    diag = np.arange(n_classes)
    C[..., diag, diag] = theta[..., None]
    return _normalize_axis(C, axis=-1, eps=eps)


def _classwise_accuracy_channel_from_theta(
    theta: np.ndarray,
    *,
    n_classes: int,
    eps: float = 1e-12,
) -> np.ndarray:
    if n_classes < 2:
        raise ValueError("class-wise accuracy channels require at least 2 classes.")
    theta = np.clip(np.asarray(theta, dtype=float), 0.0, 1.0)
    if theta.shape[-1] != n_classes:
        raise ValueError(
            "theta must have one entry per class on its last axis; "
            f"expected {n_classes}, got {theta.shape[-1]}."
        )
    C = np.broadcast_to(
        ((1.0 - theta) / (n_classes - 1))[..., :, None],
        (*theta.shape, n_classes),
    ).copy()
    diag = np.arange(n_classes)
    C[..., diag, diag] = theta
    return _normalize_axis(C, axis=-1, eps=eps)


def _full_channel_to_accuracy_uniform_params(
    B: np.ndarray,
    *,
    eps: float = 1e-12,
) -> AccuracyUniformChannelParameters:
    B = _normalize_axis(np.clip(np.asarray(B, dtype=float), eps, None), axis=-1, eps=eps)
    theta = np.mean(np.diagonal(B, axis1=-2, axis2=-1), axis=-1)
    theta = np.clip(theta, eps, 1.0 - eps)
    return AccuracyUniformChannelParameters(theta=theta)


def _full_channel_to_mace_params(
    B: np.ndarray,
    *,
    eps: float = 1e-12,
) -> MaceChannelParameters:
    B = _normalize_axis(np.clip(np.asarray(B, dtype=float), eps, None), axis=-1, eps=eps)
    n_classes = B.shape[-1]
    theta = np.mean(np.diagonal(B, axis1=-2, axis2=-1), axis=-1)
    theta = np.clip(theta, eps, 1.0 - eps)
    off_diag = B.copy()
    diag = np.arange(n_classes)
    off_diag[..., diag, diag] = 0.0
    g = off_diag.sum(axis=-2)
    fallback = B.mean(axis=-2)
    g_sum = g.sum(axis=-1, keepdims=True)
    g = np.where(g_sum > eps, g / np.maximum(g_sum, eps), fallback)
    g = _normalize_axis(np.clip(g, eps, None), axis=-1, eps=eps)
    return MaceChannelParameters(theta=theta, g=g)


def _compute_global_accuracy_counts(
    P: np.ndarray,
    y_idx: np.ndarray,
    observation_mask: np.ndarray,
    weights: np.ndarray,
    *,
    n_annotators: int,
    annotator_similarity: np.ndarray | None = None,
) -> AccuracyEvidenceCounts:
    success = np.zeros(n_annotators, dtype=float)
    failure = np.zeros(n_annotators, dtype=float)
    obs_s, obs_a = np.where(observation_mask)
    if obs_s.size == 0:
        return AccuracyEvidenceCounts(success, failure)
    obs_y = y_idx[obs_s, obs_a]
    q_y = P[obs_s, obs_y]
    if annotator_similarity is not None:
        annotator_similarity = np.asarray(annotator_similarity, dtype=float)
        if annotator_similarity.shape != (n_annotators, n_annotators):
            raise ValueError(
                "annotator_similarity must have shape "
                f"{(n_annotators, n_annotators)}, got {annotator_similarity.shape}."
            )
    for target_m in range(n_annotators):
        if annotator_similarity is None:
            sim = (obs_a == target_m).astype(float)
        else:
            sim = annotator_similarity[obs_a, target_m]
        scale = weights[obs_s] * sim
        if not np.any(scale > 0):
            continue
        success[target_m] = np.sum(scale * q_y)
        failure[target_m] = np.sum(scale * (1.0 - q_y))
    return AccuracyEvidenceCounts(success, failure)


def _compute_local_kernel_accuracy_counts(
    P: np.ndarray,
    y_idx: np.ndarray,
    observation_mask: np.ndarray,
    weights: np.ndarray,
    *,
    sample_embeddings: np.ndarray,
    candidate_indices: np.ndarray,
    annotator_indices: np.ndarray,
    n_annotators: int,
    kernel: str,
    gamma,
    normalize_embeddings: bool,
    gamma_reference_embeddings: np.ndarray | None = None,
    bandwidth_knn_k: int = 10,
    local_kernel_top_k: int | None = None,
    local_kernel_weighting: str = "kernel",
    annotator_similarity: np.ndarray | None = None,
    eps: float = 1e-12,
) -> AccuracyEvidenceCounts:
    candidate_indices = np.asarray(candidate_indices, dtype=int)
    annotator_indices = np.asarray(annotator_indices, dtype=int)
    shape = (len(candidate_indices), len(annotator_indices))
    success = np.zeros(shape, dtype=float)
    failure = np.zeros(shape, dtype=float)
    obs_s, obs_a = np.where(observation_mask)
    if obs_s.size == 0 or len(candidate_indices) == 0:
        return AccuracyEvidenceCounts(success, failure)
    obs_y = y_idx[obs_s, obs_a]
    q_y = P[obs_s, obs_y]
    Kx = _pairwise_kernel(
        sample_embeddings[obs_s],
        sample_embeddings[candidate_indices],
        kernel=kernel,
        gamma=gamma,
        normalize_embeddings=normalize_embeddings,
        gamma_reference_embeddings=gamma_reference_embeddings,
        bandwidth_knn_k=bandwidth_knn_k,
        eps=eps,
    )
    Kx = _apply_top_k_sample_support(Kx, obs_s, local_kernel_top_k, local_kernel_weighting)
    if annotator_similarity is not None:
        annotator_similarity = np.asarray(annotator_similarity, dtype=float)
        if annotator_similarity.shape != (n_annotators, n_annotators):
            raise ValueError(
                "annotator_similarity must have shape "
                f"{(n_annotators, n_annotators)}, got {annotator_similarity.shape}."
            )
    for j, target_m in enumerate(annotator_indices):
        if annotator_similarity is None:
            sim = (obs_a == target_m).astype(float)
        else:
            sim = annotator_similarity[obs_a, target_m]
        scale = weights[obs_s] * sim
        if not np.any(scale > 0):
            continue
        V = Kx * scale[:, None]
        success[:, j] = V.T @ q_y
        failure[:, j] = V.T @ (1.0 - q_y)
    return AccuracyEvidenceCounts(success, failure)


def _accuracy_params_from_counts(
    counts: AccuracyEvidenceCounts,
    *,
    alpha_prior: np.ndarray,
    beta_prior: np.ndarray,
    eps: float = 1e-12,
) -> AccuracyUniformChannelParameters:
    alpha = np.asarray(alpha_prior, dtype=float) + counts.success
    beta = np.asarray(beta_prior, dtype=float) + counts.failure
    theta = alpha / np.maximum(alpha + beta, eps)
    return AccuracyUniformChannelParameters(theta=theta, alpha=alpha, beta=beta)


def _estimate_global_accuracy_uniform_parameters(
    P: np.ndarray,
    y_idx: np.ndarray,
    observation_mask: np.ndarray,
    weights: np.ndarray,
    *,
    n_annotators: int,
    n_classes: int,
    fixed_prior_accuracy,
    fixed_prior_strength: float,
    annotator_similarity: np.ndarray | None = None,
    eps: float = 1e-12,
) -> AccuracyUniformChannelParameters:
    accuracy = _resolve_fixed_prior_accuracy(fixed_prior_accuracy, n_classes)
    strength = _resolve_fixed_prior_strength(fixed_prior_strength)
    counts = _compute_global_accuracy_counts(
        P,
        y_idx,
        observation_mask,
        weights,
        n_annotators=n_annotators,
        annotator_similarity=annotator_similarity,
    )
    return _accuracy_params_from_counts(
        counts,
        alpha_prior=np.full(n_annotators, strength * accuracy, dtype=float),
        beta_prior=np.full(n_annotators, strength * (1.0 - accuracy), dtype=float),
        eps=eps,
    )


def _estimate_local_accuracy_uniform_parameters(
    P: np.ndarray,
    y_idx: np.ndarray,
    observation_mask: np.ndarray,
    weights: np.ndarray,
    *,
    sample_embeddings: np.ndarray,
    candidate_indices: np.ndarray,
    annotator_indices: np.ndarray,
    n_annotators: int,
    n_classes: int,
    fixed_prior_accuracy,
    fixed_prior_strength: float,
    kernel: str,
    gamma,
    normalize_embeddings: bool,
    gamma_reference_embeddings: np.ndarray | None = None,
    bandwidth_knn_k: int = 10,
    local_kernel_top_k: int | None = None,
    local_kernel_weighting: str = "kernel",
    annotator_similarity: np.ndarray | None = None,
    eps: float = 1e-12,
) -> AccuracyUniformChannelParameters:
    accuracy = _resolve_fixed_prior_accuracy(fixed_prior_accuracy, n_classes)
    strength = _resolve_fixed_prior_strength(fixed_prior_strength)
    counts = _compute_local_kernel_accuracy_counts(
        P,
        y_idx,
        observation_mask,
        weights,
        sample_embeddings=sample_embeddings,
        candidate_indices=candidate_indices,
        annotator_indices=annotator_indices,
        n_annotators=n_annotators,
        kernel=kernel,
        gamma=gamma,
        normalize_embeddings=normalize_embeddings,
        gamma_reference_embeddings=gamma_reference_embeddings,
        bandwidth_knn_k=bandwidth_knn_k,
        local_kernel_top_k=local_kernel_top_k,
        local_kernel_weighting=local_kernel_weighting,
        annotator_similarity=annotator_similarity,
        eps=eps,
    )
    shape = (len(candidate_indices), len(annotator_indices))
    return _accuracy_params_from_counts(
        counts,
        alpha_prior=np.full(shape, strength * accuracy, dtype=float),
        beta_prior=np.full(shape, strength * (1.0 - accuracy), dtype=float),
        eps=eps,
    )


def _compute_global_mace_counts(
    P: np.ndarray,
    y_idx: np.ndarray,
    observation_mask: np.ndarray,
    weights: np.ndarray,
    *,
    n_annotators: int,
    n_classes: int,
    annotator_similarity: np.ndarray | None = None,
) -> MaceEvidenceCounts:
    success = np.zeros(n_annotators, dtype=float)
    failure = np.zeros(n_annotators, dtype=float)
    g_counts = np.zeros((n_annotators, n_classes), dtype=float)
    obs_s, obs_a = np.where(observation_mask)
    if obs_s.size == 0:
        return MaceEvidenceCounts(success, failure, g_counts)
    obs_y = y_idx[obs_s, obs_a]
    q_y = P[obs_s, obs_y]
    if annotator_similarity is not None:
        annotator_similarity = np.asarray(annotator_similarity, dtype=float)
        if annotator_similarity.shape != (n_annotators, n_annotators):
            raise ValueError(
                "annotator_similarity must have shape "
                f"{(n_annotators, n_annotators)}, got {annotator_similarity.shape}."
            )
    for target_m in range(n_annotators):
        if annotator_similarity is None:
            sim = (obs_a == target_m).astype(float)
        else:
            sim = annotator_similarity[obs_a, target_m]
        scale = weights[obs_s] * sim
        if not np.any(scale > 0):
            continue
        residual = scale * (1.0 - q_y)
        success[target_m] = np.sum(scale * q_y)
        failure[target_m] = np.sum(residual)
        for c in range(n_classes):
            take = obs_y == c
            if np.any(take):
                g_counts[target_m, c] = np.sum(residual[take])
    return MaceEvidenceCounts(success, failure, g_counts)


def _compute_local_kernel_mace_counts(
    P: np.ndarray,
    y_idx: np.ndarray,
    observation_mask: np.ndarray,
    weights: np.ndarray,
    *,
    sample_embeddings: np.ndarray,
    candidate_indices: np.ndarray,
    annotator_indices: np.ndarray,
    n_annotators: int,
    n_classes: int,
    kernel: str,
    gamma,
    normalize_embeddings: bool,
    gamma_reference_embeddings: np.ndarray | None = None,
    bandwidth_knn_k: int = 10,
    local_kernel_top_k: int | None = None,
    local_kernel_weighting: str = "kernel",
    annotator_similarity: np.ndarray | None = None,
    eps: float = 1e-12,
) -> MaceEvidenceCounts:
    candidate_indices = np.asarray(candidate_indices, dtype=int)
    annotator_indices = np.asarray(annotator_indices, dtype=int)
    shape = (len(candidate_indices), len(annotator_indices))
    success = np.zeros(shape, dtype=float)
    failure = np.zeros(shape, dtype=float)
    g_counts = np.zeros((*shape, n_classes), dtype=float)
    obs_s, obs_a = np.where(observation_mask)
    if obs_s.size == 0 or len(candidate_indices) == 0:
        return MaceEvidenceCounts(success, failure, g_counts)
    obs_y = y_idx[obs_s, obs_a]
    q_y = P[obs_s, obs_y]
    Kx = _pairwise_kernel(
        sample_embeddings[obs_s],
        sample_embeddings[candidate_indices],
        kernel=kernel,
        gamma=gamma,
        normalize_embeddings=normalize_embeddings,
        gamma_reference_embeddings=gamma_reference_embeddings,
        bandwidth_knn_k=bandwidth_knn_k,
        eps=eps,
    )
    Kx = _apply_top_k_sample_support(Kx, obs_s, local_kernel_top_k, local_kernel_weighting)
    if annotator_similarity is not None:
        annotator_similarity = np.asarray(annotator_similarity, dtype=float)
        if annotator_similarity.shape != (n_annotators, n_annotators):
            raise ValueError(
                "annotator_similarity must have shape "
                f"{(n_annotators, n_annotators)}, got {annotator_similarity.shape}."
            )
    for j, target_m in enumerate(annotator_indices):
        if annotator_similarity is None:
            sim = (obs_a == target_m).astype(float)
        else:
            sim = annotator_similarity[obs_a, target_m]
        scale = weights[obs_s] * sim
        if not np.any(scale > 0):
            continue
        V = Kx * scale[:, None]
        success[:, j] = V.T @ q_y
        residual = V * (1.0 - q_y)[:, None]
        failure[:, j] = residual.sum(axis=0)
        for c in range(n_classes):
            take = obs_y == c
            if np.any(take):
                g_counts[:, j, c] = residual[take].sum(axis=0)
    return MaceEvidenceCounts(success, failure, g_counts)


def _mace_params_from_counts(
    counts: MaceEvidenceCounts,
    *,
    theta_success_prior: np.ndarray,
    theta_failure_prior: np.ndarray,
    g_prior: np.ndarray,
    eps: float = 1e-12,
) -> MaceChannelParameters:
    theta_success_alpha = np.asarray(theta_success_prior, dtype=float) + counts.theta_success
    theta_failure_beta = np.asarray(theta_failure_prior, dtype=float) + counts.theta_failure
    g_alpha = np.asarray(g_prior, dtype=float) + counts.g_counts
    theta = theta_success_alpha / np.maximum(
        theta_success_alpha + theta_failure_beta,
        eps,
    )
    g = _normalize_axis(g_alpha, axis=-1, eps=eps)
    return MaceChannelParameters(
        theta=theta,
        g=g,
        theta_success_alpha=theta_success_alpha,
        theta_failure_beta=theta_failure_beta,
        g_alpha=g_alpha,
    )


def _estimate_global_mace_parameters(
    P: np.ndarray,
    y_idx: np.ndarray,
    observation_mask: np.ndarray,
    weights: np.ndarray,
    *,
    n_annotators: int,
    n_classes: int,
    fixed_prior_accuracy,
    fixed_prior_strength: float,
    annotator_similarity: np.ndarray | None = None,
    eps: float = 1e-12,
) -> MaceChannelParameters:
    accuracy = _resolve_fixed_prior_accuracy(fixed_prior_accuracy, n_classes)
    strength = _resolve_fixed_prior_strength(fixed_prior_strength)
    theta = _fixed_mace_theta_from_accuracy(accuracy, n_classes)
    g_prior = np.full(n_classes, strength / n_classes, dtype=float)
    counts = _compute_global_mace_counts(
        P,
        y_idx,
        observation_mask,
        weights,
        n_annotators=n_annotators,
        n_classes=n_classes,
        annotator_similarity=annotator_similarity,
    )
    return _mace_params_from_counts(
        counts,
        theta_success_prior=np.full(n_annotators, strength * theta, dtype=float),
        theta_failure_prior=np.full(n_annotators, strength * (1.0 - theta), dtype=float),
        g_prior=np.broadcast_to(g_prior[None, :], (n_annotators, n_classes)),
        eps=eps,
    )


def _estimate_local_mace_parameters(
    P: np.ndarray,
    y_idx: np.ndarray,
    observation_mask: np.ndarray,
    weights: np.ndarray,
    *,
    sample_embeddings: np.ndarray,
    candidate_indices: np.ndarray,
    annotator_indices: np.ndarray,
    n_annotators: int,
    n_classes: int,
    fixed_prior_accuracy,
    fixed_prior_strength: float,
    kernel: str,
    gamma,
    normalize_embeddings: bool,
    gamma_reference_embeddings: np.ndarray | None = None,
    bandwidth_knn_k: int = 10,
    local_kernel_top_k: int | None = None,
    local_kernel_weighting: str = "kernel",
    annotator_similarity: np.ndarray | None = None,
    eps: float = 1e-12,
) -> MaceChannelParameters:
    accuracy = _resolve_fixed_prior_accuracy(fixed_prior_accuracy, n_classes)
    strength = _resolve_fixed_prior_strength(fixed_prior_strength)
    theta = _fixed_mace_theta_from_accuracy(accuracy, n_classes)
    g_prior = np.full(n_classes, strength / n_classes, dtype=float)
    counts = _compute_local_kernel_mace_counts(
        P,
        y_idx,
        observation_mask,
        weights,
        sample_embeddings=sample_embeddings,
        candidate_indices=candidate_indices,
        annotator_indices=annotator_indices,
        n_annotators=n_annotators,
        n_classes=n_classes,
        kernel=kernel,
        gamma=gamma,
        normalize_embeddings=normalize_embeddings,
        gamma_reference_embeddings=gamma_reference_embeddings,
        bandwidth_knn_k=bandwidth_knn_k,
        local_kernel_top_k=local_kernel_top_k,
        local_kernel_weighting=local_kernel_weighting,
        annotator_similarity=annotator_similarity,
        eps=eps,
    )
    shape = (len(candidate_indices), len(annotator_indices))
    return _mace_params_from_counts(
        counts,
        theta_success_prior=np.full(shape, strength * theta, dtype=float),
        theta_failure_prior=np.full(shape, strength * (1.0 - theta), dtype=float),
        g_prior=np.broadcast_to(g_prior, (*shape, n_classes)),
        eps=eps,
    )


def _estimate_base_uniform(n_annotators: int, n_classes: int) -> np.ndarray:
    return np.full((n_annotators, n_classes, n_classes), 1.0 / n_classes, dtype=float)


def _resolve_fixed_prior_accuracy(value, n_classes: int) -> float:
    if n_classes < 2:
        raise ValueError("fixed priors require at least 2 classes.")
    chance = 1.0 / n_classes
    if value is None:
        return chance
    accuracy = float(value)
    if accuracy < chance or accuracy > 1.0:
        raise ValueError(
            "fixed_prior_accuracy must be None or a scalar in "
            f"[1 / n_classes, 1], got {accuracy} for n_classes={n_classes}."
        )
    return accuracy


def _resolve_fixed_prior_strength(value) -> float:
    strength = float(value)
    if strength <= 0:
        raise ValueError("fixed_prior_strength must be > 0.")
    return strength


def _fixed_mace_theta_from_accuracy(accuracy: float, n_classes: int) -> float:
    return (n_classes * accuracy - 1.0) / (n_classes - 1.0)


def _fixed_full_channel(
    *,
    n_annotators: int,
    n_classes: int,
    fixed_prior_accuracy,
    eps: float = 1e-12,
) -> np.ndarray:
    accuracy = _resolve_fixed_prior_accuracy(fixed_prior_accuracy, n_classes)
    theta = np.full(n_annotators, accuracy, dtype=float)
    return _accuracy_uniform_channel_from_theta(theta, n_classes=n_classes, eps=eps)


def _fixed_mace_parameters(
    *,
    n_annotators: int,
    n_classes: int,
    fixed_prior_accuracy,
) -> MaceChannelParameters:
    accuracy = _resolve_fixed_prior_accuracy(fixed_prior_accuracy, n_classes)
    theta = _fixed_mace_theta_from_accuracy(accuracy, n_classes)
    g = np.full((n_annotators, n_classes), 1.0 / n_classes, dtype=float)
    return MaceChannelParameters(
        theta=np.full(n_annotators, theta, dtype=float),
        g=g,
    )


def _fixed_accuracy_uniform_parameters(
    *,
    n_annotators: int,
    n_classes: int,
    fixed_prior_accuracy,
) -> AccuracyUniformChannelParameters:
    accuracy = _resolve_fixed_prior_accuracy(fixed_prior_accuracy, n_classes)
    theta = np.full(n_annotators, accuracy, dtype=float)
    return AccuracyUniformChannelParameters(theta=theta)


def _estimate_base_mace(
    P: np.ndarray,
    y_idx: np.ndarray,
    observation_mask: np.ndarray,
    weights: np.ndarray,
    *,
    n_annotators: int,
    n_classes: int,
    fixed_prior_accuracy,
    fixed_prior_strength: float,
    eps: float = 1e-12,
) -> np.ndarray:
    params = _estimate_global_mace_parameters(
        P,
        y_idx,
        observation_mask,
        weights,
        n_annotators=n_annotators,
        n_classes=n_classes,
        fixed_prior_accuracy=fixed_prior_accuracy,
        fixed_prior_strength=fixed_prior_strength,
        eps=eps,
    )
    return _mace_channel_from_theta_g(
        params.theta,
        params.g,
        n_classes=n_classes,
        eps=eps,
    )


def _estimate_base_diag_uniform(
    P: np.ndarray,
    y_idx: np.ndarray,
    observation_mask: np.ndarray,
    weights: np.ndarray,
    *,
    n_annotators: int,
    n_classes: int,
    theta_prior: tuple[float, float],
    eps: float = 1e-12,
) -> np.ndarray:
    if n_classes < 2:
        raise ValueError("diag_uniform requires at least 2 classes.")
    a, b = map(float, theta_prior)
    if a <= 0 or b <= 0:
        raise ValueError("diag_theta_prior entries must be positive.")
    B = np.empty((n_annotators, n_classes, n_classes), dtype=float)
    for m in range(n_annotators):
        rows = np.flatnonzero(observation_mask[:, m])
        theta = np.full(n_classes, a / (a + b), dtype=float)
        if rows.size > 0:
            labels = y_idx[rows, m]
            w = weights[rows]
            for z in range(n_classes):
                responsibility = w * P[rows, z]
                correct = labels == z
                theta[z] = (
                    a + np.sum(responsibility[correct])
                ) / (
                    a + b + np.sum(responsibility)
                )
        B[m] = (1.0 - theta[:, None]) / (n_classes - 1)
        B[m, np.arange(n_classes), np.arange(n_classes)] = theta
    return _normalize_axis(B, axis=2, eps=eps)


def _estimate_base_global_full(
    P: np.ndarray,
    y_idx: np.ndarray,
    observation_mask: np.ndarray,
    weights: np.ndarray,
    *,
    n_annotators: int,
    n_classes: int,
    fixed_prior_accuracy,
    fixed_prior_strength: float,
    eps: float = 1e-12,
) -> np.ndarray:
    strength = _resolve_fixed_prior_strength(fixed_prior_strength)
    counts = _compute_global_soft_counts(
        P,
        y_idx,
        observation_mask,
        weights,
        n_annotators=n_annotators,
        n_classes=n_classes,
    )
    prior = strength * _fixed_full_channel(
        n_annotators=n_annotators,
        n_classes=n_classes,
        fixed_prior_accuracy=fixed_prior_accuracy,
        eps=eps,
    )
    return _normalize_axis(counts + prior, axis=2, eps=eps)


def _combine_base_only(
    B: np.ndarray,
    *,
    n_candidates: int,
    annotator_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray | None]:
    if B.ndim == 4:
        expected = (n_candidates, len(annotator_indices))
        if B.shape[:2] != expected:
            raise ValueError(
                "local base channel must have leading shape "
                f"{expected}, got {B.shape[:2]}."
            )
        return B.copy(), None
    C = np.broadcast_to(
        B[np.asarray(annotator_indices, dtype=int)][None, :, :, :],
        (n_candidates, len(annotator_indices), B.shape[1], B.shape[2]),
    ).copy()
    return C, None


def _combine_dirichlet_global(
    B: np.ndarray,
    N: np.ndarray,
    *,
    prior_strength: float,
    n_candidates: int,
    annotator_indices: np.ndarray,
    eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    alpha_global = prior_strength * B + N
    C_global = _normalize_axis(alpha_global, axis=2, eps=eps)
    annotator_indices = np.asarray(annotator_indices, dtype=int)
    alpha = np.broadcast_to(
        alpha_global[annotator_indices][None, :, :, :],
        (n_candidates, len(annotator_indices), B.shape[1], B.shape[2]),
    ).copy()
    C = np.broadcast_to(
        C_global[annotator_indices][None, :, :, :],
        alpha.shape,
    ).copy()
    return C, alpha


def _combine_dirichlet_local(
    B: np.ndarray,
    N_local: np.ndarray,
    *,
    prior_strength: float,
    annotator_indices: np.ndarray,
    eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    if B.ndim == 4:
        if B.shape[:2] != N_local.shape[:2]:
            raise ValueError(
                "local base channel and local evidence counts must have matching "
                f"candidate/annotator axes, got {B.shape[:2]} and {N_local.shape[:2]}."
            )
        B_sel = B
    else:
        B_sel = B[np.asarray(annotator_indices, dtype=int)]
    alpha = prior_strength * B_sel + N_local if B.ndim == 4 else prior_strength * B_sel[None, :, :, :] + N_local
    C = _normalize_axis(alpha, axis=3, eps=eps)
    return C, alpha


def _broadcast_global_mace_params(
    params: MaceChannelParameters,
    *,
    n_candidates: int,
    annotator_indices: np.ndarray,
) -> MaceChannelParameters:
    annotator_indices = np.asarray(annotator_indices, dtype=int)
    theta = np.broadcast_to(
        params.theta[annotator_indices][None, :],
        (n_candidates, len(annotator_indices)),
    ).copy()
    g = np.broadcast_to(
        params.g[annotator_indices][None, :, :],
        (n_candidates, len(annotator_indices), params.g.shape[-1]),
    ).copy()

    def _maybe_broadcast(arr):
        if arr is None:
            return None
        arr = np.asarray(arr, dtype=float)
        if arr.ndim == 1:
            return np.broadcast_to(
                arr[annotator_indices][None, :],
                theta.shape,
            ).copy()
        return np.broadcast_to(
            arr[annotator_indices][None, :, :],
            g.shape,
        ).copy()

    return MaceChannelParameters(
        theta=theta,
        g=g,
        theta_success_alpha=_maybe_broadcast(params.theta_success_alpha),
        theta_failure_beta=_maybe_broadcast(params.theta_failure_beta),
        g_alpha=_maybe_broadcast(params.g_alpha),
    )


def _combine_mace_global(
    base_params: MaceChannelParameters,
    counts: MaceEvidenceCounts,
    *,
    prior_strength: float,
    n_candidates: int,
    annotator_indices: np.ndarray,
    n_classes: int,
    eps: float = 1e-12,
) -> tuple[np.ndarray, MaceChannelParameters]:
    if base_params.theta.ndim != 1:
        raise ValueError("posterior='global_mace' requires global MACE base parameters.")
    posterior_params_global = _mace_params_from_counts(
        counts,
        theta_success_prior=prior_strength * base_params.theta,
        theta_failure_prior=prior_strength * (1.0 - base_params.theta),
        g_prior=prior_strength * base_params.g,
        eps=eps,
    )
    selected_params = _broadcast_global_mace_params(
        posterior_params_global,
        n_candidates=n_candidates,
        annotator_indices=annotator_indices,
    )
    C = _mace_channel_from_theta_g(
        selected_params.theta,
        selected_params.g,
        n_classes=n_classes,
        eps=eps,
    )
    return C, selected_params


def _combine_mace_local(
    base_params: MaceChannelParameters,
    counts: MaceEvidenceCounts,
    *,
    prior_strength: float,
    n_candidates: int,
    annotator_indices: np.ndarray,
    n_classes: int,
    eps: float = 1e-12,
) -> tuple[np.ndarray, MaceChannelParameters]:
    if base_params.theta.ndim == 1:
        base_params = _broadcast_global_mace_params(
            base_params,
            n_candidates=n_candidates,
            annotator_indices=annotator_indices,
        )
    elif base_params.theta.shape != counts.theta_success.shape:
        raise ValueError(
            "posterior='local_mace' requires local MACE base parameters with "
            f"shape {counts.theta_success.shape}, got {base_params.theta.shape}."
        )
    posterior_params = _mace_params_from_counts(
        counts,
        theta_success_prior=prior_strength * base_params.theta,
        theta_failure_prior=prior_strength * (1.0 - base_params.theta),
        g_prior=prior_strength * base_params.g,
        eps=eps,
    )
    C = _mace_channel_from_theta_g(
        posterior_params.theta,
        posterior_params.g,
        n_classes=n_classes,
        eps=eps,
    )
    return C, posterior_params


def _broadcast_global_accuracy_params(
    params: AccuracyUniformChannelParameters,
    *,
    n_candidates: int,
    annotator_indices: np.ndarray,
) -> AccuracyUniformChannelParameters:
    annotator_indices = np.asarray(annotator_indices, dtype=int)
    theta = np.broadcast_to(
        params.theta[annotator_indices][None, :],
        (n_candidates, len(annotator_indices)),
    ).copy()

    def _maybe_broadcast(arr):
        if arr is None:
            return None
        return np.broadcast_to(
            np.asarray(arr, dtype=float)[annotator_indices][None, :],
            theta.shape,
        ).copy()

    return AccuracyUniformChannelParameters(
        theta=theta,
        alpha=_maybe_broadcast(params.alpha),
        beta=_maybe_broadcast(params.beta),
    )


def _combine_accuracy_uniform_global(
    base_params: AccuracyUniformChannelParameters,
    counts: AccuracyEvidenceCounts,
    *,
    prior_strength: float,
    n_candidates: int,
    annotator_indices: np.ndarray,
    n_classes: int,
    eps: float = 1e-12,
) -> tuple[np.ndarray, AccuracyUniformChannelParameters]:
    if base_params.theta.ndim != 1:
        raise ValueError(
            "posterior='global_accuracy_uniform' requires global accuracy-uniform base parameters."
        )
    posterior_params_global = _accuracy_params_from_counts(
        counts,
        alpha_prior=prior_strength * base_params.theta,
        beta_prior=prior_strength * (1.0 - base_params.theta),
        eps=eps,
    )
    selected_params = _broadcast_global_accuracy_params(
        posterior_params_global,
        n_candidates=n_candidates,
        annotator_indices=annotator_indices,
    )
    C = _accuracy_uniform_channel_from_theta(
        selected_params.theta,
        n_classes=n_classes,
        eps=eps,
    )
    return C, selected_params


def _combine_accuracy_uniform_local(
    base_params: AccuracyUniformChannelParameters,
    counts: AccuracyEvidenceCounts,
    *,
    prior_strength: float,
    n_candidates: int,
    annotator_indices: np.ndarray,
    n_classes: int,
    eps: float = 1e-12,
) -> tuple[np.ndarray, AccuracyUniformChannelParameters]:
    if base_params.theta.ndim == 1:
        base_params = _broadcast_global_accuracy_params(
            base_params,
            n_candidates=n_candidates,
            annotator_indices=annotator_indices,
        )
    elif base_params.theta.shape != counts.success.shape:
        raise ValueError(
            "posterior='local_accuracy_uniform' requires local accuracy-uniform "
            f"base parameters with shape {counts.success.shape}, got {base_params.theta.shape}."
        )
    posterior_params = _accuracy_params_from_counts(
        counts,
        alpha_prior=prior_strength * base_params.theta,
        beta_prior=prior_strength * (1.0 - base_params.theta),
        eps=eps,
    )
    C = _accuracy_uniform_channel_from_theta(
        posterior_params.theta,
        n_classes=n_classes,
        eps=eps,
    )
    return C, posterior_params


def _compute_expected_accuracy(p: np.ndarray, C: np.ndarray) -> np.ndarray:
    diag = np.diagonal(C, axis1=2, axis2=3)
    return np.sum(p[:, None, :] * diag, axis=2)


def _compute_bias_corrected_accuracy(
    p: np.ndarray,
    C: np.ndarray,
    *,
    eps: float = 1e-12,
) -> np.ndarray:
    A = _compute_expected_accuracy(p, C)
    g = np.mean(C, axis=2)
    baseline = np.sum(p[:, None, :] * g, axis=2)
    return (A - baseline) / np.maximum(1.0 - baseline, eps)


def _compute_information_gain(
    p: np.ndarray,
    C: np.ndarray,
    *,
    eps: float = 1e-12,
) -> np.ndarray:
    p = _normalize_axis(np.clip(p, eps, 1.0), axis=1, eps=eps)
    C = _normalize_axis(np.clip(C, eps, 1.0), axis=3, eps=eps)
    p_y = np.sum(p[:, None, :, None] * C, axis=2)
    log_ratio = np.log(C) - np.log(np.maximum(p_y[:, :, None, :], eps))
    return np.sum(p[:, None, :, None] * C * log_ratio, axis=(2, 3))


def _compute_instance_difficulty_gate(
    p: np.ndarray,
    *,
    power: float = 1.0,
    eps: float = 1e-12,
) -> np.ndarray:
    p = _normalize_axis(np.clip(p, eps, 1.0), axis=1, eps=eps)
    K = p.shape[1]
    if K < 2:
        raise ValueError("instance difficulty gating requires at least 2 classes.")
    entropy = -np.sum(p * np.log(p), axis=1)
    gate = 1.0 - entropy / np.log(K)
    gate = np.clip(gate, 0.0, 1.0)
    return gate ** float(power)


def _apply_instance_difficulty_gate(
    p: np.ndarray,
    C: np.ndarray,
    *,
    power: float = 1.0,
    eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    gate = _compute_instance_difficulty_gate(p, power=power, eps=eps)
    C = _normalize_axis(np.clip(C, eps, 1.0), axis=3, eps=eps)
    guess = np.mean(C, axis=2)
    C_guess = np.broadcast_to(guess[:, :, None, :], C.shape)
    C_gated = gate[:, None, None, None] * C + (
        1.0 - gate[:, None, None, None]
    ) * C_guess
    return _normalize_axis(C_gated, axis=3, eps=eps), gate


def _compute_difficulty_gated_information_gain(
    p: np.ndarray,
    C: np.ndarray,
    *,
    power: float = 1.0,
    eps: float = 1e-12,
) -> np.ndarray:
    C_gated, _ = _apply_instance_difficulty_gate(p, C, power=power, eps=eps)
    return _compute_information_gain(p, C_gated, eps=eps)


class KernelSmoothedBayesianAnnotatorGainNew(PairScorer):
    """Preset-driven post-hoc annotator-channel scorer.

    The scorer estimates annotator response channels from frozen classifier
    probabilities and observed noisy labels. Channel estimation is separated
    into a global base channel ``B``, optional global/local evidence counts,
    and a Dirichlet posterior combiner.
    """

    def __init__(
        self,
        *,
        preset: str | None = "local_full_global_mace_prior",
        base_channel: str | None = None,
        evidence: str | None = None,
        posterior: str | None = None,
        utility: str = "information_gain",
        class_prior: str = "kernel",
        top_m: int | None = None,
        prior_strength: float = 1.0,
        prior_observations: str = "same",
        prior_indices=None,
        evidence_indices=None,
        initial_indices=None,
        prior_mask=None,
        evidence_mask=None,
        initial_observation_mask=None,
        evidence_weight: str = "none",
        kernel: str = "rbf",
        gamma="median",
        bandwidth_reference: str | int | float = "labeled",
        bandwidth_reference_sample: int | float | None = None,
        bandwidth_knn_k: int = 10,
        local_kernel_top_k: int | None = None,
        local_kernel_weighting: str = "kernel",
        budget_aware_locality: bool = False,
        budget_total: int | float | None = None,
        budget_T0: float = 10.0,
        budget_rho: float = 1.0,
        budget_k_min: int = 20,
        budget_k_max: int = 500,
        budget_s_min: float = 5.0,
        budget_s_max: float = 50.0,
        embedding_source: str = "classifier",
        normalize_embeddings: bool = True,
        use_annotator_embeddings: bool = False,
        annotator_kernel: str = "rbf",
        annotator_gamma=None,
        diag_theta_prior=(1.0, 1.0),
        fixed_prior_accuracy: float | None = None,
        fixed_prior_strength: float = 1.0,
        class_prior_alpha0=1.0,
        class_prior_kernel: str | None = "rbf",
        class_prior_gamma="minimum",
        class_prior_support: str = "observed",
        class_prior_evidence_weight: str = "none",
        observed_class_prior: str = "kernel",
        observed_class_prior_kernel: str | None = None,
        observed_class_prior_gamma=None,
        observed_class_prior_support: str = "observed",
        observed_class_prior_leave_one_out: bool = False,
        laplace_prior_precision: float = 1.0,
        laplace_include_bias: bool = True,
        laplace_predictive_samples: int = 32,
        laplace_variance_scale: float = 1.0,
        difficulty_gate_power: float = 1.0,
        n_mc_samples: int = 0,
        sample_class_prior: bool = False,
        sample_channel: bool = False,
        utility_aggregation: str = "mean",
        utility_quantile: float | None = None,
        store_utility_draws: bool = False,
        eps: float = 1e-12,
        missing_label=None,
        random_state=None,
    ):
        self.preset = None if preset is None else str(preset)
        preset_cfg = self._apply_preset(self.preset)
        self.base_channel = str(
            base_channel if base_channel is not None else preset_cfg.get("base_channel")
        )
        if self.base_channel == "mace":
            self.base_channel = "global_mace"
        if self.base_channel == "local_ba":
            self.base_channel = "local_balanced_accuracy"
        self.evidence = str(evidence if evidence is not None else preset_cfg.get("evidence"))
        self.posterior = str(posterior if posterior is not None else preset_cfg.get("posterior"))
        self.utility = str(utility)
        self.class_prior = str(class_prior)
        self.top_m = None if top_m is None else int(top_m)
        self.prior_strength = float(prior_strength)
        self.prior_observations = str(prior_observations)
        self.prior_indices = prior_indices
        self.evidence_indices = evidence_indices
        self.initial_indices = initial_indices
        self.prior_mask = prior_mask
        self.evidence_mask = evidence_mask
        self.initial_observation_mask = initial_observation_mask
        if (
            bandwidth_reference_sample is None
            and isinstance(bandwidth_reference, (int, float))
            and not isinstance(bandwidth_reference, bool)
        ):
            bandwidth_reference_sample = bandwidth_reference
            bandwidth_reference = "all"
        self.evidence_weight = str(evidence_weight)
        self.kernel = str(kernel)
        self.gamma = gamma
        self.bandwidth_reference = str(bandwidth_reference)
        self.bandwidth_reference_sample = bandwidth_reference_sample
        self.bandwidth_knn_k = int(bandwidth_knn_k)
        self.local_kernel_top_k = (
            None if local_kernel_top_k is None else int(local_kernel_top_k)
        )
        self.local_kernel_weighting = str(local_kernel_weighting)
        self.budget_aware_locality = bool(budget_aware_locality)
        self.budget_total = None if budget_total is None else float(budget_total)
        self.budget_T0 = float(budget_T0)
        self.budget_rho = float(budget_rho)
        self.budget_k_min = int(budget_k_min)
        self.budget_k_max = int(budget_k_max)
        self.budget_s_min = float(budget_s_min)
        self.budget_s_max = float(budget_s_max)
        self.embedding_source = str(embedding_source)
        self.normalize_embeddings = bool(normalize_embeddings)
        self.use_annotator_embeddings = bool(use_annotator_embeddings)
        self.annotator_kernel = str(annotator_kernel)
        self.annotator_gamma = annotator_gamma
        self.diag_theta_prior = tuple(diag_theta_prior)
        self.fixed_prior_accuracy = (
            None if fixed_prior_accuracy is None else float(fixed_prior_accuracy)
        )
        self.fixed_prior_strength = float(fixed_prior_strength)
        self.class_prior_alpha0 = class_prior_alpha0
        self.class_prior_kernel = class_prior_kernel
        self.class_prior_gamma = class_prior_gamma
        self.class_prior_support = str(class_prior_support)
        self.class_prior_evidence_weight = str(class_prior_evidence_weight)
        self.observed_class_prior = str(observed_class_prior)
        self.observed_class_prior_kernel = observed_class_prior_kernel
        self.observed_class_prior_gamma = observed_class_prior_gamma
        self.observed_class_prior_support = str(observed_class_prior_support)
        self.observed_class_prior_leave_one_out = bool(observed_class_prior_leave_one_out)
        self.laplace_prior_precision = float(laplace_prior_precision)
        self.laplace_include_bias = bool(laplace_include_bias)
        self.laplace_predictive_samples = int(laplace_predictive_samples)
        self.laplace_variance_scale = float(laplace_variance_scale)
        self.difficulty_gate_power = float(difficulty_gate_power)
        self.n_mc_samples = int(n_mc_samples)
        self.sample_class_prior = bool(sample_class_prior)
        self.sample_channel = bool(sample_channel)
        self.utility_aggregation = str(utility_aggregation)
        self.utility_quantile = None if utility_quantile is None else float(utility_quantile)
        self.store_utility_draws = bool(store_utility_draws)
        self.eps = float(eps)
        self.missing_label = missing_label
        self.random_state = check_random_state(random_state)

        self._validate_combination()
        self._clear_diagnostics()

    def __call__(
        self,
        X,
        y,
        *,
        sample_indices=None,
        annotator_indices=None,
        candidates=None,
        annotators=None,
        available_mask=None,
        **kwargs,
    ):
        sample_indices = _resolve_candidates(
            y,
            candidates=candidates,
            sample_indices=sample_indices,
        )
        annotator_indices, available_mask = _resolve_annotators(
            y,
            sample_indices=sample_indices,
            annotators=annotators,
            annotator_indices=annotator_indices,
            available_mask=available_mask,
        )
        return super().__call__(
            X,
            y,
            sample_indices=sample_indices,
            annotator_indices=annotator_indices,
            available_mask=available_mask,
            **kwargs,
        )

    @staticmethod
    def _apply_preset(preset: str | None) -> dict[str, str]:
        if preset is None:
            return {}
        if preset not in _PRESETS:
            raise ValueError(f"Unknown preset={preset!r}. Expected one of {sorted(_PRESETS)}.")
        return dict(_PRESETS[preset])

    def _validate_combination(self):
        if self.base_channel not in {
            "uniform",
            "fixed_full",
            "fixed_mace",
            "global_mace",
            "local_mace",
            "global_accuracy_uniform",
            "fixed_accuracy_uniform",
            "local_accuracy_uniform",
            "diag_uniform",
            "global_full",
            "local_balanced_accuracy",
        }:
            raise ValueError(
                "base_channel must be one of "
                "{'uniform', 'fixed_full', 'fixed_mace', "
                "'global_mace', 'local_mace', "
                "'global_accuracy_uniform', 'fixed_accuracy_uniform', "
                "'local_accuracy_uniform', "
                "'diag_uniform', 'global_full', 'local_balanced_accuracy'}."
            )
        if self.evidence not in {
            "none",
            "global_soft_counts",
            "local_kernel_soft_counts",
            "global_mace_counts",
            "local_kernel_mace_counts",
            "global_accuracy_counts",
            "local_kernel_accuracy_counts",
            "local_kernel_balanced_accuracy_counts",
        }:
            raise ValueError(
                "evidence must be one of "
                "{'none', 'global_soft_counts', 'local_kernel_soft_counts', "
                "'global_mace_counts', 'local_kernel_mace_counts', "
                "'global_accuracy_counts', 'local_kernel_accuracy_counts', "
                "'local_kernel_balanced_accuracy_counts'}."
            )
        if self.posterior not in {
            "base_only",
            "dirichlet_global",
            "dirichlet_local",
            "global_mace",
            "local_mace",
            "global_accuracy_uniform",
            "local_accuracy_uniform",
            "local_balanced_accuracy",
        }:
            raise ValueError(
                "posterior must be one of "
                "{'base_only', 'dirichlet_global', 'dirichlet_local', "
                "'global_mace', 'local_mace', "
                "'global_accuracy_uniform', 'local_accuracy_uniform', "
                "'local_balanced_accuracy'}."
            )
        if self.evidence == "none" and self.posterior != "base_only":
            raise ValueError("evidence='none' requires posterior='base_only'.")
        if self.posterior == "base_only" and self.evidence != "none":
            raise ValueError("posterior='base_only' requires evidence='none'.")
        if self.posterior == "dirichlet_global" and self.evidence != "global_soft_counts":
            raise ValueError(
                "posterior='dirichlet_global' requires evidence='global_soft_counts'."
            )
        if self.posterior == "dirichlet_local" and self.evidence != "local_kernel_soft_counts":
            raise ValueError(
                "posterior='dirichlet_local' requires evidence='local_kernel_soft_counts'."
            )
        if self.posterior == "global_mace" and self.evidence != "global_mace_counts":
            raise ValueError(
                "posterior='global_mace' requires evidence='global_mace_counts'."
            )
        if self.posterior == "local_mace" and self.evidence != "local_kernel_mace_counts":
            raise ValueError(
                "posterior='local_mace' requires evidence='local_kernel_mace_counts'."
            )
        if self.posterior == "global_accuracy_uniform" and self.evidence != "global_accuracy_counts":
            raise ValueError(
                "posterior='global_accuracy_uniform' requires evidence='global_accuracy_counts'."
            )
        if (
            self.posterior == "local_accuracy_uniform"
            and self.evidence != "local_kernel_accuracy_counts"
        ):
            raise ValueError(
                "posterior='local_accuracy_uniform' requires "
                "evidence='local_kernel_accuracy_counts'."
            )
        if (
            self.posterior == "local_balanced_accuracy"
            and self.evidence != "local_kernel_balanced_accuracy_counts"
        ):
            raise ValueError(
                "posterior='local_balanced_accuracy' requires "
                "evidence='local_kernel_balanced_accuracy_counts'."
            )
        if self.posterior in {"global_mace", "local_mace"} and self.base_channel not in {
            "fixed_mace",
            "global_mace",
            "global_full",
        }:
            raise ValueError(
                "MACE posteriors require base_channel in "
                "{'fixed_mace', 'global_mace', 'global_full'}."
            )
        if self.posterior in {
            "global_accuracy_uniform",
            "local_accuracy_uniform",
        } and self.base_channel not in {
            "uniform",
            "global_accuracy_uniform",
            "fixed_accuracy_uniform",
            "global_full",
        }:
            raise ValueError(
                "accuracy-uniform posteriors require base_channel in "
                "{'uniform', 'global_accuracy_uniform', 'fixed_accuracy_uniform', "
                "'global_full'}."
            )
        if self.posterior == "dirichlet_global" and self.base_channel in {
            "local_mace",
            "local_accuracy_uniform",
            "local_balanced_accuracy",
        }:
            raise ValueError("posterior='dirichlet_global' requires a global base_channel.")
        if self.posterior == "local_balanced_accuracy" and self.base_channel != "global_full":
            raise ValueError(
                "posterior='local_balanced_accuracy' requires base_channel='global_full'."
            )
        if self.prior_strength <= 0:
            raise ValueError("prior_strength must be > 0.")
        if self.fixed_prior_accuracy is not None and not (
            0.0 < self.fixed_prior_accuracy <= 1.0
        ):
            raise ValueError("fixed_prior_accuracy must be None or in (0, 1].")
        if self.fixed_prior_strength <= 0:
            raise ValueError("fixed_prior_strength must be > 0.")
        if self.prior_observations not in {"same", "separate", "none", "initial"}:
            raise ValueError(
                "prior_observations must be one of {'same', 'separate', 'none', 'initial'}."
            )
        if self.prior_observations == "none" and self.base_channel != "uniform":
            raise ValueError("prior_observations='none' currently requires base_channel='uniform'.")
        if self.utility not in {
            "expected_accuracy",
            "bias_corrected_accuracy",
            "information_gain",
            "difficulty_gated_information_gain",
            "local_balanced_accuracy",
            "local_corrected_balanced_accuracy",
            "local_ba",
            "local_cba",
        }:
            raise ValueError(
                "utility must be one of "
                "{'expected_accuracy', 'bias_corrected_accuracy', 'information_gain', "
                "'difficulty_gated_information_gain', 'local_balanced_accuracy', "
                "'local_corrected_balanced_accuracy', 'local_ba', 'local_cba'}."
            )
        if self.utility in {
            "local_balanced_accuracy",
            "local_corrected_balanced_accuracy",
            "local_ba",
            "local_cba",
        } and self.base_channel != "global_full":
            raise ValueError(
                "local balanced-accuracy utilities require base_channel='global_full'."
            )
        if self.class_prior not in {
            "classifier",
            "uniform",
            "top_m",
            "kernel",
            "evidence_shrunk",
            "laplace",
        }:
            raise ValueError(
                "class_prior must be one of "
                "{'classifier', 'uniform', 'top_m', 'kernel', "
                "'evidence_shrunk', 'laplace'}."
            )
        if self.class_prior == "top_m":
            if self.top_m is None:
                raise ValueError("class_prior='top_m' requires top_m.")
        elif self.top_m is not None:
            raise ValueError("top_m is only used when class_prior='top_m'.")
        if np.any(np.asarray(self.class_prior_alpha0, dtype=float) <= 0):
            raise ValueError("class_prior_alpha0 must be > 0.")
        if self.class_prior_support not in {"evidence", "observed"}:
            raise ValueError("class_prior_support must be one of {'evidence', 'observed'}.")
        if self.class_prior in {"kernel", "evidence_shrunk"} and self.class_prior_gamma is None:
            raise ValueError(
                f"class_prior={self.class_prior!r} requires explicit class_prior_gamma."
            )
        _compute_evidence_weights(
            np.full((2, 2), 0.5, dtype=float),
            self.class_prior_evidence_weight,
            eps=self.eps,
        )
        if self.observed_class_prior not in {
            "classifier",
            "kernel",
            "evidence_shrunk",
            "laplace",
        }:
            raise ValueError(
                "observed_class_prior must be one of "
                "{'classifier', 'kernel', 'evidence_shrunk', 'laplace'}."
            )
        if self.observed_class_prior_support not in {"observed", "prior", "evidence"}:
            raise ValueError(
                "observed_class_prior_support must be one of {'observed', 'prior', 'evidence'}."
            )
        if (
            self.observed_class_prior in {"kernel", "evidence_shrunk"}
            and self.observed_class_prior_gamma is None
            and self.class_prior_gamma is None
        ):
            raise ValueError(
                f"observed_class_prior={self.observed_class_prior!r} requires "
                "observed_class_prior_gamma or class_prior_gamma."
            )
        if (
            self.observed_class_prior_leave_one_out
            and self.observed_class_prior not in {"kernel", "evidence_shrunk", "laplace"}
        ):
            raise ValueError(
                "observed_class_prior_leave_one_out=True requires observed_class_prior "
                "in {'kernel', 'evidence_shrunk', 'laplace'}."
            )
        if self.laplace_prior_precision <= 0:
            raise ValueError("laplace_prior_precision must be > 0.")
        if self.laplace_predictive_samples <= 0:
            raise ValueError("laplace_predictive_samples must be > 0.")
        if self.laplace_variance_scale < 0:
            raise ValueError("laplace_variance_scale must be >= 0.")
        if self.difficulty_gate_power <= 0:
            raise ValueError("difficulty_gate_power must be > 0.")
        if self.bandwidth_reference not in {"labeled", "all"}:
            raise ValueError("bandwidth_reference must be one of {'labeled', 'all'}.")
        if self.bandwidth_reference_sample is not None:
            sample = self.bandwidth_reference_sample
            if isinstance(sample, bool):
                raise ValueError("bandwidth_reference_sample must be a positive int or a float in (0, 1].")
            if isinstance(sample, (int, np.integer)):
                if int(sample) <= 0:
                    raise ValueError("bandwidth_reference_sample integer must be > 0.")
            else:
                sample = float(sample)
                if not (0.0 < sample <= 1.0):
                    raise ValueError("bandwidth_reference_sample float must be in (0, 1].")
        if self.bandwidth_knn_k <= 0:
            raise ValueError("bandwidth_knn_k must be > 0.")
        if self.local_kernel_top_k is not None and self.local_kernel_top_k <= 0:
            raise ValueError("local_kernel_top_k must be None or > 0.")
        if self.local_kernel_weighting not in {"kernel", "constant"}:
            raise ValueError("local_kernel_weighting must be one of {'kernel', 'constant'}.")
        if self.budget_aware_locality:
            if self.budget_total is not None and self.budget_total <= 0:
                raise ValueError("budget_total must be None or > 0.")
            compute_budget_aware_k_and_prior_strength(
                N=1,
                M=1,
                B_total=1.0 if self.budget_total is None else self.budget_total,
                B_t=1.0,
                T0=self.budget_T0,
                rho=self.budget_rho,
                k_min=self.budget_k_min,
                k_max=self.budget_k_max,
                s_min=self.budget_s_min,
                s_max=self.budget_s_max,
                eps=self.eps,
            )
        if self.embedding_source not in {"classifier", "input"}:
            raise ValueError("embedding_source must be one of {'classifier', 'input'}.")
        if self.n_mc_samples < 0:
            raise ValueError("n_mc_samples must be >= 0.")
        if (self.sample_class_prior or self.sample_channel) and self.n_mc_samples <= 0:
            raise ValueError("Sampling requires n_mc_samples > 0.")
        if self.sample_class_prior and self.class_prior not in {
            "kernel",
            "evidence_shrunk",
            "laplace",
        }:
            raise ValueError(
                "sample_class_prior=True requires class_prior in "
                "{'kernel', 'evidence_shrunk', 'laplace'}."
            )
        if self.sample_channel and self.posterior == "base_only":
            raise ValueError("sample_channel=True requires a posterior distribution.")
        if self.utility_aggregation not in {"mean", "quantile"}:
            raise ValueError("utility_aggregation must be one of {'mean', 'quantile'}.")
        if self.utility_aggregation == "quantile":
            if self.n_mc_samples <= 0:
                raise ValueError("utility_aggregation='quantile' requires n_mc_samples > 0.")
            if self.utility_quantile is None or not (0.0 < self.utility_quantile < 1.0):
                raise ValueError("utility_quantile must be in (0, 1).")
        elif self.utility_quantile is not None:
            raise ValueError("utility_quantile is only used with utility_aggregation='quantile'.")

    def _clear_diagnostics(self):
        self.last_result_ = None
        self.last_base_channel_ = None
        self.last_base_mace_theta_ = None
        self.last_base_mace_g_ = None
        self.last_base_accuracy_theta_ = None
        self.last_evidence_counts_ = None
        self.last_evidence_mace_counts_ = None
        self.last_evidence_accuracy_counts_ = None
        self.last_posterior_alpha_ = None
        self.last_posterior_channel_ = None
        self.last_posterior_mace_theta_ = None
        self.last_posterior_mace_g_ = None
        self.last_posterior_accuracy_theta_ = None
        self.last_posterior_concentration_ = None
        self.last_classifier_proba_ = None
        self.last_observed_class_proba_ = None
        self.last_candidate_class_prior_ = None
        self.last_class_prior_ = None
        self.last_class_prior_alpha_ = None
        self.last_class_prior_laplace_logits_ = None
        self.last_class_prior_laplace_logit_variance_ = None
        self.last_observed_laplace_logit_variance_ = None
        self._last_class_prior_laplace_logits = None
        self._last_class_prior_laplace_logit_variance = None
        self._last_observed_laplace_logit_variance = None
        self.last_prior_mask_ = None
        self.last_evidence_mask_ = None
        self.last_prior_observations_ = None
        self.last_uses_same_prior_and_evidence_ = None
        self.last_utilities_ = None
        self.last_utility_mean_ = None
        self.last_utility_std_ = None
        self.last_utility_lcb_ = None
        self.last_utility_ucb_ = None
        self.last_utility_draws_ = None
        self.last_instance_difficulty_gate_ = None
        self.last_local_class_accuracy_ = None
        self.last_local_balanced_accuracy_ = None
        self.last_local_corrected_balanced_accuracy_ = None
        self.last_local_balanced_total_ = None
        self.last_local_balanced_correct_ = None
        self.last_budget_aware_locality_ = None
        self.last_local_kernel_top_k_ = None
        self.last_local_kernel_weighting_ = None
        self.last_local_prior_strength_ = None
        self._last_local_balanced_accuracy_result = None
        self._runtime_budget_aware_locality = None
        self._runtime_local_kernel_top_k = self.local_kernel_top_k
        self._runtime_local_prior_strength = self.prior_strength

    def _compute(
        self,
        X,
        y,
        sample_indices,
        annotator_indices,
        available_mask,
        clf=None,
        **kwargs,
    ):
        result = self.estimate_channels(
            X=X,
            y=y,
            sample_indices=sample_indices,
            annotator_indices=annotator_indices,
            clf=clf,
            **kwargs,
        )
        utilities = self._utilities_from_result(result)
        if available_mask is not None:
            utilities = np.where(np.asarray(available_mask, dtype=bool), utilities, np.nan)
        self._store_result_diagnostics(result, utilities)
        return utilities

    def _configure_budget_aware_locality(
        self,
        *,
        y,
        observed_mask,
        budget_total=None,
        budget_used=None,
    ) -> BudgetAwareLocalityResult | None:
        self._runtime_budget_aware_locality = None
        self._runtime_local_kernel_top_k = self.local_kernel_top_k
        self._runtime_local_prior_strength = self.prior_strength
        if not self.budget_aware_locality:
            return None
        resolved_budget_total = self.budget_total if budget_total is None else float(budget_total)
        if resolved_budget_total is None:
            raise ValueError(
                "budget_aware_locality=True requires budget_total in the scorer config "
                "or as a runtime scorer argument."
            )
        resolved_budget_used = (
            float(np.count_nonzero(observed_mask))
            if budget_used is None
            else float(budget_used)
        )
        result = compute_budget_aware_k_and_prior_strength(
            N=y.shape[0],
            M=y.shape[1],
            B_total=resolved_budget_total,
            B_t=resolved_budget_used,
            T0=self.budget_T0,
            rho=self.budget_rho,
            k_min=self.budget_k_min,
            k_max=self.budget_k_max,
            s_min=self.budget_s_min,
            s_max=self.budget_s_max,
            eps=self.eps,
        )
        self._runtime_budget_aware_locality = result
        self._runtime_local_kernel_top_k = result.k_t
        self._runtime_local_prior_strength = result.s_local
        return result

    def _local_kernel_top_k(self) -> int | None:
        return self._runtime_local_kernel_top_k

    def _local_prior_strength(self) -> float:
        return float(self._runtime_local_prior_strength)

    def estimate_channels(
        self,
        *,
        X,
        y,
        sample_indices=None,
        annotator_indices=None,
        clf=None,
        prior_indices=None,
        evidence_indices=None,
        initial_indices=None,
        prior_mask=None,
        evidence_mask=None,
        initial_observation_mask=None,
        rng=None,
        budget_total=None,
        budget_used=None,
        **kwargs,
    ) -> ChannelEstimationResult:
        if clf is None:
            raise ValueError("`clf` must be provided.")
        X = np.asarray(X)
        y = np.asarray(y)
        sample_indices = (
            np.arange(y.shape[0], dtype=int)
            if sample_indices is None
            else np.asarray(sample_indices, dtype=int)
        )
        annotator_indices = (
            np.arange(y.shape[1], dtype=int)
            if annotator_indices is None
            else np.asarray(annotator_indices, dtype=int)
        )
        if np.any(sample_indices < 0) or np.any(sample_indices >= y.shape[0]):
            raise ValueError("sample_indices contain out-of-bounds entries.")
        if np.any(annotator_indices < 0) or np.any(annotator_indices >= y.shape[1]):
            raise ValueError("annotator_indices contain out-of-bounds entries.")
        classes = np.asarray(clf.classes_)
        K = len(classes)
        if K < 2:
            raise ValueError("Channel utilities require at least 2 classes.")
        if self.class_prior == "top_m" and not (1 <= int(self.top_m) <= K):
            raise ValueError("top_m must be in [1, n_classes].")
        missing_label = self._resolve_missing_label(clf)
        (
            P_all,
            sample_embeddings,
            annotator_embeddings,
            logits_all,
        ) = self._predict_probabilities_and_embeddings(
            X=X,
            clf=clf,
            need_sample_embeddings=self._needs_sample_embeddings(),
            need_annotator_embeddings=self.use_annotator_embeddings,
            need_logits=self._needs_logits(),
        )
        P_balanced_accuracy = _normalize_axis(
            np.clip(P_all, 0.0, 1.0),
            axis=1,
            eps=self.eps,
        )
        P_all = _normalize_axis(np.clip(P_all, self.eps, 1.0), axis=1, eps=self.eps)
        y_idx = _labels_to_indices(y, classes, missing_label=missing_label)
        prior_mask_resolved, evidence_mask_resolved = self._resolve_prior_evidence_masks(
            y=y,
            missing_label=missing_label,
            prior_indices=prior_indices,
            evidence_indices=evidence_indices,
            initial_indices=initial_indices,
            prior_mask=prior_mask,
            evidence_mask=evidence_mask,
            initial_observation_mask=initial_observation_mask,
        )
        observed_mask = _make_observation_mask(y, missing_label=missing_label)
        budget_aware_result = self._configure_budget_aware_locality(
            y=y,
            observed_mask=observed_mask,
            budget_total=budget_total,
            budget_used=budget_used,
        )
        gamma_reference_embeddings = self._resolve_bandwidth_reference_embeddings(
            sample_embeddings=sample_embeddings,
            observed_mask=observed_mask,
        )
        P_channel = self._resolve_observed_class_probabilities(
            P=P_all,
            logits=logits_all,
            sample_embeddings=sample_embeddings,
            gamma_reference_embeddings=gamma_reference_embeddings,
            prior_mask=prior_mask_resolved,
            evidence_mask=evidence_mask_resolved,
            observed_mask=observed_mask,
            n_classes=K,
        )
        observed_laplace_logit_variance = self._last_observed_laplace_logit_variance
        weights = _compute_evidence_weights(P_channel, self.evidence_weight, eps=self.eps)
        annotator_similarity = self._resolve_annotator_similarity(
            annotator_embeddings,
            n_annotators=y.shape[1],
        )
        self._last_local_balanced_accuracy_result = None
        P_base = (
            P_balanced_accuracy
            if self.base_channel == "local_balanced_accuracy"
            else P_channel
        )
        B, base_mace_params, base_accuracy_params = self._estimate_base_channel(
            P=P_base,
            y_idx=y_idx,
            prior_mask=prior_mask_resolved,
            weights=weights,
            sample_embeddings=sample_embeddings,
            gamma_reference_embeddings=gamma_reference_embeddings,
            sample_indices=sample_indices,
            annotator_indices=annotator_indices,
            n_annotators=y.shape[1],
            n_classes=K,
            annotator_similarity=annotator_similarity,
        )
        local_ba_result = self._compute_local_balanced_accuracy_result(
            P=P_balanced_accuracy,
            y_idx=y_idx,
            evidence_mask=evidence_mask_resolved,
            sample_embeddings=sample_embeddings,
            gamma_reference_embeddings=gamma_reference_embeddings,
            sample_indices=sample_indices,
            annotator_indices=annotator_indices,
            B=B,
            n_classes=K,
        )
        evidence_counts = self._compute_evidence_counts(
            P=P_channel,
            y_idx=y_idx,
            evidence_mask=evidence_mask_resolved,
            weights=weights,
            sample_embeddings=sample_embeddings,
            gamma_reference_embeddings=gamma_reference_embeddings,
            sample_indices=sample_indices,
            annotator_indices=annotator_indices,
            n_annotators=y.shape[1],
            n_classes=K,
            annotator_similarity=annotator_similarity,
        )
        C, alpha, posterior_mace_params, posterior_accuracy_params = self._combine_posterior(
            B=B,
            base_mace_params=base_mace_params,
            base_accuracy_params=base_accuracy_params,
            evidence_counts=evidence_counts,
            sample_indices=sample_indices,
            annotator_indices=annotator_indices,
            n_classes=K,
        )
        p, class_alpha = self._resolve_candidate_class_prior(
            P=P_all,
            logits=logits_all,
            sample_embeddings=sample_embeddings,
            gamma_reference_embeddings=gamma_reference_embeddings,
            sample_indices=sample_indices,
            evidence_mask=evidence_mask_resolved,
            observed_mask=observed_mask,
            n_classes=K,
        )
        class_laplace_logits = self._last_class_prior_laplace_logits
        class_laplace_logit_variance = self._last_class_prior_laplace_logit_variance
        uses_same = np.array_equal(prior_mask_resolved, evidence_mask_resolved)
        result = ChannelEstimationResult(
            base_channel=B,
            evidence_counts=evidence_counts,
            posterior_alpha=alpha,
            posterior_channel=C,
            base_mace_params=base_mace_params,
            evidence_mace_counts=(
                evidence_counts if isinstance(evidence_counts, MaceEvidenceCounts) else None
            ),
            posterior_mace_params=posterior_mace_params,
            base_accuracy_params=base_accuracy_params,
            evidence_accuracy_counts=(
                evidence_counts if isinstance(evidence_counts, AccuracyEvidenceCounts) else None
            ),
            posterior_accuracy_params=posterior_accuracy_params,
            classifier_proba=P_all,
            observed_class_proba=P_channel,
            class_prior=p,
            class_prior_alpha=class_alpha,
            class_prior_laplace_logits=class_laplace_logits,
            class_prior_laplace_logit_variance=class_laplace_logit_variance,
            observed_laplace_logit_variance=observed_laplace_logit_variance,
            local_class_accuracy=(
                None if local_ba_result is None else local_ba_result.theta
            ),
            local_balanced_accuracy=(
                None if local_ba_result is None else local_ba_result.balanced_accuracy
            ),
            local_corrected_balanced_accuracy=(
                None
                if local_ba_result is None
                else local_ba_result.corrected_balanced_accuracy
            ),
            local_balanced_total=(
                None if local_ba_result is None else local_ba_result.total
            ),
            local_balanced_correct=(
                None if local_ba_result is None else local_ba_result.correct
            ),
            budget_aware_locality=budget_aware_result,
            prior_mask=prior_mask_resolved,
            evidence_mask=evidence_mask_resolved,
            prior_observations=self.prior_observations,
            uses_same_prior_and_evidence=uses_same,
            sample_indices=sample_indices,
            annotator_indices=annotator_indices,
        )
        return result

    def _resolve_missing_label(self, clf):
        if self.missing_label is not None:
            return self.missing_label
        return getattr(clf, "missing_label", MISSING_LABEL)

    def _needs_sample_embeddings(self) -> bool:
        return (
            self.base_channel == "local_mace"
            or self.base_channel == "local_accuracy_uniform"
            or self.base_channel == "local_balanced_accuracy"
            or self.evidence in {
                "local_kernel_soft_counts",
                "local_kernel_mace_counts",
                "local_kernel_accuracy_counts",
                "local_kernel_balanced_accuracy_counts",
            }
            or self.class_prior in {"kernel", "evidence_shrunk", "laplace"}
            or self.observed_class_prior in {"kernel", "evidence_shrunk", "laplace"}
            or self._uses_local_balanced_accuracy_utility()
            or self._uses_local_balanced_accuracy_posterior()
        )

    def _needs_logits(self) -> bool:
        return self.class_prior == "laplace" or self.observed_class_prior == "laplace"

    def _uses_local_balanced_accuracy_utility(self) -> bool:
        return self.utility in {
            "local_balanced_accuracy",
            "local_corrected_balanced_accuracy",
            "local_ba",
            "local_cba",
        }

    def _uses_local_balanced_accuracy_posterior(self) -> bool:
        return self.posterior == "local_balanced_accuracy"

    def _resolve_bandwidth_reference_embeddings(
        self,
        *,
        sample_embeddings: np.ndarray | None,
        observed_mask: np.ndarray,
    ) -> np.ndarray | None:
        if sample_embeddings is None:
            return None
        if self.bandwidth_reference == "all":
            return self._subsample_bandwidth_reference(sample_embeddings)
        if self.bandwidth_reference == "labeled":
            labeled = np.asarray(observed_mask, dtype=bool).any(axis=1)
            if np.any(labeled):
                return self._subsample_bandwidth_reference(sample_embeddings[labeled])
            return self._subsample_bandwidth_reference(sample_embeddings)
        raise RuntimeError("unreachable bandwidth_reference")

    def _subsample_bandwidth_reference(self, embeddings: np.ndarray) -> np.ndarray:
        embeddings = np.asarray(embeddings, dtype=float)
        sample = self.bandwidth_reference_sample
        if sample is None or embeddings.shape[0] <= 1:
            return embeddings
        n = embeddings.shape[0]
        if isinstance(sample, (int, np.integer)) and not isinstance(sample, bool):
            size = min(int(sample), n)
        else:
            size = int(np.ceil(float(sample) * n))
            size = min(max(size, 1), n)
        if size >= n:
            return embeddings
        choice = self.random_state.choice(n, size=size, replace=False)
        return embeddings[np.sort(choice)]

    def _predict_probabilities_and_embeddings(
        self,
        *,
        X,
        clf,
        need_sample_embeddings: bool,
        need_annotator_embeddings: bool,
        need_logits: bool,
    ):
        extra = []
        if need_logits:
            extra.append("logits")
        if need_sample_embeddings and self.embedding_source == "classifier":
            extra.append("embeddings")
        if need_annotator_embeddings:
            extra.append("annotator_embeddings")
        if extra:
            out = clf.predict_proba(X, extra_outputs=extra)
            if not isinstance(out, (tuple, list)) or len(out) != len(extra) + 1:
                raise ValueError("clf.predict_proba returned unexpected extra outputs.")
            P = np.asarray(out[0], dtype=float)
            named = {name: value for name, value in zip(extra, out[1:])}
        else:
            P = np.asarray(clf.predict_proba(X), dtype=float)
            named = {}
        if P.ndim != 2:
            raise ValueError(f"Classifier probabilities must be 2D, got {P.shape}.")
        logits = None
        if need_logits:
            logits = np.asarray(named["logits"], dtype=float)
            if logits.shape != P.shape:
                raise ValueError(
                    "Classifier logits must have the same shape as probabilities; "
                    f"got {logits.shape} and {P.shape}."
                )
        sample_embeddings = None
        if need_sample_embeddings:
            if self.embedding_source == "classifier":
                sample_embeddings = np.asarray(named["embeddings"], dtype=float)
            else:
                sample_embeddings = np.asarray(X, dtype=float)
            sample_embeddings = sample_embeddings.reshape(sample_embeddings.shape[0], -1)
        annotator_embeddings = None
        if need_annotator_embeddings:
            annotator_embeddings = np.asarray(named["annotator_embeddings"], dtype=float)
            annotator_embeddings = annotator_embeddings.reshape(annotator_embeddings.shape[0], -1)
        return P, sample_embeddings, annotator_embeddings, logits

    def _resolve_prior_evidence_masks(
        self,
        *,
        y,
        missing_label,
        prior_indices=None,
        evidence_indices=None,
        initial_indices=None,
        prior_mask=None,
        evidence_mask=None,
        initial_observation_mask=None,
    ) -> tuple[np.ndarray, np.ndarray]:
        prior_indices = self.prior_indices if prior_indices is None else prior_indices
        evidence_indices = self.evidence_indices if evidence_indices is None else evidence_indices
        initial_indices = self.initial_indices if initial_indices is None else initial_indices
        prior_mask = self.prior_mask if prior_mask is None else prior_mask
        evidence_mask = self.evidence_mask if evidence_mask is None else evidence_mask
        initial_observation_mask = (
            self.initial_observation_mask
            if initial_observation_mask is None
            else initial_observation_mask
        )
        all_observed = _make_observation_mask(y, missing_label=missing_label)
        mode = self.prior_observations
        if mode == "same":
            prior_has_selector = prior_indices is not None or prior_mask is not None
            evidence_has_selector = evidence_indices is not None or evidence_mask is not None
            if prior_has_selector and evidence_has_selector:
                p_mask = _make_observation_mask(
                    y,
                    missing_label=missing_label,
                    indices=prior_indices,
                    mask=prior_mask,
                )
                e_mask = _make_observation_mask(
                    y,
                    missing_label=missing_label,
                    indices=evidence_indices,
                    mask=evidence_mask,
                )
                if not np.array_equal(p_mask, e_mask):
                    raise ValueError(
                        "prior_observations='same' received different prior and evidence selectors."
                    )
                return p_mask, e_mask
            if evidence_has_selector:
                shared = _make_observation_mask(
                    y,
                    missing_label=missing_label,
                    indices=evidence_indices,
                    mask=evidence_mask,
                )
            elif prior_has_selector:
                shared = _make_observation_mask(
                    y,
                    missing_label=missing_label,
                    indices=prior_indices,
                    mask=prior_mask,
                )
            else:
                shared = all_observed
            return shared, shared.copy()
        if mode == "separate":
            if prior_indices is None and prior_mask is None:
                raise ValueError("prior_observations='separate' requires prior_indices or prior_mask.")
            if evidence_indices is None and evidence_mask is None:
                raise ValueError(
                    "prior_observations='separate' requires evidence_indices or evidence_mask."
                )
            return (
                _make_observation_mask(
                    y,
                    missing_label=missing_label,
                    indices=prior_indices,
                    mask=prior_mask,
                ),
                _make_observation_mask(
                    y,
                    missing_label=missing_label,
                    indices=evidence_indices,
                    mask=evidence_mask,
                ),
            )
        if mode == "none":
            e_mask = _make_observation_mask(
                y,
                missing_label=missing_label,
                indices=evidence_indices,
                mask=evidence_mask,
            )
            return np.zeros_like(all_observed, dtype=bool), e_mask
        if mode == "initial":
            if initial_indices is None and initial_observation_mask is None:
                raise ValueError(
                    "prior_observations='initial' requires initial_indices or initial_observation_mask."
                )
            p_mask = _make_observation_mask(
                y,
                missing_label=missing_label,
                indices=initial_indices,
                mask=initial_observation_mask,
            )
            if evidence_indices is not None or evidence_mask is not None:
                e_mask = _make_observation_mask(
                    y,
                    missing_label=missing_label,
                    indices=evidence_indices,
                    mask=evidence_mask,
                )
            else:
                e_mask = all_observed & ~p_mask
            return p_mask, e_mask
        raise RuntimeError("unreachable prior_observations mode")

    def _resolve_annotator_similarity(
        self,
        annotator_embeddings: np.ndarray | None,
        *,
        n_annotators: int,
    ) -> np.ndarray | None:
        if not self.use_annotator_embeddings:
            return None
        if annotator_embeddings is None:
            raise ValueError("use_annotator_embeddings=True requires annotator_embeddings.")
        if annotator_embeddings.shape[0] != n_annotators:
            raise ValueError(
                "annotator_embeddings must provide one row per annotator; "
                f"expected {n_annotators}, got {annotator_embeddings.shape[0]}."
            )
        return _pairwise_kernel(
            annotator_embeddings,
            annotator_embeddings,
            kernel=self.annotator_kernel,
            gamma=self.annotator_gamma,
            normalize_embeddings=self.normalize_embeddings,
            bandwidth_knn_k=self.bandwidth_knn_k,
            eps=self.eps,
        )

    def _estimate_base_channel(
        self,
        *,
        P,
        y_idx,
        prior_mask,
        weights,
        sample_embeddings,
        gamma_reference_embeddings,
        sample_indices,
        annotator_indices,
        n_annotators,
        n_classes,
        annotator_similarity,
    ) -> tuple[
        np.ndarray,
        MaceChannelParameters | None,
        AccuracyUniformChannelParameters | None,
    ]:
        if self.base_channel == "uniform":
            return _estimate_base_uniform(n_annotators, n_classes), None, None
        if self.base_channel == "fixed_full":
            return _fixed_full_channel(
                n_annotators=n_annotators,
                n_classes=n_classes,
                fixed_prior_accuracy=self.fixed_prior_accuracy,
                eps=self.eps,
            ), None, None
        if self.base_channel == "fixed_mace":
            params = _fixed_mace_parameters(
                n_annotators=n_annotators,
                n_classes=n_classes,
                fixed_prior_accuracy=self.fixed_prior_accuracy,
            )
            return (
                _mace_channel_from_theta_g(
                    params.theta,
                    params.g,
                    n_classes=n_classes,
                    eps=self.eps,
                ),
                params,
                None,
            )
        if self.base_channel == "fixed_accuracy_uniform":
            params = _fixed_accuracy_uniform_parameters(
                n_annotators=n_annotators,
                n_classes=n_classes,
                fixed_prior_accuracy=self.fixed_prior_accuracy,
            )
            return (
                _accuracy_uniform_channel_from_theta(
                    params.theta,
                    n_classes=n_classes,
                    eps=self.eps,
                ),
                None,
                params,
            )
        if self.base_channel == "global_mace":
            params = _estimate_global_mace_parameters(
                P,
                y_idx,
                prior_mask,
                weights,
                n_annotators=n_annotators,
                n_classes=n_classes,
                fixed_prior_accuracy=self.fixed_prior_accuracy,
                fixed_prior_strength=self.fixed_prior_strength,
                annotator_similarity=annotator_similarity,
                eps=self.eps,
            )
            return (
                _mace_channel_from_theta_g(
                    params.theta,
                    params.g,
                    n_classes=n_classes,
                    eps=self.eps,
                ),
                params,
                None,
            )
        if self.base_channel == "local_mace":
            if sample_embeddings is None:
                raise ValueError("base_channel='local_mace' requires sample embeddings.")
            params = _estimate_local_mace_parameters(
                P,
                y_idx,
                prior_mask,
                weights,
                sample_embeddings=sample_embeddings,
                candidate_indices=sample_indices,
                annotator_indices=annotator_indices,
                n_annotators=n_annotators,
                n_classes=n_classes,
                fixed_prior_accuracy=self.fixed_prior_accuracy,
                fixed_prior_strength=self.fixed_prior_strength,
                kernel=self.kernel,
                gamma=self.gamma,
                normalize_embeddings=self.normalize_embeddings,
                gamma_reference_embeddings=gamma_reference_embeddings,
                bandwidth_knn_k=self.bandwidth_knn_k,
                local_kernel_top_k=self._local_kernel_top_k(),
                local_kernel_weighting=self.local_kernel_weighting,
                annotator_similarity=annotator_similarity,
                eps=self.eps,
            )
            return (
                _mace_channel_from_theta_g(
                    params.theta,
                    params.g,
                    n_classes=n_classes,
                    eps=self.eps,
                ),
                params,
                None,
            )
        if self.base_channel == "global_accuracy_uniform":
            params = _estimate_global_accuracy_uniform_parameters(
                P,
                y_idx,
                prior_mask,
                weights,
                n_annotators=n_annotators,
                n_classes=n_classes,
                fixed_prior_accuracy=self.fixed_prior_accuracy,
                fixed_prior_strength=self.fixed_prior_strength,
                annotator_similarity=annotator_similarity,
                eps=self.eps,
            )
            return (
                _accuracy_uniform_channel_from_theta(
                    params.theta,
                    n_classes=n_classes,
                    eps=self.eps,
                ),
                None,
                params,
            )
        if self.base_channel == "local_accuracy_uniform":
            if sample_embeddings is None:
                raise ValueError("base_channel='local_accuracy_uniform' requires sample embeddings.")
            params = _estimate_local_accuracy_uniform_parameters(
                P,
                y_idx,
                prior_mask,
                weights,
                sample_embeddings=sample_embeddings,
                candidate_indices=sample_indices,
                annotator_indices=annotator_indices,
                n_annotators=n_annotators,
                n_classes=n_classes,
                fixed_prior_accuracy=self.fixed_prior_accuracy,
                fixed_prior_strength=self.fixed_prior_strength,
                kernel=self.kernel,
                gamma=self.gamma,
                normalize_embeddings=self.normalize_embeddings,
                gamma_reference_embeddings=gamma_reference_embeddings,
                bandwidth_knn_k=self.bandwidth_knn_k,
                local_kernel_top_k=self._local_kernel_top_k(),
                local_kernel_weighting=self.local_kernel_weighting,
                annotator_similarity=annotator_similarity,
                eps=self.eps,
            )
            return (
                _accuracy_uniform_channel_from_theta(
                    params.theta,
                    n_classes=n_classes,
                    eps=self.eps,
                ),
                None,
                params,
            )
        if self.base_channel == "local_balanced_accuracy":
            if sample_embeddings is None:
                raise ValueError(
                    "base_channel='local_balanced_accuracy' requires sample embeddings."
                )
            B_prior = _estimate_base_global_full(
                P,
                y_idx,
                prior_mask,
                weights,
                n_annotators=n_annotators,
                n_classes=n_classes,
                fixed_prior_accuracy=self.fixed_prior_accuracy,
                fixed_prior_strength=self.fixed_prior_strength,
                eps=self.eps,
            )
            prior_diag = np.diagonal(B_prior, axis1=1, axis2=2)[
                np.asarray(annotator_indices, dtype=int)
            ]
            result = _compute_local_kernel_balanced_accuracy(
                P,
                y_idx,
                prior_mask,
                sample_embeddings=sample_embeddings,
                candidate_indices=sample_indices,
                annotator_indices=annotator_indices,
                prior_diag=prior_diag,
                prior_strength=self._local_prior_strength(),
                n_classes=n_classes,
                kernel=self.kernel,
                gamma=self.gamma,
                normalize_embeddings=self.normalize_embeddings,
                gamma_reference_embeddings=gamma_reference_embeddings,
                bandwidth_knn_k=self.bandwidth_knn_k,
                local_kernel_top_k=self._local_kernel_top_k(),
                local_kernel_weighting=self.local_kernel_weighting,
                eps=self.eps,
            )
            self._last_local_balanced_accuracy_result = result
            return (
                _classwise_accuracy_channel_from_theta(
                    result.theta,
                    n_classes=n_classes,
                    eps=self.eps,
                ),
                None,
                None,
            )
        if self.base_channel == "diag_uniform":
            return _estimate_base_diag_uniform(
                P,
                y_idx,
                prior_mask,
                weights,
                n_annotators=n_annotators,
                n_classes=n_classes,
                theta_prior=self.diag_theta_prior,
                eps=self.eps,
            ), None, None
        if self.base_channel == "global_full":
            return _estimate_base_global_full(
                P,
                y_idx,
                prior_mask,
                weights,
                n_annotators=n_annotators,
                n_classes=n_classes,
                fixed_prior_accuracy=self.fixed_prior_accuracy,
                fixed_prior_strength=self.fixed_prior_strength,
                eps=self.eps,
            ), None, None
        raise RuntimeError("unreachable base_channel")

    def _compute_local_balanced_accuracy_result(
        self,
        *,
        P,
        y_idx,
        evidence_mask,
        sample_embeddings,
        gamma_reference_embeddings,
        sample_indices,
        annotator_indices,
        B,
        n_classes,
    ) -> LocalBalancedAccuracyResult | None:
        if self._last_local_balanced_accuracy_result is not None:
            return self._last_local_balanced_accuracy_result
        if not (
            self._uses_local_balanced_accuracy_utility()
            or self._uses_local_balanced_accuracy_posterior()
        ):
            return None
        if sample_embeddings is None:
            raise ValueError("local balanced-accuracy utilities require sample embeddings.")
        B = np.asarray(B, dtype=float)
        if B.ndim != 3:
            raise ValueError(
                "local balanced-accuracy utilities require a global full confusion prior."
            )
        prior_diag = np.diagonal(B, axis1=1, axis2=2)[
            np.asarray(annotator_indices, dtype=int)
        ]
        result = _compute_local_kernel_balanced_accuracy(
            P,
            y_idx,
            evidence_mask,
            sample_embeddings=sample_embeddings,
            candidate_indices=sample_indices,
            annotator_indices=annotator_indices,
            prior_diag=prior_diag,
            prior_strength=self._local_prior_strength(),
            n_classes=n_classes,
            kernel=self.kernel,
            gamma=self.gamma,
            normalize_embeddings=self.normalize_embeddings,
            gamma_reference_embeddings=gamma_reference_embeddings,
            bandwidth_knn_k=self.bandwidth_knn_k,
            local_kernel_top_k=self._local_kernel_top_k(),
            local_kernel_weighting=self.local_kernel_weighting,
            eps=self.eps,
        )
        self._last_local_balanced_accuracy_result = result
        return result

    def _compute_evidence_counts(
        self,
        *,
        P,
        y_idx,
        evidence_mask,
        weights,
        sample_embeddings,
        gamma_reference_embeddings,
        sample_indices,
        annotator_indices,
        n_annotators,
        n_classes,
        annotator_similarity,
    ):
        if self.evidence == "none":
            return None
        if self.evidence == "global_soft_counts":
            return _compute_global_soft_counts(
                P,
                y_idx,
                evidence_mask,
                weights,
                n_annotators=n_annotators,
                n_classes=n_classes,
                annotator_similarity=annotator_similarity,
            )
        if self.evidence == "local_kernel_soft_counts":
            if sample_embeddings is None:
                raise ValueError("local_kernel_soft_counts requires sample embeddings.")
            return _compute_local_kernel_soft_counts(
                P,
                y_idx,
                evidence_mask,
                weights,
                sample_embeddings=sample_embeddings,
                candidate_indices=sample_indices,
                annotator_indices=annotator_indices,
                n_annotators=n_annotators,
                n_classes=n_classes,
                kernel=self.kernel,
                gamma=self.gamma,
                normalize_embeddings=self.normalize_embeddings,
                gamma_reference_embeddings=gamma_reference_embeddings,
                bandwidth_knn_k=self.bandwidth_knn_k,
                local_kernel_top_k=self._local_kernel_top_k(),
                local_kernel_weighting=self.local_kernel_weighting,
                annotator_similarity=annotator_similarity,
                eps=self.eps,
            )
        if self.evidence == "global_mace_counts":
            return _compute_global_mace_counts(
                P,
                y_idx,
                evidence_mask,
                weights,
                n_annotators=n_annotators,
                n_classes=n_classes,
                annotator_similarity=annotator_similarity,
            )
        if self.evidence == "local_kernel_mace_counts":
            if sample_embeddings is None:
                raise ValueError("local_kernel_mace_counts requires sample embeddings.")
            return _compute_local_kernel_mace_counts(
                P,
                y_idx,
                evidence_mask,
                weights,
                sample_embeddings=sample_embeddings,
                candidate_indices=sample_indices,
                annotator_indices=annotator_indices,
                n_annotators=n_annotators,
                n_classes=n_classes,
                kernel=self.kernel,
                gamma=self.gamma,
                normalize_embeddings=self.normalize_embeddings,
                gamma_reference_embeddings=gamma_reference_embeddings,
                bandwidth_knn_k=self.bandwidth_knn_k,
                local_kernel_top_k=self._local_kernel_top_k(),
                local_kernel_weighting=self.local_kernel_weighting,
                annotator_similarity=annotator_similarity,
                eps=self.eps,
            )
        if self.evidence == "global_accuracy_counts":
            return _compute_global_accuracy_counts(
                P,
                y_idx,
                evidence_mask,
                weights,
                n_annotators=n_annotators,
                annotator_similarity=annotator_similarity,
            )
        if self.evidence == "local_kernel_accuracy_counts":
            if sample_embeddings is None:
                raise ValueError("local_kernel_accuracy_counts requires sample embeddings.")
            return _compute_local_kernel_accuracy_counts(
                P,
                y_idx,
                evidence_mask,
                weights,
                sample_embeddings=sample_embeddings,
                candidate_indices=sample_indices,
                annotator_indices=annotator_indices,
                n_annotators=n_annotators,
                kernel=self.kernel,
                gamma=self.gamma,
                normalize_embeddings=self.normalize_embeddings,
                gamma_reference_embeddings=gamma_reference_embeddings,
                bandwidth_knn_k=self.bandwidth_knn_k,
                local_kernel_top_k=self._local_kernel_top_k(),
                local_kernel_weighting=self.local_kernel_weighting,
                annotator_similarity=annotator_similarity,
                eps=self.eps,
            )
        if self.evidence == "local_kernel_balanced_accuracy_counts":
            if self._last_local_balanced_accuracy_result is None:
                raise ValueError(
                    "local_kernel_balanced_accuracy_counts requires a computed "
                    "local balanced-accuracy result."
                )
            return self._last_local_balanced_accuracy_result
        raise RuntimeError("unreachable evidence")

    def _combine_posterior(
        self,
        *,
        B,
        base_mace_params,
        base_accuracy_params,
        evidence_counts,
        sample_indices,
        annotator_indices,
        n_classes,
    ):
        if self.posterior == "base_only":
            C, alpha = _combine_base_only(
                B,
                n_candidates=len(sample_indices),
                annotator_indices=annotator_indices,
            )
            if base_mace_params is None:
                mace_params = None
            elif base_mace_params.theta.ndim == 1:
                mace_params = _broadcast_global_mace_params(
                    base_mace_params,
                    n_candidates=len(sample_indices),
                    annotator_indices=annotator_indices,
                )
            else:
                mace_params = base_mace_params
            if base_accuracy_params is None:
                accuracy_params = None
            elif base_accuracy_params.theta.ndim == 1:
                accuracy_params = _broadcast_global_accuracy_params(
                    base_accuracy_params,
                    n_candidates=len(sample_indices),
                    annotator_indices=annotator_indices,
                )
            else:
                accuracy_params = base_accuracy_params
            return C, alpha, mace_params, accuracy_params
        if self.posterior == "dirichlet_global":
            C, alpha = _combine_dirichlet_global(
                B,
                evidence_counts,
                prior_strength=self.prior_strength,
                n_candidates=len(sample_indices),
                annotator_indices=annotator_indices,
                eps=self.eps,
            )
            return C, alpha, None, None
        if self.posterior == "dirichlet_local":
            C, alpha = _combine_dirichlet_local(
                B,
                evidence_counts,
                prior_strength=self._local_prior_strength(),
                annotator_indices=annotator_indices,
                eps=self.eps,
            )
            return C, alpha, None, None
        if self.posterior == "global_mace":
            if base_mace_params is None:
                if self.base_channel != "global_full":
                    raise ValueError("posterior='global_mace' requires a MACE base channel.")
                base_mace_params = _full_channel_to_mace_params(B, eps=self.eps)
            C, mace_params = _combine_mace_global(
                base_mace_params,
                evidence_counts,
                prior_strength=self.prior_strength,
                n_candidates=len(sample_indices),
                annotator_indices=annotator_indices,
                n_classes=n_classes,
                eps=self.eps,
            )
            return C, None, mace_params, None
        if self.posterior == "local_mace":
            if base_mace_params is None:
                if self.base_channel != "global_full":
                    raise ValueError("posterior='local_mace' requires a MACE base channel.")
                base_mace_params = _full_channel_to_mace_params(B, eps=self.eps)
            C, mace_params = _combine_mace_local(
                base_mace_params,
                evidence_counts,
                prior_strength=self._local_prior_strength(),
                n_candidates=len(sample_indices),
                annotator_indices=annotator_indices,
                n_classes=n_classes,
                eps=self.eps,
            )
            return C, None, mace_params, None
        if self.posterior == "global_accuracy_uniform":
            if base_accuracy_params is None:
                if self.base_channel not in {"uniform", "global_full"}:
                    raise ValueError(
                        "posterior='global_accuracy_uniform' requires an "
                        "accuracy-uniform base channel."
                    )
                base_accuracy_params = _full_channel_to_accuracy_uniform_params(
                    B,
                    eps=self.eps,
                )
            C, accuracy_params = _combine_accuracy_uniform_global(
                base_accuracy_params,
                evidence_counts,
                prior_strength=self.prior_strength,
                n_candidates=len(sample_indices),
                annotator_indices=annotator_indices,
                n_classes=n_classes,
                eps=self.eps,
            )
            return C, None, None, accuracy_params
        if self.posterior == "local_accuracy_uniform":
            if base_accuracy_params is None:
                if self.base_channel not in {"uniform", "global_full"}:
                    raise ValueError(
                        "posterior='local_accuracy_uniform' requires an "
                        "accuracy-uniform base channel."
                    )
                base_accuracy_params = _full_channel_to_accuracy_uniform_params(
                    B,
                    eps=self.eps,
                )
            C, accuracy_params = _combine_accuracy_uniform_local(
                base_accuracy_params,
                evidence_counts,
                prior_strength=self._local_prior_strength(),
                n_candidates=len(sample_indices),
                annotator_indices=annotator_indices,
                n_classes=n_classes,
                eps=self.eps,
            )
            return C, None, None, accuracy_params
        if self.posterior == "local_balanced_accuracy":
            if not isinstance(evidence_counts, LocalBalancedAccuracyResult):
                raise ValueError(
                    "posterior='local_balanced_accuracy' requires local balanced-accuracy evidence."
                )
            C = _classwise_accuracy_channel_from_theta(
                evidence_counts.theta,
                n_classes=n_classes,
                eps=self.eps,
            )
            return C, None, None, None
        raise RuntimeError("unreachable posterior")

    @staticmethod
    def compute_budget_aware_k_and_prior_strength(**kwargs) -> BudgetAwareLocalityResult:
        return compute_budget_aware_k_and_prior_strength(**kwargs)

    def _observation_support_mask(
        self,
        *,
        prior_mask,
        evidence_mask,
        observed_mask,
    ) -> np.ndarray:
        if self.observed_class_prior_support == "observed":
            return observed_mask.any(axis=1)
        if self.observed_class_prior_support == "prior":
            return prior_mask.any(axis=1)
        if self.observed_class_prior_support == "evidence":
            return evidence_mask.any(axis=1)
        raise RuntimeError("unreachable observed_class_prior_support")

    @staticmethod
    def laplace_mean_field_predictive_proba(
        *,
        support_logits,
        support_embeddings,
        query_logits,
        query_embeddings,
        prior_precision: float = 1.0,
        include_bias: bool = True,
        predictive_samples: int = 32,
        variance_scale: float = 1.0,
        query_support_indices=None,
        random_state=None,
        eps: float = 1e-12,
    ) -> LaplacePredictiveResult:
        support_logits = np.asarray(support_logits, dtype=float)
        query_logits = np.asarray(query_logits, dtype=float)
        support_embeddings = np.asarray(support_embeddings, dtype=float)
        query_embeddings = np.asarray(query_embeddings, dtype=float)
        prior_precision = float(prior_precision)
        predictive_samples = int(predictive_samples)
        variance_scale = float(variance_scale)
        if support_logits.ndim != 2 or query_logits.ndim != 2:
            raise ValueError("support_logits and query_logits must be 2D arrays.")
        if support_embeddings.ndim != 2 or query_embeddings.ndim != 2:
            raise ValueError("support_embeddings and query_embeddings must be 2D arrays.")
        if support_logits.shape[1] != query_logits.shape[1]:
            raise ValueError("support_logits and query_logits must agree on n_classes.")
        if support_embeddings.shape[1] != query_embeddings.shape[1]:
            raise ValueError(
                "support_embeddings and query_embeddings must have the same feature dimension."
            )
        if support_logits.shape[0] != support_embeddings.shape[0]:
            raise ValueError("support logits and embeddings must have the same n_samples.")
        if query_logits.shape[0] != query_embeddings.shape[0]:
            raise ValueError("query logits and embeddings must have the same n_samples.")
        if prior_precision <= 0:
            raise ValueError("prior_precision must be > 0.")
        if predictive_samples <= 0:
            raise ValueError("predictive_samples must be > 0.")
        if variance_scale < 0:
            raise ValueError("variance_scale must be >= 0.")

        K = query_logits.shape[1]
        D = query_embeddings.shape[1]
        if support_logits.shape[0] == 0:
            precision = np.full((K, D), prior_precision, dtype=float)
            bias_precision = np.full(K, prior_precision, dtype=float)
            curvature = np.empty((0, K), dtype=float)
        else:
            p_support = _softmax_logits(support_logits, axis=1)
            curvature = p_support * (1.0 - p_support)
            precision = prior_precision + curvature.T @ (support_embeddings ** 2)
            bias_precision = prior_precision + curvature.sum(axis=0)
        precision = np.maximum(precision, eps)
        bias_precision = np.maximum(bias_precision, eps)

        query_z2 = query_embeddings ** 2
        if query_support_indices is None:
            logit_variance = query_z2 @ (1.0 / precision).T
            if include_bias:
                logit_variance += 1.0 / bias_precision[None, :]
        else:
            query_support_indices = np.asarray(query_support_indices, dtype=int)
            if query_support_indices.shape != (query_logits.shape[0],):
                raise ValueError("query_support_indices must have shape (n_query,).")
            logit_variance = np.empty((query_logits.shape[0], K), dtype=float)
            for i, support_row in enumerate(query_support_indices):
                precision_i = precision
                bias_precision_i = bias_precision
                if support_row >= 0:
                    if support_row >= support_logits.shape[0]:
                        raise ValueError("query_support_indices contains an out-of-bounds row.")
                    precision_i = precision - (
                        curvature[support_row, :, None]
                        * (support_embeddings[support_row][None, :] ** 2)
                    )
                    precision_i = np.maximum(precision_i, eps)
                    if include_bias:
                        bias_precision_i = np.maximum(
                            bias_precision - curvature[support_row],
                            eps,
                        )
                logit_variance[i] = query_z2[i] @ (1.0 / precision_i).T
                if include_bias:
                    logit_variance[i] += 1.0 / bias_precision_i
        logit_variance = variance_scale * np.maximum(logit_variance, 0.0)

        rng = check_random_state(random_state)
        noise = rng.normal(
            loc=0.0,
            scale=np.sqrt(logit_variance)[None, :, :],
            size=(predictive_samples, query_logits.shape[0], K),
        )
        samples = _softmax_logits(query_logits[None, :, :] + noise, axis=-1)
        proba = _normalize_axis(samples.mean(axis=0), axis=1, eps=eps)
        return LaplacePredictiveResult(
            proba=proba,
            logit_variance=logit_variance,
            samples=samples,
        )

    def _resolve_observed_class_probabilities(
        self,
        *,
        P,
        logits,
        sample_embeddings,
        gamma_reference_embeddings,
        prior_mask,
        evidence_mask,
        observed_mask,
        n_classes,
    ) -> np.ndarray:
        self._last_observed_laplace_logit_variance = None
        if self.observed_class_prior == "classifier":
            return P.copy()
        if self.observed_class_prior == "kernel":
            if sample_embeddings is None:
                raise ValueError("observed_class_prior='kernel' requires sample embeddings.")
            support_mask = self._observation_support_mask(
                prior_mask=prior_mask,
                evidence_mask=evidence_mask,
                observed_mask=observed_mask,
            )
            support = np.flatnonzero(support_mask)
            if support.size == 0:
                return P.copy()
            kernel_name = self.observed_class_prior_kernel or self.class_prior_kernel or self.kernel
            gamma = (
                self.observed_class_prior_gamma
                if self.observed_class_prior_gamma is not None
                else self.class_prior_gamma
            )
            Kx = _pairwise_kernel(
                sample_embeddings[support],
                sample_embeddings,
                kernel=kernel_name,
                gamma=gamma,
                normalize_embeddings=self.normalize_embeddings,
                gamma_reference_embeddings=gamma_reference_embeddings,
                bandwidth_knn_k=self.bandwidth_knn_k,
                eps=self.eps,
            )
            if self.observed_class_prior_leave_one_out:
                support_to_row = {sample_id: row for row, sample_id in enumerate(support)}
                for sample_id, row in support_to_row.items():
                    Kx[row, sample_id] = 0.0
            alpha0 = _class_prior_alpha0_vector(self.class_prior_alpha0, n_classes)
            alpha = alpha0[None, :] + Kx.T @ P[support]
            return _normalize_axis(alpha, axis=1, eps=self.eps)
        if self.observed_class_prior == "evidence_shrunk":
            if sample_embeddings is None:
                raise ValueError(
                    "observed_class_prior='evidence_shrunk' requires sample embeddings."
                )
            support_mask = self._observation_support_mask(
                prior_mask=prior_mask,
                evidence_mask=evidence_mask,
                observed_mask=observed_mask,
            )
            support = np.flatnonzero(support_mask)
            alpha0 = _class_prior_alpha0_vector(self.class_prior_alpha0, n_classes)
            if support.size == 0:
                evidence_strength = np.zeros(P.shape[0], dtype=float)
            else:
                kernel_name = self.observed_class_prior_kernel or self.class_prior_kernel or self.kernel
                gamma = (
                    self.observed_class_prior_gamma
                    if self.observed_class_prior_gamma is not None
                    else self.class_prior_gamma
                )
                Kx = _pairwise_kernel(
                    sample_embeddings[support],
                    sample_embeddings,
                    kernel=kernel_name,
                    gamma=gamma,
                    normalize_embeddings=self.normalize_embeddings,
                    gamma_reference_embeddings=gamma_reference_embeddings,
                    bandwidth_knn_k=self.bandwidth_knn_k,
                    eps=self.eps,
                )
                if self.observed_class_prior_leave_one_out:
                    support_to_row = {sample_id: row for row, sample_id in enumerate(support)}
                    for sample_id, row in support_to_row.items():
                        Kx[row, sample_id] = 0.0
                eta = _compute_evidence_weights(
                    P,
                    self.class_prior_evidence_weight,
                    eps=self.eps,
                )
                evidence_strength = Kx.T @ eta[support]
            alpha = alpha0[None, :] + evidence_strength[:, None] * P
            return _normalize_axis(alpha, axis=1, eps=self.eps)
        if self.observed_class_prior == "laplace":
            if sample_embeddings is None:
                raise ValueError("observed_class_prior='laplace' requires sample embeddings.")
            if logits is None:
                raise ValueError("observed_class_prior='laplace' requires classifier logits.")
            support_mask = self._observation_support_mask(
                prior_mask=prior_mask,
                evidence_mask=evidence_mask,
                observed_mask=observed_mask,
            )
            support = np.flatnonzero(support_mask)
            if support.size == 0:
                self._last_observed_laplace_logit_variance = np.zeros_like(P, dtype=float)
                return P.copy()
            query_support_rows = None
            if self.observed_class_prior_leave_one_out:
                support_to_row = {sample_id: row for row, sample_id in enumerate(support)}
                query_support_rows = np.asarray(
                    [support_to_row.get(sample_id, -1) for sample_id in range(P.shape[0])],
                    dtype=int,
                )
            laplace = self.laplace_mean_field_predictive_proba(
                support_logits=logits[support],
                support_embeddings=sample_embeddings[support],
                query_logits=logits,
                query_embeddings=sample_embeddings,
                prior_precision=self.laplace_prior_precision,
                include_bias=self.laplace_include_bias,
                predictive_samples=self.laplace_predictive_samples,
                variance_scale=self.laplace_variance_scale,
                query_support_indices=query_support_rows,
                random_state=self.random_state,
                eps=self.eps,
            )
            self._last_observed_laplace_logit_variance = laplace.logit_variance
            return laplace.proba
        raise RuntimeError("unreachable observed_class_prior")

    def _resolve_candidate_class_prior(
        self,
        *,
        P,
        logits,
        sample_embeddings,
        gamma_reference_embeddings,
        sample_indices,
        evidence_mask,
        observed_mask,
        n_classes,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        self._last_class_prior_laplace_logits = None
        self._last_class_prior_laplace_logit_variance = None
        P_cand = P[sample_indices]
        if self.class_prior == "classifier":
            return P_cand, None
        if self.class_prior == "uniform":
            return np.full_like(P_cand, 1.0 / n_classes, dtype=float), None
        if self.class_prior == "top_m":
            out = np.zeros_like(P_cand, dtype=float)
            top_m = int(self.top_m)
            top = np.argpartition(P_cand, kth=-top_m, axis=1)[:, -top_m:]
            rows = np.arange(P_cand.shape[0])[:, None]
            out[rows, top] = 1.0 / top_m
            return out, None
        if self.class_prior == "kernel":
            if sample_embeddings is None:
                raise ValueError("class_prior='kernel' requires sample embeddings.")
            support_mask = (
                evidence_mask.any(axis=1)
                if self.class_prior_support == "evidence"
                else observed_mask.any(axis=1)
            )
            support = np.flatnonzero(support_mask)
            alpha0 = _class_prior_alpha0_vector(self.class_prior_alpha0, n_classes)
            if support.size == 0:
                alpha = np.broadcast_to(alpha0[None, :], P_cand.shape).copy()
            else:
                kernel_name = self.class_prior_kernel or self.kernel
                Kx = _pairwise_kernel(
                    sample_embeddings[support],
                    sample_embeddings[sample_indices],
                    kernel=kernel_name,
                    gamma=self.class_prior_gamma,
                    normalize_embeddings=self.normalize_embeddings,
                    gamma_reference_embeddings=gamma_reference_embeddings,
                    bandwidth_knn_k=self.bandwidth_knn_k,
                    eps=self.eps,
                )
                alpha = alpha0[None, :] + Kx.T @ P[support]
            return _normalize_axis(alpha, axis=1, eps=self.eps), alpha
        if self.class_prior == "evidence_shrunk":
            if sample_embeddings is None:
                raise ValueError("class_prior='evidence_shrunk' requires sample embeddings.")
            support_mask = (
                evidence_mask.any(axis=1)
                if self.class_prior_support == "evidence"
                else observed_mask.any(axis=1)
            )
            support = np.flatnonzero(support_mask)
            alpha0 = _class_prior_alpha0_vector(self.class_prior_alpha0, n_classes)
            if support.size == 0:
                evidence_strength = np.zeros(P_cand.shape[0], dtype=float)
            else:
                kernel_name = self.class_prior_kernel or self.kernel
                Kx = _pairwise_kernel(
                    sample_embeddings[support],
                    sample_embeddings[sample_indices],
                    kernel=kernel_name,
                    gamma=self.class_prior_gamma,
                    normalize_embeddings=self.normalize_embeddings,
                    gamma_reference_embeddings=gamma_reference_embeddings,
                    bandwidth_knn_k=self.bandwidth_knn_k,
                    eps=self.eps,
                )
                eta = _compute_evidence_weights(
                    P,
                    self.class_prior_evidence_weight,
                    eps=self.eps,
                )
                evidence_strength = Kx.T @ eta[support]
            alpha = alpha0[None, :] + evidence_strength[:, None] * P_cand
            return _normalize_axis(alpha, axis=1, eps=self.eps), alpha
        if self.class_prior == "laplace":
            if sample_embeddings is None:
                raise ValueError("class_prior='laplace' requires sample embeddings.")
            if logits is None:
                raise ValueError("class_prior='laplace' requires classifier logits.")
            support_mask = (
                evidence_mask.any(axis=1)
                if self.class_prior_support == "evidence"
                else observed_mask.any(axis=1)
            )
            support = np.flatnonzero(support_mask)
            if support.size == 0:
                self._last_class_prior_laplace_logits = logits[sample_indices].copy()
                self._last_class_prior_laplace_logit_variance = np.zeros_like(
                    P_cand,
                    dtype=float,
                )
                return P_cand, None
            laplace = self.laplace_mean_field_predictive_proba(
                support_logits=logits[support],
                support_embeddings=sample_embeddings[support],
                query_logits=logits[sample_indices],
                query_embeddings=sample_embeddings[sample_indices],
                prior_precision=self.laplace_prior_precision,
                include_bias=self.laplace_include_bias,
                predictive_samples=self.laplace_predictive_samples,
                variance_scale=self.laplace_variance_scale,
                random_state=self.random_state,
                eps=self.eps,
            )
            self._last_class_prior_laplace_logits = logits[sample_indices].copy()
            self._last_class_prior_laplace_logit_variance = laplace.logit_variance
            return laplace.proba, None
        raise RuntimeError("unreachable class_prior")

    def _utilities_from_result(self, result: ChannelEstimationResult) -> np.ndarray:
        if self._uses_local_balanced_accuracy_utility():
            if self.utility in {"local_balanced_accuracy", "local_ba"}:
                utilities = result.local_balanced_accuracy
            else:
                utilities = result.local_corrected_balanced_accuracy
            if utilities is None:
                raise ValueError("Local balanced-accuracy utility was not computed.")
            utilities = np.asarray(utilities, dtype=float)
            self.last_utility_mean_ = utilities
            self.last_utility_std_ = np.zeros_like(utilities, dtype=float)
            self.last_utility_lcb_ = None
            self.last_utility_ucb_ = None
            self.last_utility_draws_ = None
            self.last_instance_difficulty_gate_ = None
            return utilities
        rng = np.random.default_rng(self.random_state.randint(0, 2**32 - 1))
        n_draws = (
            self.n_mc_samples
            if (self.sample_class_prior or self.sample_channel)
            else 1
        )
        draws = np.empty(
            (
                n_draws,
                len(result.sample_indices),
                len(result.annotator_indices),
            ),
            dtype=float,
        )
        for t in range(n_draws):
            p = result.class_prior
            C = result.posterior_channel
            if self.sample_class_prior:
                if self.class_prior == "laplace":
                    p = self._sample_laplace_class_prior(result, rng)
                else:
                    p = self._sample_dirichlet_rows(result.class_prior_alpha, rng)
            if self.sample_channel:
                if result.posterior_mace_params is not None:
                    C = self._sample_mace_channel(
                        result.posterior_mace_params,
                        rng,
                    )
                elif result.posterior_accuracy_params is not None:
                    C = self._sample_accuracy_uniform_channel(
                        result.posterior_accuracy_params,
                        rng,
                        n_classes=C.shape[-1],
                    )
                elif isinstance(result.evidence_counts, LocalBalancedAccuracyResult):
                    C = self._sample_local_balanced_accuracy_channel(
                        result.evidence_counts,
                        rng,
                        n_classes=C.shape[-1],
                    )
                else:
                    C = self._sample_dirichlet_rows(result.posterior_alpha, rng)
            draws[t] = self._compute_utility(p, C)
        self.last_utility_mean_ = np.mean(draws, axis=0)
        self.last_utility_std_ = np.std(draws, axis=0)
        self.last_utility_lcb_ = np.quantile(draws, 0.05, axis=0) if n_draws > 1 else None
        self.last_utility_ucb_ = np.quantile(draws, 0.95, axis=0) if n_draws > 1 else None
        self.last_utility_draws_ = draws if self.store_utility_draws else None
        self.last_instance_difficulty_gate_ = (
            _compute_instance_difficulty_gate(
                result.class_prior,
                power=self.difficulty_gate_power,
                eps=self.eps,
            )
            if self.utility == "difficulty_gated_information_gain"
            else None
        )
        if self.utility_aggregation == "quantile":
            return np.quantile(draws, self.utility_quantile, axis=0)
        return self.last_utility_mean_

    def _compute_utility(self, p: np.ndarray, C: np.ndarray) -> np.ndarray:
        if self.utility == "expected_accuracy":
            return _compute_expected_accuracy(p, C)
        if self.utility == "bias_corrected_accuracy":
            return _compute_bias_corrected_accuracy(p, C, eps=self.eps)
        if self.utility == "information_gain":
            return _compute_information_gain(p, C, eps=self.eps)
        if self.utility == "difficulty_gated_information_gain":
            return _compute_difficulty_gated_information_gain(
                p,
                C,
                power=self.difficulty_gate_power,
                eps=self.eps,
            )
        raise RuntimeError("unreachable utility")

    def _sample_dirichlet_rows(self, alpha: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        if alpha is None:
            raise ValueError("Cannot sample Dirichlet rows without alpha.")
        alpha = np.clip(np.asarray(alpha, dtype=float), self.eps, None)
        samples = rng.gamma(shape=alpha, scale=1.0)
        return _normalize_axis(samples, axis=-1, eps=self.eps)

    def _sample_laplace_class_prior(
        self,
        result: ChannelEstimationResult,
        rng: np.random.Generator,
    ) -> np.ndarray:
        if (
            result.class_prior_laplace_logits is None
            or result.class_prior_laplace_logit_variance is None
        ):
            raise ValueError("Cannot sample Laplace class prior without stored logit moments.")
        logits = np.asarray(result.class_prior_laplace_logits, dtype=float)
        variance = np.asarray(result.class_prior_laplace_logit_variance, dtype=float)
        noise = rng.normal(
            loc=0.0,
            scale=np.sqrt(np.maximum(variance, 0.0)),
            size=logits.shape,
        )
        return _softmax_logits(logits + noise, axis=1)

    def _sample_local_balanced_accuracy_channel(
        self,
        evidence: LocalBalancedAccuracyResult,
        rng: np.random.Generator,
        *,
        n_classes: int,
    ) -> np.ndarray:
        prior_diag = np.asarray(evidence.prior_diag, dtype=float)
        prior_strength = float(evidence.prior_strength)
        alpha = (
            prior_strength * prior_diag[None, :, :]
            + np.asarray(evidence.correct, dtype=float)
        )
        beta = (
            prior_strength * (1.0 - prior_diag)[None, :, :]
            + np.asarray(evidence.total, dtype=float)
            - np.asarray(evidence.correct, dtype=float)
        )
        theta = rng.beta(
            np.clip(alpha, self.eps, None),
            np.clip(beta, self.eps, None),
        )
        return _classwise_accuracy_channel_from_theta(
            theta,
            n_classes=n_classes,
            eps=self.eps,
        )

    def _sample_mace_channel(
        self,
        params: MaceChannelParameters,
        rng: np.random.Generator,
    ) -> np.ndarray:
        if (
            params.theta_success_alpha is None
            or params.theta_failure_beta is None
            or params.g_alpha is None
        ):
            raise ValueError("Cannot sample MACE channel without posterior MACE parameters.")
        a = np.clip(params.theta_success_alpha, self.eps, None)
        b = np.clip(params.theta_failure_beta, self.eps, None)
        theta = rng.beta(a, b)
        g = self._sample_dirichlet_rows(params.g_alpha, rng)
        return _mace_channel_from_theta_g(
            theta,
            g,
            n_classes=g.shape[-1],
            eps=self.eps,
        )

    def _sample_accuracy_uniform_channel(
        self,
        params: AccuracyUniformChannelParameters,
        rng: np.random.Generator,
        *,
        n_classes: int,
    ) -> np.ndarray:
        if params.alpha is None or params.beta is None:
            raise ValueError(
                "Cannot sample accuracy-uniform channel without posterior Beta parameters."
            )
        alpha = np.clip(params.alpha, self.eps, None)
        beta = np.clip(params.beta, self.eps, None)
        theta = rng.beta(alpha, beta)
        return _accuracy_uniform_channel_from_theta(
            theta,
            n_classes=n_classes,
            eps=self.eps,
        )

    def _store_result_diagnostics(
        self,
        result: ChannelEstimationResult,
        utilities: np.ndarray,
    ):
        self.last_result_ = result
        self.last_base_channel_ = result.base_channel
        self.last_base_mace_theta_ = (
            None if result.base_mace_params is None else result.base_mace_params.theta
        )
        self.last_base_mace_g_ = (
            None if result.base_mace_params is None else result.base_mace_params.g
        )
        self.last_base_accuracy_theta_ = (
            None
            if result.base_accuracy_params is None
            else result.base_accuracy_params.theta
        )
        self.last_evidence_counts_ = result.evidence_counts
        self.last_evidence_mace_counts_ = result.evidence_mace_counts
        self.last_evidence_accuracy_counts_ = result.evidence_accuracy_counts
        self.last_posterior_alpha_ = result.posterior_alpha
        self.last_posterior_channel_ = result.posterior_channel
        self.last_posterior_mace_theta_ = (
            None
            if result.posterior_mace_params is None
            else result.posterior_mace_params.theta
        )
        self.last_posterior_mace_g_ = (
            None
            if result.posterior_mace_params is None
            else result.posterior_mace_params.g
        )
        self.last_posterior_accuracy_theta_ = (
            None
            if result.posterior_accuracy_params is None
            else result.posterior_accuracy_params.theta
        )
        self.last_posterior_concentration_ = (
            None
            if result.posterior_alpha is None
            else result.posterior_alpha.sum(axis=-1)
        )
        self.last_classifier_proba_ = result.classifier_proba
        self.last_observed_class_proba_ = result.observed_class_proba
        self.last_candidate_class_prior_ = result.class_prior
        self.last_class_prior_ = result.class_prior
        self.last_class_prior_alpha_ = result.class_prior_alpha
        self.last_class_prior_laplace_logits_ = result.class_prior_laplace_logits
        self.last_class_prior_laplace_logit_variance_ = (
            result.class_prior_laplace_logit_variance
        )
        self.last_observed_laplace_logit_variance_ = result.observed_laplace_logit_variance
        self.last_local_class_accuracy_ = result.local_class_accuracy
        self.last_local_balanced_accuracy_ = result.local_balanced_accuracy
        self.last_local_corrected_balanced_accuracy_ = (
            result.local_corrected_balanced_accuracy
        )
        self.last_local_balanced_total_ = result.local_balanced_total
        self.last_local_balanced_correct_ = result.local_balanced_correct
        self.last_budget_aware_locality_ = result.budget_aware_locality
        self.last_local_kernel_top_k_ = self._local_kernel_top_k()
        self.last_local_kernel_weighting_ = self.local_kernel_weighting
        self.last_local_prior_strength_ = self._local_prior_strength()
        self.last_prior_mask_ = result.prior_mask
        self.last_evidence_mask_ = result.evidence_mask
        self.last_prior_observations_ = result.prior_observations
        self.last_uses_same_prior_and_evidence_ = result.uses_same_prior_and_evidence
        self.last_utilities_ = utilities
