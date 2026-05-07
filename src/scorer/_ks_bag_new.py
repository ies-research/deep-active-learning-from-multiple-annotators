from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.utils import check_random_state

from skactiveml.utils import MISSING_LABEL

from ._base import PairScorer


_PRESETS = {
    "global_mace": {
        "base_channel": "mace",
        "evidence": "none",
        "posterior": "base_only",
    },
    "global_full_uniform": {
        "base_channel": "uniform",
        "evidence": "global_soft_counts",
        "posterior": "dirichlet_global",
    },
    "global_full_mace_prior": {
        "base_channel": "mace",
        "evidence": "global_soft_counts",
        "posterior": "dirichlet_global",
    },
    "local_full_global_full_prior": {
        "base_channel": "global_full",
        "evidence": "local_kernel_soft_counts",
        "posterior": "dirichlet_local",
    },
    "local_full_mace_prior": {
        "base_channel": "mace",
        "evidence": "local_kernel_soft_counts",
        "posterior": "dirichlet_local",
    },
}


@dataclass
class ChannelEstimationResult:
    base_channel: np.ndarray
    evidence_counts: np.ndarray | None
    posterior_alpha: np.ndarray | None
    posterior_channel: np.ndarray
    class_prior: np.ndarray
    class_prior_alpha: np.ndarray | None
    prior_mask: np.ndarray
    evidence_mask: np.ndarray
    prior_observations: str
    uses_same_prior_and_evidence: bool
    sample_indices: np.ndarray
    annotator_indices: np.ndarray


def _normalize_axis(X: np.ndarray, *, axis: int = -1, eps: float = 1e-12):
    X = np.asarray(X, dtype=float)
    return X / np.maximum(X.sum(axis=axis, keepdims=True), eps)


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
    eps: float = 1e-12,
) -> float:
    if gamma is None or str(gamma).lower() == "median":
        X = np.asarray(X, dtype=float)
        if X.shape[0] < 2:
            return 1.0
        diff = X[:, None, :] - X[None, :, :]
        d2 = np.sum(diff * diff, axis=2)
        d2 = d2[np.triu_indices(X.shape[0], k=1)]
        d2 = d2[d2 > eps]
        if d2.size == 0:
            return 1.0
        return float(1.0 / np.median(d2))
    gamma = float(gamma)
    if gamma <= 0:
        raise ValueError("gamma must be positive, None, or 'median'.")
    return gamma


def _pairwise_kernel(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    kernel: str,
    gamma=None,
    normalize_embeddings: bool = True,
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
        gamma_value = _resolve_gamma_from_embeddings(
            X if X.shape[0] >= 2 else Y,
            gamma,
            eps=eps,
        )
        Xr = _l2_normalize(X, eps=eps) if normalize_embeddings else X
        Yr = _l2_normalize(Y, eps=eps) if normalize_embeddings else Y
        x2 = np.sum(Xr * Xr, axis=1)[:, None]
        y2 = np.sum(Yr * Yr, axis=1)[None, :]
        d2 = np.maximum(x2 + y2 - 2.0 * (Xr @ Yr.T), 0.0)
        return np.exp(-float(gamma_value) * d2)
    raise ValueError("kernel must be one of {'rbf', 'cosine'}.")


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
        eps=eps,
    )
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


def _estimate_base_uniform(n_annotators: int, n_classes: int) -> np.ndarray:
    return np.full((n_annotators, n_classes, n_classes), 1.0 / n_classes, dtype=float)


def _estimate_base_mace(
    P: np.ndarray,
    y_idx: np.ndarray,
    observation_mask: np.ndarray,
    weights: np.ndarray,
    *,
    n_annotators: int,
    n_classes: int,
    theta_prior: tuple[float, float],
    bias_prior,
    eps: float = 1e-12,
) -> np.ndarray:
    a, b = map(float, theta_prior)
    if a <= 0 or b <= 0:
        raise ValueError("mace_theta_prior entries must be positive.")
    beta = np.asarray(bias_prior, dtype=float)
    if beta.ndim == 0:
        beta = np.full(n_classes, float(beta), dtype=float)
    if beta.shape != (n_classes,) or np.any(beta <= 0):
        raise ValueError("mace_bias_prior must be a positive scalar or length-K vector.")
    B = np.empty((n_annotators, n_classes, n_classes), dtype=float)
    eye = np.eye(n_classes, dtype=float)
    for m in range(n_annotators):
        rows = np.flatnonzero(observation_mask[:, m])
        if rows.size == 0:
            theta = a / (a + b)
            g = beta / beta.sum()
        else:
            labels = y_idx[rows, m]
            w = weights[rows]
            q_y = P[rows, labels]
            theta = (a + np.sum(w * q_y)) / (a + b + np.sum(w))
            residual = w * (1.0 - q_y)
            g_num = beta.copy()
            for c in range(n_classes):
                take = labels == c
                if np.any(take):
                    g_num[c] += np.sum(residual[take])
            g = g_num / max(float(g_num.sum()), eps)
        B[m] = theta * eye + (1.0 - theta) * g[None, :]
    return _normalize_axis(B, axis=2, eps=eps)


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
    alpha0,
    eps: float = 1e-12,
) -> np.ndarray:
    alpha0 = np.asarray(alpha0, dtype=float)
    if alpha0.ndim == 0:
        alpha0 = np.full(n_classes, float(alpha0), dtype=float)
    if alpha0.shape != (n_classes,) or np.any(alpha0 <= 0):
        raise ValueError("global_full_alpha0 must be a positive scalar or length-K vector.")
    counts = _compute_global_soft_counts(
        P,
        y_idx,
        observation_mask,
        weights,
        n_annotators=n_annotators,
        n_classes=n_classes,
    )
    return _normalize_axis(counts + alpha0[None, None, :], axis=2, eps=eps)


def _combine_base_only(
    B: np.ndarray,
    *,
    n_candidates: int,
    annotator_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray | None]:
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
    B_sel = B[np.asarray(annotator_indices, dtype=int)]
    alpha = prior_strength * B_sel[None, :, :, :] + N_local
    C = _normalize_axis(alpha, axis=3, eps=eps)
    return C, alpha


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
        preset: str | None = "local_full_mace_prior",
        base_channel: str | None = None,
        evidence: str | None = None,
        posterior: str | None = None,
        utility: str = "information_gain",
        class_prior: str = "classifier",
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
        gamma=None,
        embedding_source: str = "classifier",
        normalize_embeddings: bool = True,
        use_annotator_embeddings: bool = False,
        annotator_kernel: str = "rbf",
        annotator_gamma=None,
        mace_theta_prior=(1.0, 1.0),
        mace_bias_prior=1.0,
        diag_theta_prior=(1.0, 1.0),
        global_full_alpha0=1.0,
        class_prior_strength: float = 1.0,
        class_prior_kernel: str | None = None,
        class_prior_gamma=None,
        class_prior_support: str = "evidence",
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
        self.evidence_weight = str(evidence_weight)
        self.kernel = str(kernel)
        self.gamma = gamma
        self.embedding_source = str(embedding_source)
        self.normalize_embeddings = bool(normalize_embeddings)
        self.use_annotator_embeddings = bool(use_annotator_embeddings)
        self.annotator_kernel = str(annotator_kernel)
        self.annotator_gamma = annotator_gamma
        self.mace_theta_prior = tuple(mace_theta_prior)
        self.mace_bias_prior = mace_bias_prior
        self.diag_theta_prior = tuple(diag_theta_prior)
        self.global_full_alpha0 = global_full_alpha0
        self.class_prior_strength = float(class_prior_strength)
        self.class_prior_kernel = class_prior_kernel
        self.class_prior_gamma = class_prior_gamma
        self.class_prior_support = str(class_prior_support)
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
        if preset == "local_mace":
            raise ValueError("preset='local_mace' is planned but not implemented.")
        if preset not in _PRESETS:
            raise ValueError(f"Unknown preset={preset!r}. Expected one of {sorted(_PRESETS)}.")
        return dict(_PRESETS[preset])

    def _validate_combination(self):
        if self.base_channel == "local_mace":
            raise ValueError("base_channel='local_mace' is planned but not implemented.")
        if self.base_channel not in {"uniform", "mace", "diag_uniform", "global_full"}:
            raise ValueError(
                "base_channel must be one of "
                "{'uniform', 'mace', 'diag_uniform', 'global_full'}."
            )
        if self.evidence not in {"none", "global_soft_counts", "local_kernel_soft_counts"}:
            raise ValueError(
                "evidence must be one of "
                "{'none', 'global_soft_counts', 'local_kernel_soft_counts'}."
            )
        if self.posterior not in {"base_only", "dirichlet_global", "dirichlet_local"}:
            raise ValueError(
                "posterior must be one of "
                "{'base_only', 'dirichlet_global', 'dirichlet_local'}."
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
        if self.prior_strength <= 0:
            raise ValueError("prior_strength must be > 0.")
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
        }:
            raise ValueError(
                "utility must be one of "
                "{'expected_accuracy', 'bias_corrected_accuracy', 'information_gain'}."
            )
        if self.class_prior not in {"classifier", "uniform", "top_m", "kernel"}:
            raise ValueError(
                "class_prior must be one of {'classifier', 'uniform', 'top_m', 'kernel'}."
            )
        if self.class_prior == "top_m":
            if self.top_m is None:
                raise ValueError("class_prior='top_m' requires top_m.")
        elif self.top_m is not None:
            raise ValueError("top_m is only used when class_prior='top_m'.")
        if self.class_prior_strength <= 0:
            raise ValueError("class_prior_strength must be > 0.")
        if self.class_prior_support not in {"evidence", "observed"}:
            raise ValueError("class_prior_support must be one of {'evidence', 'observed'}.")
        if self.embedding_source not in {"classifier", "input"}:
            raise ValueError("embedding_source must be one of {'classifier', 'input'}.")
        if self.n_mc_samples < 0:
            raise ValueError("n_mc_samples must be >= 0.")
        if (self.sample_class_prior or self.sample_channel) and self.n_mc_samples <= 0:
            raise ValueError("Sampling requires n_mc_samples > 0.")
        if self.sample_class_prior and self.class_prior != "kernel":
            raise ValueError("sample_class_prior=True requires class_prior='kernel'.")
        if self.sample_channel and self.posterior == "base_only":
            raise ValueError("sample_channel=True requires a Dirichlet posterior.")
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
        self.last_evidence_counts_ = None
        self.last_posterior_alpha_ = None
        self.last_posterior_channel_ = None
        self.last_posterior_concentration_ = None
        self.last_class_prior_ = None
        self.last_class_prior_alpha_ = None
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
        P_all, sample_embeddings, annotator_embeddings = self._predict_probabilities_and_embeddings(
            X=X,
            clf=clf,
            need_sample_embeddings=self._needs_sample_embeddings(),
            need_annotator_embeddings=self.use_annotator_embeddings,
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
        weights = _compute_evidence_weights(P_all, self.evidence_weight, eps=self.eps)
        annotator_similarity = self._resolve_annotator_similarity(
            annotator_embeddings,
            n_annotators=y.shape[1],
        )
        B = self._estimate_base_channel(
            P=P_all,
            y_idx=y_idx,
            prior_mask=prior_mask_resolved,
            weights=weights,
            n_annotators=y.shape[1],
            n_classes=K,
        )
        evidence_counts = self._compute_evidence_counts(
            P=P_all,
            y_idx=y_idx,
            evidence_mask=evidence_mask_resolved,
            weights=weights,
            sample_embeddings=sample_embeddings,
            sample_indices=sample_indices,
            annotator_indices=annotator_indices,
            n_annotators=y.shape[1],
            n_classes=K,
            annotator_similarity=annotator_similarity,
        )
        C, alpha = self._combine_posterior(
            B=B,
            evidence_counts=evidence_counts,
            sample_indices=sample_indices,
            annotator_indices=annotator_indices,
        )
        p, class_alpha = self._resolve_candidate_class_prior(
            P=P_all,
            sample_embeddings=sample_embeddings,
            sample_indices=sample_indices,
            evidence_mask=evidence_mask_resolved,
            observed_mask=_make_observation_mask(y, missing_label=missing_label),
            n_classes=K,
        )
        uses_same = np.array_equal(prior_mask_resolved, evidence_mask_resolved)
        result = ChannelEstimationResult(
            base_channel=B,
            evidence_counts=evidence_counts,
            posterior_alpha=alpha,
            posterior_channel=C,
            class_prior=p,
            class_prior_alpha=class_alpha,
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
            self.evidence == "local_kernel_soft_counts"
            or self.class_prior == "kernel"
        )

    def _predict_probabilities_and_embeddings(
        self,
        *,
        X,
        clf,
        need_sample_embeddings: bool,
        need_annotator_embeddings: bool,
    ):
        extra = []
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
        return P, sample_embeddings, annotator_embeddings

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
            eps=self.eps,
        )

    def _estimate_base_channel(
        self,
        *,
        P,
        y_idx,
        prior_mask,
        weights,
        n_annotators,
        n_classes,
    ) -> np.ndarray:
        if self.base_channel == "uniform":
            return _estimate_base_uniform(n_annotators, n_classes)
        if self.base_channel == "mace":
            return _estimate_base_mace(
                P,
                y_idx,
                prior_mask,
                weights,
                n_annotators=n_annotators,
                n_classes=n_classes,
                theta_prior=self.mace_theta_prior,
                bias_prior=self.mace_bias_prior,
                eps=self.eps,
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
            )
        if self.base_channel == "global_full":
            return _estimate_base_global_full(
                P,
                y_idx,
                prior_mask,
                weights,
                n_annotators=n_annotators,
                n_classes=n_classes,
                alpha0=self.global_full_alpha0,
                eps=self.eps,
            )
        raise RuntimeError("unreachable base_channel")

    def _compute_evidence_counts(
        self,
        *,
        P,
        y_idx,
        evidence_mask,
        weights,
        sample_embeddings,
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
                annotator_similarity=annotator_similarity,
                eps=self.eps,
            )
        raise RuntimeError("unreachable evidence")

    def _combine_posterior(
        self,
        *,
        B,
        evidence_counts,
        sample_indices,
        annotator_indices,
    ):
        if self.posterior == "base_only":
            return _combine_base_only(
                B,
                n_candidates=len(sample_indices),
                annotator_indices=annotator_indices,
            )
        if self.posterior == "dirichlet_global":
            return _combine_dirichlet_global(
                B,
                evidence_counts,
                prior_strength=self.prior_strength,
                n_candidates=len(sample_indices),
                annotator_indices=annotator_indices,
                eps=self.eps,
            )
        if self.posterior == "dirichlet_local":
            return _combine_dirichlet_local(
                B,
                evidence_counts,
                prior_strength=self.prior_strength,
                annotator_indices=annotator_indices,
                eps=self.eps,
            )
        raise RuntimeError("unreachable posterior")

    def _resolve_candidate_class_prior(
        self,
        *,
        P,
        sample_embeddings,
        sample_indices,
        evidence_mask,
        observed_mask,
        n_classes,
    ) -> tuple[np.ndarray, np.ndarray | None]:
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
            alpha0 = np.full(n_classes, self.class_prior_strength / n_classes, dtype=float)
            if support.size == 0:
                alpha = np.broadcast_to(alpha0[None, :], P_cand.shape).copy()
            else:
                kernel_name = self.class_prior_kernel or self.kernel
                gamma = self.class_prior_gamma if self.class_prior_gamma is not None else self.gamma
                Kx = _pairwise_kernel(
                    sample_embeddings[support],
                    sample_embeddings[sample_indices],
                    kernel=kernel_name,
                    gamma=gamma,
                    normalize_embeddings=self.normalize_embeddings,
                    eps=self.eps,
                )
                alpha = alpha0[None, :] + Kx.T @ P[support]
            return _normalize_axis(alpha, axis=1, eps=self.eps), alpha
        raise RuntimeError("unreachable class_prior")

    def _utilities_from_result(self, result: ChannelEstimationResult) -> np.ndarray:
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
                p = self._sample_dirichlet_rows(result.class_prior_alpha, rng)
            if self.sample_channel:
                C = self._sample_dirichlet_rows(result.posterior_alpha, rng)
            draws[t] = self._compute_utility(p, C)
        self.last_utility_mean_ = np.mean(draws, axis=0)
        self.last_utility_std_ = np.std(draws, axis=0)
        self.last_utility_lcb_ = np.quantile(draws, 0.05, axis=0) if n_draws > 1 else None
        self.last_utility_ucb_ = np.quantile(draws, 0.95, axis=0) if n_draws > 1 else None
        self.last_utility_draws_ = draws if self.store_utility_draws else None
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
        raise RuntimeError("unreachable utility")

    def _sample_dirichlet_rows(self, alpha: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        if alpha is None:
            raise ValueError("Cannot sample Dirichlet rows without alpha.")
        alpha = np.clip(np.asarray(alpha, dtype=float), self.eps, None)
        samples = rng.gamma(shape=alpha, scale=1.0)
        return _normalize_axis(samples, axis=-1, eps=self.eps)

    def _store_result_diagnostics(
        self,
        result: ChannelEstimationResult,
        utilities: np.ndarray,
    ):
        self.last_result_ = result
        self.last_base_channel_ = result.base_channel
        self.last_evidence_counts_ = result.evidence_counts
        self.last_posterior_alpha_ = result.posterior_alpha
        self.last_posterior_channel_ = result.posterior_channel
        self.last_posterior_concentration_ = (
            None
            if result.posterior_alpha is None
            else result.posterior_alpha.sum(axis=-1)
        )
        self.last_class_prior_ = result.class_prior
        self.last_class_prior_alpha_ = result.class_prior_alpha
        self.last_prior_mask_ = result.prior_mask
        self.last_evidence_mask_ = result.evidence_mask
        self.last_prior_observations_ = result.prior_observations
        self.last_uses_same_prior_and_evidence_ = result.uses_same_prior_and_evidence
        self.last_utilities_ = utilities
