import numpy as np


def _normalize_axis(X: np.ndarray, *, axis: int, eps: float) -> np.ndarray:
    return X / np.maximum(X.sum(axis=axis, keepdims=True), eps)


def information_gain(
    P: np.ndarray,
    P_perf: np.ndarray | None = None,
    P_annot: np.ndarray | None = None,
    *,
    C: np.ndarray | None = None,
    eps: float = 1e-12,
    log_base: float = 2.0,
    normalize: bool = True,
    check_input: bool = True,
    batch_size: int | None = None,
) -> np.ndarray:
    """
    Compute expected information gain in either of two modes:

    - `channel` mode (legacy): pass `P_perf` and `P_annot`.
    - `confusion` mode: pass `C` directly.

    Parameters
    ----------
    P : np.ndarray
        Class posterior array:
        - channel mode: shape `(n_samples, n_classes)`.
        - confusion mode: shape `(..., n_classes)`.
    P_perf : np.ndarray or None, default=None
        Channel mode only. Correctness probabilities `theta` with shape
        `(n_samples, n_annotators)`.
    P_annot : np.ndarray or None, default=None
        Channel mode only. Fallback label distribution `g` with shape
        `(n_samples, n_annotators, n_classes)`.
    C : np.ndarray or None, default=None
        Confusion mode only. Confusion matrices with shape
        `(..., n_classes, n_classes)`.
    eps : float, default=1e-12
        Numerical stability constant for clipping before logs/divisions.
    log_base : float, default=2.0
        Logarithm base. Use 2.0 for bits, np.e for nats.
    normalize : bool, default=True
        If True, normalize probability inputs along their stochastic axes.
    check_input : bool, default=True
        If True, validate shapes and basic ranges.
    batch_size : int or None, default=None
        Optional number of chunks along axis 0. If None, compute in one vectorized
        pass. Set for lower peak memory on large arrays.

    Returns
    -------
    IG : np.ndarray
        Information gain with shape:
        - channel mode: `(n_samples, n_annotators)`.
        - confusion mode: `P.shape[:-1]`.
    """
    if batch_size is not None and batch_size <= 0:
        raise ValueError("batch_size must be > 0 when provided.")
    if C is not None:
        if P_perf is not None or P_annot is not None:
            raise ValueError(
                "Use either confusion mode (`C`) or channel mode "
                "(`P_perf` and `P_annot`), not both."
            )

        r = np.asarray(P, dtype=float)
        C = np.asarray(C, dtype=float)
        if batch_size is None or r.ndim == 1:
            return _information_gain_from_confusion_batch(
                r=r,
                C=C,
                eps=eps,
                log_base=log_base,
                normalize=normalize,
                check_input=check_input,
            )

        n0 = r.shape[0]
        IG = np.empty(r.shape[:-1], dtype=float)
        for start in range(0, n0, batch_size):
            stop = min(start + batch_size, n0)
            IG[start:stop] = _information_gain_from_confusion_batch(
                r=r[start:stop],
                C=C[start:stop],
                eps=eps,
                log_base=log_base,
                normalize=normalize,
                check_input=check_input,
            )
        return IG

    if (P_perf is None) != (P_annot is None):
        raise ValueError(
            "Channel mode requires both `P_perf` and `P_annot`, or use `C`."
        )
    if P_perf is None and P_annot is None:
        raise ValueError(
            "Missing inputs. Provide either (`P_perf`, `P_annot`) or `C`."
        )

    P = np.asarray(P, dtype=float)
    P_perf = np.asarray(P_perf, dtype=float)
    P_annot = np.asarray(P_annot, dtype=float)

    if check_input:
        if P.ndim != 2:
            raise ValueError(
                f"P must be 2D (n_samples, n_classes), got shape {P.shape}."
            )
        if P_perf.ndim != 2:
            raise ValueError(
                f"P_perf must be 2D (n_samples, n_annotators), got {P_perf.shape}."
            )
        if P_annot.ndim != 3:
            raise ValueError(
                f"P_annot must be 3D (n_samples, n_annotators, n_classes), got {P_annot.shape}."
            )
        n, k = P.shape
        if P_perf.shape[0] != n:
            raise ValueError("P_perf must have same n_samples as P.")
        if P_annot.shape[0] != n:
            raise ValueError("P_annot must have same n_samples as P.")
        if P_annot.shape[2] != k:
            raise ValueError("P_annot must have same n_classes as P.")
        if P_annot.shape[1] != P_perf.shape[1]:
            raise ValueError("P_annot and P_perf must have same n_annotators.")
        if np.any((P_perf < -eps) | (P_perf > 1 + eps)):
            raise ValueError("P_perf must be in [0, 1].")

    if normalize:
        P = _normalize_axis(P, axis=1, eps=eps)
        P_annot = _normalize_axis(P_annot, axis=2, eps=eps)

    def _compute_chunk(P_chunk, P_perf_chunk, P_annot_chunk):
        C_chunk = _channel_confusion_from_theta_g_batch(
            theta=P_perf_chunk,
            g=P_annot_chunk,
            eps=eps,
            normalize_g=False,
            check_input=False,
        )
        r_chunk = np.broadcast_to(P_chunk[:, None, :], P_annot_chunk.shape)
        return _information_gain_from_confusion_batch(
            r=r_chunk,
            C=C_chunk,
            eps=eps,
            log_base=log_base,
            normalize=False,
            check_input=False,
        )

    n, _ = P_perf.shape
    if batch_size is None:
        return _compute_chunk(P, P_perf, P_annot)

    IG = np.empty(P_perf.shape, dtype=float)
    for start in range(0, n, batch_size):
        stop = min(start + batch_size, n)
        IG[start:stop] = _compute_chunk(
            P[start:stop], P_perf[start:stop], P_annot[start:stop]
        )
    return IG


def _channel_confusion_from_theta_g_batch(
    *,
    theta: np.ndarray,
    g: np.ndarray,
    eps: float = 1e-12,
    normalize_g: bool = True,
    check_input: bool = True,
) -> np.ndarray:
    """
    Build confusion matrices for batch inputs under the (theta, g) channel model.

    Parameters
    ----------
    theta : np.ndarray of shape (...)
        Probability of correct label.
    g : np.ndarray of shape (..., K)
        Fallback label distribution used when incorrect.
    eps : float, default=1e-12
        Numerical stability constant.
    normalize_g : bool, default=True
        If True, normalize `g` on the class axis before conversion.
    check_input : bool, default=True
        If True, validate input shapes.

    Returns
    -------
    C : np.ndarray of shape (..., K, K)
        Confusion matrices with rows indexed by true class and columns by response.
    """
    theta = np.asarray(theta, dtype=float)
    g = np.asarray(g, dtype=float)

    if check_input:
        if g.ndim < 1:
            raise ValueError("g must have shape (..., K).")
        if theta.shape != g.shape[:-1]:
            raise ValueError(
                f"theta shape {theta.shape} must match g leading shape {g.shape[:-1]}."
            )
        if g.shape[-1] < 2:
            raise ValueError("K must be >= 2 for channel confusion.")

    K = g.shape[-1]
    if K < 2:
        raise ValueError("K must be >= 2 for channel confusion.")

    theta = np.clip(theta, 0.0, 1.0)
    g = np.clip(g, eps, 1.0)
    if normalize_g:
        g = _normalize_axis(g, axis=-1, eps=eps)

    off = np.broadcast_to(g[..., None, :], g.shape[:-1] + (K, K)).copy()
    idx = np.arange(K)
    off[..., idx, idx] = 0.0

    off_mass = off.sum(axis=-1, keepdims=True)
    fallback = (
        (np.ones((K, K), dtype=float) - np.eye(K, dtype=float)) / (K - 1)
    )
    fallback = fallback.reshape((1,) * (off.ndim - 2) + (K, K))
    off = np.where(off_mass > eps, off / np.maximum(off_mass, eps), fallback)

    C = (1.0 - theta)[..., None, None] * off
    C[..., idx, idx] = np.broadcast_to(theta[..., None], C[..., idx, idx].shape)
    return C


def _information_gain_from_confusion_batch(
    *,
    r: np.ndarray,
    C: np.ndarray,
    eps: float = 1e-12,
    log_base: float = 2.0,
    normalize: bool = True,
    check_input: bool = True,
) -> np.ndarray:
    """
    Vectorized IG from class posteriors and confusion matrices.

    Parameters
    ----------
    r : np.ndarray of shape (..., K)
        Class posterior distributions.
    C : np.ndarray of shape (..., K, K)
        Confusion matrices.
    eps : float, default=1e-12
        Numerical stability constant.
    log_base : float, default=2.0
        Logarithm base.
    normalize : bool, default=True
        If True, normalize `r` and rows of `C`.
    check_input : bool, default=True
        If True, validate input shapes.

    Returns
    -------
    IG : np.ndarray of shape (...)
        Information gain for each leading batch element.
    """
    r = np.asarray(r, dtype=float)
    C = np.asarray(C, dtype=float)

    if check_input:
        if r.ndim < 1:
            raise ValueError(f"r must have shape (..., K), got {r.shape}.")
        if C.ndim < 2:
            raise ValueError(
                f"C must have shape (..., K, K), got {C.shape}."
            )
        if C.shape[-2] != C.shape[-1]:
            raise ValueError(f"C must be square on the last two axes, got {C.shape}.")
        if r.shape[:-1] != C.shape[:-2]:
            raise ValueError(
                f"r leading shape {r.shape[:-1]} must match C leading shape {C.shape[:-2]}."
            )
        if r.shape[-1] != C.shape[-1]:
            raise ValueError(
                f"Shape mismatch: r {r.shape}, C {C.shape}. Expected C (...,K,K)."
            )

    r = np.clip(r, eps, 1.0)
    C = np.clip(C, eps, 1.0)
    if normalize:
        r = _normalize_axis(r, axis=-1, eps=eps)
        C = _normalize_axis(C, axis=-1, eps=eps)

    py = np.einsum("...k,...ky->...y", r, C)
    py = _normalize_axis(np.clip(py, eps, 1.0), axis=-1, eps=eps)

    joint = r[..., :, None] * C
    post = joint / np.maximum(joint.sum(axis=-2, keepdims=True), eps)

    log_denom = np.log(log_base) if log_base != np.e else 1.0
    H_prior = -(r * (np.log(np.clip(r, eps, 1.0)) / log_denom)).sum(axis=-1)
    H_post = -(
        post * (np.log(np.clip(post, eps, 1.0)) / log_denom)
    ).sum(axis=-2)
    H_cond = (py * H_post).sum(axis=-1)

    return np.maximum(H_prior - H_cond, 0.0)
