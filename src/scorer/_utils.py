import numpy as np

def information_gain(
    P: np.ndarray,
    P_perf: np.ndarray,
    P_annot: np.ndarray,
    *,
    eps: float = 1e-12,
    log_base: float = 2.0,
    normalize: bool = True,
    check_input: bool = True,
) -> np.ndarray:
    """
    Compute expected information gain for querying an annotator on a sample.

    This returns, for each (sample n, annotator m), the mutual information
        IG[n, m] = I(Z; Y | x_n, annotator m)
    where Z is the latent true class and Y is the annotator's response.

    Channel model (per sample n, annotator m):
        Z ~ Categorical(P[n])
        With prob theta = P_perf[n, m]:    Y = Z
        With prob 1 - theta:              Y ~ Categorical(g), g = P_annot[n, m, :]

    Parameters
    ----------
    P : np.ndarray of shape (n_samples, n_classes)
        Class probabilities p(Z=k | x_n). Rows should sum to 1.
    P_perf : np.ndarray of shape (n_samples, n_annotators)
        Correctness probabilities theta(n,m) in [0, 1].
    P_annot : np.ndarray of shape (n_samples, n_annotators, n_classes)
        Fallback label distribution g(n,m,·) used when the annotator is incorrect.
        Each (n,m) slice should sum to 1 (will be normalized if `normalize=True`).
    eps : float, default=1e-12
        Numerical stability constant for clipping before logs/divisions.
    log_base : float, default=2.0
        Logarithm base. Use 2.0 for bits, np.e for nats.
    normalize : bool, default=True
        If True, renormalize P and P_annot along the class axis.
    check_input : bool, default=True
        If True, validate shapes and basic ranges.

    Returns
    -------
    IG : np.ndarray of shape (n_samples, n_annotators)
        Expected information gain I(Z;Y) for each sample-annotator pair.
    """
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

    def _normalize_rows(X: np.ndarray, axis: int) -> np.ndarray:
        s = X.sum(axis=axis, keepdims=True)
        return X / np.maximum(s, eps)

    if normalize:
        P = _normalize_rows(P, axis=1)
        P_annot = _normalize_rows(P_annot, axis=2)

    # Convert logs to desired base.
    log_denom = np.log(log_base) if log_base != np.e else 1.0

    def _log(x: np.ndarray) -> np.ndarray:
        return np.log(np.clip(x, eps, 1.0)) / log_denom

    # Prior entropy per sample: H(Z|x) = -sum_k P log P
    S = (P * _log(P)).sum(axis=1)  # S = sum P log P  (<= 0)
    H_prior = -S  # (n_samples,)

    # Broadcast shapes:
    # P:      (n, 1, k)
    # theta:  (n, m, 1)
    # g:      (n, m, k)
    P_b = P[:, None, :]
    theta = P_perf[:, :, None]
    g = P_annot

    # Predictive distribution of annotator response:
    # p(Y=j) = theta * P_j + (1-theta) * g_j
    denom = theta * P_b + (1.0 - theta) * g
    denom = np.clip(denom, eps, 1.0)  # stability

    # a_j = (1-theta) g_j / denom_j
    a = (1.0 - theta) * g / denom

    # base_j = a_j * P_j
    base_j = a * P_b

    # new_j = base_j + theta * P_j / denom_j
    new_j = base_j + (theta * P_b) / denom

    # Efficient conditional entropy without building KxK matrices:
    # For each observed j:
    #   H(Z|Y=j) = -[ sum_k post_k log post_k ]
    # where post_k = a_j P_k for k!=j, and post_j = a_j P_j + theta P_j/denom_j (= new_j).
    #
    # sum_k (a_j P_k) log(a_j P_k) = a_j log a_j + a_j * sum_k P_k log P_k = a_j log a_j + a_j * S
    S_b = S[:, None, None]  # (n,1,1)

    base_total = a * _log(a) + a * S_b
    term = base_total - base_j * _log(base_j) + new_j * _log(new_j)
    H_given_y = -term  # (n, m, k): H(Z | Y=j)
    H_cond = (denom * H_given_y).sum(axis=2)  # (n, m): E_Y[H(Z|Y)]

    IG = H_prior[:, None] - H_cond
    # Numerical noise can produce tiny negatives around 0.
    return np.maximum(IG, 0.0)
