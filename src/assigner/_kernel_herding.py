from __future__ import annotations

import numpy as np
from sklearn.utils import check_random_state

from skactiveml.utils import is_labeled

from ._base import PairAssigner


def _l2_normalize(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    norm = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(norm, eps)


class KernelHerdingPairAssigner(PairAssigner):
    """
    Kernel herding assigner over a local sample-annotator utility block.

    Non-negative pair utilities are direct relevance weights by default. At
    each step, the assigner selects the feasible pair maximizing relevance
    times marginal coverage gain under the additive pair kernel.
    """

    def __init__(
        self,
        *,
        instance_kernel: str = "rbf",
        annotator_kernel: str = "dirac",
        rbf_gamma: float | str | None = None,
        rbf_gamma_scope: str = "current_and_labeled",
        gamma_max_samples: int | None = 4096,
        instance_weight: float = 0.5,
        objective: str = "gain_coverage",
        history_coverage: str = "samples",
        history_chunk_size: int = 4096,
        history_max_samples: int | None = None,
        max_pair_uherding_pairs: int = 20000,
        normalize_embeddings: bool = True,
        random_state=None,
    ):
        self.instance_kernel = str(instance_kernel)
        self.annotator_kernel = str(annotator_kernel)
        self.rbf_gamma = rbf_gamma
        self.rbf_gamma_scope = str(rbf_gamma_scope)
        self.gamma_max_samples = (
            None if gamma_max_samples is None else int(gamma_max_samples)
        )
        self.instance_weight = float(instance_weight)
        self.objective = str(objective)
        self.history_coverage = str(history_coverage)
        self.history_chunk_size = int(history_chunk_size)
        self.history_max_samples = (
            None if history_max_samples is None else int(history_max_samples)
        )
        self.max_pair_uherding_pairs = int(max_pair_uherding_pairs)
        self.normalize_embeddings = bool(normalize_embeddings)
        self.random_state = check_random_state(random_state)

        if self.instance_kernel not in {"rbf", "cosine"}:
            raise ValueError("instance_kernel must be one of {'rbf', 'cosine'}.")
        if self.annotator_kernel != "dirac":
            raise ValueError("Only annotator_kernel='dirac' is supported in v1.")
        if self.rbf_gamma_scope not in {"current", "labeled", "current_and_labeled"}:
            raise ValueError(
                "rbf_gamma_scope must be one of "
                "{'current', 'labeled', 'current_and_labeled'}."
            )
        if self.gamma_max_samples is not None and self.gamma_max_samples <= 1:
            raise ValueError("gamma_max_samples must be > 1 or None.")
        if not (0.0 <= self.instance_weight <= 1.0):
            raise ValueError("instance_weight must be in [0, 1].")
        if self.objective not in {"gain_coverage", "pair_uherding"}:
            raise ValueError(
                "objective must be one of {'gain_coverage', 'pair_uherding'}."
            )
        if self.history_coverage not in {"none", "samples"}:
            raise ValueError("history_coverage must be one of {'none', 'samples'}.")
        if self.history_chunk_size <= 0:
            raise ValueError("history_chunk_size must be > 0.")
        if self.history_max_samples is not None and self.history_max_samples <= 0:
            raise ValueError("history_max_samples must be positive or None.")
        if self.max_pair_uherding_pairs <= 0:
            raise ValueError("max_pair_uherding_pairs must be > 0.")

        self.last_herding_scores_ = None
        self.last_target_distribution_ = None
        self.last_pair_utilities_ = None
        self.last_coverage_gains_ = None
        self.last_initial_sample_cover_ = None

    def _assign(
        self,
        utilities,
        sample_indices,
        annotator_indices,
        budget,
        X=None,
        clf=None,
        X_embed=None,
        annotator_embed=None,
        annotator_remaining_counts=None,
        y=None,
        missing_label=np.nan,
        **kwargs,
    ):
        U = np.asarray(utilities, dtype=float)
        sample_indices = np.asarray(sample_indices, dtype=int)
        annotator_indices = np.asarray(annotator_indices, dtype=int)
        budget = int(budget)

        self.last_herding_scores_ = np.empty((0,) + U.shape, dtype=float)
        self.last_coverage_gains_ = np.empty((0,) + U.shape, dtype=float)
        self.last_target_distribution_ = np.zeros_like(U, dtype=float)
        self.last_pair_utilities_ = np.where(
            np.isfinite(U), np.maximum(U, 0.0), 0.0
        )
        self.last_initial_sample_cover_ = np.zeros(U.shape[0], dtype=float)

        S, A = U.shape
        if budget <= 0 or S == 0 or A == 0:
            return np.empty((0, 2), dtype=int)

        feasible = np.isfinite(U)
        remaining = self._coerce_annotator_remaining(
            annotator_indices=annotator_indices,
            annotator_remaining_counts=annotator_remaining_counts,
        )
        if remaining is not None:
            feasible &= remaining[None, :] > 0
        if not feasible.any():
            return np.empty((0, 2), dtype=int)

        gamma = self._resolve_call_rbf_gamma(
            X=X,
            sample_indices=sample_indices,
            clf=clf,
            X_embed=X_embed,
            y=y,
            missing_label=missing_label,
        )
        Kx = self._compute_instance_kernel(
            X=X,
            sample_indices=sample_indices,
            clf=clf,
            X_embed=X_embed,
            gamma=gamma,
        )
        Ka = self._compute_annotator_kernel(
            annotator_indices=annotator_indices,
            annotator_embed=annotator_embed,
        )

        target_distribution = self._target_distribution(U=U, feasible=feasible)
        self.last_target_distribution_ = target_distribution.copy()
        utility_weight = self._utility_weight(U=U, feasible=feasible)

        wx = self.instance_weight
        wa = 1.0 - wx

        if self.objective == "pair_uherding":
            return self._assign_pair_uherding(
                U=U,
                utility_weight=utility_weight,
                feasible=feasible,
                remaining=remaining,
                sample_indices=sample_indices,
                annotator_indices=annotator_indices,
                budget=budget,
                Kx=Kx,
                Ka=Ka,
                wx=wx,
                wa=wa,
            )

        selected = []
        selected_mask = np.zeros_like(feasible, dtype=bool)
        sample_cover = self._initial_sample_cover(
            X=X,
            sample_indices=sample_indices,
            X_embed=X_embed,
            clf=clf,
            y=y,
            missing_label=missing_label,
            gamma=gamma,
        )
        self.last_initial_sample_cover_ = sample_cover.copy()
        annotator_cover = np.zeros(A, dtype=float)
        batch_a = np.zeros(A, dtype=int)
        score_history = []
        coverage_history = []

        for _ in range(budget):
            step_feasible = feasible & ~selected_mask
            if remaining is not None:
                step_feasible &= batch_a[None, :] < remaining[None, :]
            if not step_feasible.any():
                break

            coverage_gain = self._decomposed_coverage_gain(
                feasible=feasible,
                Kx=Kx,
                Ka=Ka,
                sample_cover=sample_cover,
                annotator_cover=annotator_cover,
                instance_weight=wx,
                annotator_weight=wa,
            )
            score = utility_weight * coverage_gain
            score = np.where(step_feasible, score, -np.inf)
            coverage_history.append(np.where(step_feasible, coverage_gain, np.nan))
            score_history.append(score.copy())
            flat_idx = int(np.argmax(score))
            if not np.isfinite(score.ravel()[flat_idx]):
                break
            s_loc, a_loc = np.unravel_index(flat_idx, U.shape)

            selected.append(
                (int(sample_indices[s_loc]), int(annotator_indices[a_loc]))
            )
            selected_mask[s_loc, a_loc] = True
            sample_cover = np.maximum(sample_cover, Kx[:, s_loc])
            annotator_cover = np.maximum(annotator_cover, Ka[:, a_loc])
            batch_a[a_loc] += 1

        if score_history:
            self.last_herding_scores_ = np.stack(score_history, axis=0)
            self.last_coverage_gains_ = np.stack(coverage_history, axis=0)
        else:
            self.last_herding_scores_ = np.empty((0,) + U.shape, dtype=float)
            self.last_coverage_gains_ = np.empty((0,) + U.shape, dtype=float)
        return np.asarray(selected, dtype=int).reshape(-1, 2)

    def _assign_pair_uherding(
        self,
        *,
        U,
        utility_weight,
        feasible,
        remaining,
        sample_indices,
        annotator_indices,
        budget,
        Kx,
        Ka,
        wx,
        wa,
    ):
        n_feasible = int(np.sum(feasible))
        if n_feasible > self.max_pair_uherding_pairs:
            raise ValueError(
                "objective='pair_uherding' is exact-only and received "
                f"{n_feasible} feasible pairs, which exceeds "
                f"max_pair_uherding_pairs={self.max_pair_uherding_pairs}. "
                "Use objective='gain_coverage' or reduce the candidate block."
            )

        selected = []
        selected_mask = np.zeros_like(feasible, dtype=bool)
        current_cover = np.zeros_like(U, dtype=float)
        batch_a = np.zeros(U.shape[1], dtype=int)
        score_history = []
        coverage_history = []

        for _ in range(budget):
            step_feasible = feasible & ~selected_mask
            if remaining is not None:
                step_feasible &= batch_a[None, :] < remaining[None, :]
            if not step_feasible.any():
                break

            score = self._pair_uherding_scores(
                utility_weight=utility_weight,
                feasible=feasible,
                step_feasible=step_feasible,
                current_cover=current_cover,
                Kx=Kx,
                Ka=Ka,
                instance_weight=wx,
                annotator_weight=wa,
            )
            score = np.where(step_feasible, score, -np.inf)
            coverage_history.append(np.where(step_feasible, score, np.nan))
            score_history.append(score.copy())
            flat_idx = int(np.argmax(score))
            if not np.isfinite(score.ravel()[flat_idx]):
                break
            s_loc, a_loc = np.unravel_index(flat_idx, U.shape)

            selected.append(
                (int(sample_indices[s_loc]), int(annotator_indices[a_loc]))
            )
            selected_mask[s_loc, a_loc] = True
            selected_kernel = (
                wx * Kx[:, s_loc][:, None]
                + wa * Ka[:, a_loc][None, :]
            )
            current_cover = np.maximum(current_cover, selected_kernel)
            batch_a[a_loc] += 1

        if score_history:
            self.last_herding_scores_ = np.stack(score_history, axis=0)
            self.last_coverage_gains_ = np.stack(coverage_history, axis=0)
        else:
            self.last_herding_scores_ = np.empty((0,) + U.shape, dtype=float)
            self.last_coverage_gains_ = np.empty((0,) + U.shape, dtype=float)
        return np.asarray(selected, dtype=int).reshape(-1, 2)

    @staticmethod
    def _coerce_annotator_remaining(annotator_indices, annotator_remaining_counts):
        if annotator_remaining_counts is None:
            return None
        arr = np.asarray(annotator_remaining_counts)
        if arr.ndim != 1:
            raise ValueError("annotator_remaining_counts must be one-dimensional.")
        if arr.shape[0] == len(annotator_indices):
            return arr.astype(int, copy=False)
        max_idx = int(annotator_indices.max(initial=-1))
        if arr.shape[0] <= max_idx:
            raise ValueError(
                "annotator_remaining_counts must be in local annotator order or "
                "global annotator order."
            )
        return arr[annotator_indices].astype(int, copy=False)

    def _target_distribution(self, *, U: np.ndarray, feasible: np.ndarray) -> np.ndarray:
        util = np.zeros_like(U, dtype=float)
        finite = feasible & np.isfinite(U)
        util[finite] = np.maximum(U[finite], 0.0)
        total = float(util.sum())
        if total > 0.0 and np.isfinite(total):
            return util / total

        p = np.zeros_like(U, dtype=float)
        p[feasible] = 1.0 / float(np.sum(feasible))
        return p

    def _utility_weight(self, *, U: np.ndarray, feasible: np.ndarray) -> np.ndarray:
        util = np.zeros_like(U, dtype=float)
        finite = feasible & np.isfinite(U)
        util[finite] = np.maximum(U[finite], 0.0)
        total = float(util.sum())
        if total > 0.0 and np.isfinite(total):
            return util

        util[feasible] = 1.0
        return util

    @staticmethod
    def _decomposed_coverage_gain(
        *,
        feasible,
        Kx,
        Ka,
        sample_cover,
        annotator_cover,
        instance_weight,
        annotator_weight,
    ) -> np.ndarray:
        sample_gain = np.maximum(Kx - sample_cover[:, None], 0.0).sum(axis=0)
        annotator_gain = np.maximum(
            Ka - annotator_cover[:, None], 0.0
        ).sum(axis=0)
        coverage = (
            instance_weight * sample_gain[:, None]
            + annotator_weight * annotator_gain[None, :]
        )
        return np.where(feasible, coverage, 0.0)

    @staticmethod
    def _pair_uherding_scores(
        *,
        utility_weight,
        feasible,
        step_feasible,
        current_cover,
        Kx,
        Ka,
        instance_weight,
        annotator_weight,
    ) -> np.ndarray:
        S, A = step_feasible.shape
        scores = np.full((S, A), -np.inf, dtype=float)
        covered_weight = np.where(feasible, utility_weight, 0.0)
        for s in range(S):
            kx_col = Kx[:, s][:, None]
            for a in np.flatnonzero(step_feasible[s]):
                pair_kernel = (
                    instance_weight * kx_col
                    + annotator_weight * Ka[:, a][None, :]
                )
                marginal = np.maximum(pair_kernel - current_cover, 0.0)
                scores[s, a] = float(np.sum(covered_weight * marginal))
        return scores

    def _initial_sample_cover(
        self,
        *,
        X,
        sample_indices,
        X_embed,
        clf,
        y,
        missing_label,
        gamma,
    ) -> np.ndarray:
        n_samples = len(sample_indices)
        if self.history_coverage == "none" or y is None:
            return np.zeros(n_samples, dtype=float)

        hist_indices = self._historical_sample_indices(
            y=y,
            sample_indices=sample_indices,
            missing_label=missing_label,
        )
        if hist_indices.size == 0:
            return np.zeros(n_samples, dtype=float)

        hist_indices = self._cap_history_indices(hist_indices)
        E_cand = self._resolve_sample_embeddings(
            X=X,
            sample_indices=sample_indices,
            clf=clf,
            X_embed=X_embed,
        )
        cover = np.zeros(n_samples, dtype=float)
        for start in range(0, hist_indices.size, self.history_chunk_size):
            stop = min(start + self.history_chunk_size, hist_indices.size)
            E_hist = self._resolve_sample_embeddings(
                X=X,
                sample_indices=hist_indices[start:stop],
                clf=clf,
                X_embed=X_embed,
            )
            cover = np.maximum(
                cover,
                self._max_instance_similarity_to_history(
                    E_cand,
                    E_hist,
                    gamma=gamma,
                ),
            )
        return cover

    def _cap_history_indices(self, hist_indices: np.ndarray) -> np.ndarray:
        if self.history_max_samples is None:
            return hist_indices
        if hist_indices.size <= self.history_max_samples:
            return hist_indices
        chosen = self.random_state.choice(
            hist_indices.size,
            size=self.history_max_samples,
            replace=False,
        )
        return np.sort(hist_indices[chosen])

    def _max_instance_similarity_to_history(
        self,
        E_cand,
        E_hist,
        *,
        gamma,
    ) -> np.ndarray:
        if E_hist.shape[0] == 0:
            return np.zeros(E_cand.shape[0], dtype=float)
        K = self._instance_kernel_between(E_cand, E_hist, gamma=gamma)
        return K.max(axis=1)

    def _compute_instance_kernel(self, *, X, sample_indices, clf, X_embed, gamma):
        E = self._resolve_sample_embeddings(
            X=X,
            sample_indices=sample_indices,
            clf=clf,
            X_embed=X_embed,
        )
        return self._instance_kernel_between(E, E, gamma=gamma)

    def _instance_kernel_between(
        self,
        A: np.ndarray,
        B: np.ndarray,
        *,
        gamma,
    ) -> np.ndarray:
        if self.normalize_embeddings:
            A_cos = _l2_normalize(A)
            B_cos = _l2_normalize(B)
        else:
            A_cos = np.asarray(A, dtype=float)
            B_cos = np.asarray(B, dtype=float)

        if self.instance_kernel == "cosine":
            K = A_cos @ B_cos.T
            return np.clip((K + 1.0) * 0.5, 0.0, 1.0)

        d2 = self._squared_distances(A, B)
        return np.exp(-float(gamma) * d2)

    def _resolve_sample_embeddings(self, *, X, sample_indices, clf, X_embed):
        if X_embed is not None:
            E = np.asarray(X_embed, dtype=float)
            if E.shape[0] == len(sample_indices):
                return self._ensure_2d(E, name="X_embed")
            if E.shape[0] > int(sample_indices.max(initial=-1)):
                return self._ensure_2d(E[sample_indices], name="X_embed")
            raise ValueError(
                "X_embed must be in local sample order or global sample order."
            )

        if clf is not None and X is not None:
            try:
                out = clf.predict_proba(
                    np.asarray(X)[sample_indices],
                    extra_outputs=["embeddings"],
                )
                if isinstance(out, (tuple, list)) and len(out) >= 2:
                    return self._ensure_2d(
                        np.asarray(out[1], dtype=float), name="embeddings"
                    )
            except Exception:
                pass

        if X is None:
            raise ValueError(
                "KernelHerdingPairAssigner requires `X`, `X_embed`, or "
                "classifier embeddings."
            )
        return self._ensure_2d(np.asarray(X)[sample_indices], name="X")

    @staticmethod
    def _ensure_2d(X, *, name: str) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[:, None]
        if X.ndim != 2:
            raise ValueError(f"{name} must resolve to a 2D numeric array.")
        return X

    def _compute_annotator_kernel(self, *, annotator_indices, annotator_embed):
        if self.annotator_kernel != "dirac":
            raise ValueError("Only annotator_kernel='dirac' is supported in v1.")
        return np.eye(len(annotator_indices), dtype=float)

    def _resolve_call_rbf_gamma(
        self,
        *,
        X,
        sample_indices,
        clf,
        X_embed,
        y,
        missing_label,
    ) -> float:
        if self.instance_kernel != "rbf":
            return 1.0
        if self.rbf_gamma is not None and str(self.rbf_gamma).lower() != "median":
            return float(self.rbf_gamma)

        indices = [np.asarray(sample_indices, dtype=int)]
        if self.rbf_gamma_scope in {"labeled", "current_and_labeled"} and y is not None:
            hist = self._historical_sample_indices(
                y=y,
                sample_indices=sample_indices,
                missing_label=missing_label,
            )
            if hist.size > 0:
                indices.append(hist)
        if self.rbf_gamma_scope == "labeled" and len(indices) > 1:
            gamma_indices = indices[1]
        elif self.rbf_gamma_scope == "labeled":
            gamma_indices = indices[0]
        else:
            gamma_indices = np.unique(np.concatenate(indices))

        gamma_indices = self._cap_gamma_indices(gamma_indices)
        E = self._resolve_sample_embeddings(
            X=X,
            sample_indices=gamma_indices,
            clf=clf,
            X_embed=X_embed,
        )
        return self._resolve_rbf_gamma(E)

    def _historical_sample_indices(self, *, y, sample_indices, missing_label):
        y = np.asarray(y)
        if y.ndim != 2:
            raise ValueError("`y` must have shape (n_samples, n_annotators).")
        hist_mask = is_labeled(y=y, missing_label=missing_label).any(axis=1)
        if hist_mask.shape[0] <= int(np.asarray(sample_indices).max(initial=-1)):
            raise ValueError("`y` must be in global sample order.")
        hist_mask[np.asarray(sample_indices, dtype=int)] = False
        return np.flatnonzero(hist_mask)

    def _cap_gamma_indices(self, indices: np.ndarray) -> np.ndarray:
        indices = np.asarray(indices, dtype=int)
        if self.gamma_max_samples is None:
            return indices
        if indices.size <= self.gamma_max_samples:
            return indices
        chosen = self.random_state.choice(
            indices.size,
            size=self.gamma_max_samples,
            replace=False,
        )
        return np.sort(indices[chosen])

    def _resolve_rbf_gamma(self, E: np.ndarray) -> float:
        if self.rbf_gamma is None or str(self.rbf_gamma).lower() == "median":
            d2 = self._squared_distances(E, E)
            vals = d2[np.triu_indices_from(d2, k=1)]
            vals = vals[vals > 1e-12]
            if vals.size == 0:
                return 1.0
            return 1.0 / float(np.median(vals))
        return float(self.rbf_gamma)

    @staticmethod
    def _squared_distances(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        A = np.asarray(A, dtype=float)
        B = np.asarray(B, dtype=float)
        d2 = (
            np.sum(A * A, axis=1)[:, None]
            + np.sum(B * B, axis=1)[None, :]
            - 2.0 * (A @ B.T)
        )
        return np.maximum(d2, 0.0)
