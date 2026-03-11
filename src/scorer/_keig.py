from __future__ import annotations

import numpy as np

from sklearn.metrics.pairwise import rbf_kernel
from sklearn.utils import check_random_state

from skactiveml.utils import is_labeled
from ._base import PairScorer
from ._utils import information_gain


def _l2_normalize(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.size == 0:
        return X.reshape((0,) + X.shape[1:])
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(n, eps)


class KernelEvidenceInformationGain(PairScorer):
    """
    Information gain with model means and kernel-based evidence.

    The candidate classifier output and annotator-model outputs define the
    posterior means for the latent class prior and the annotator channel.
    Kernel-weighted neighborhoods of labeled samples control the concentration
    of those posteriors:

    - the class prior uses all labeled samples near the candidate sample;
    - annotator-specific quantities use only samples labeled by that annotator.

    Supported channel variants are:

    - ``"channel"``:
      Beta posterior for annotator accuracy and Dirichlet posterior for the
      annotator label-distribution estimate returned by
      ``extra_outputs=["annotator_perf", "annotator_class"]``.
    - ``"scalar_uniform_confusion"``:
      Beta posterior for a scalar annotator accuracy and a symmetric confusion
      channel.
    - ``"full_confusion"``:
      Row-wise Dirichlet posterior for confusion matrices returned by
      ``extra_outputs=["annotator_confusion_matrices"]``.
    """

    def __init__(
        self,
        *,
        channel_variant: str = "channel",
        accuracy_mean: float = 0.95,
        accuracy_strength: float = 10.0,
        class_prior_strength: float = 1.0,
        channel_label_dirichlet_strength: float = 1.0,
        gamma_x="median",
        gamma_x_scope: str = "global",
        use_ess_class_prior: bool = False,
        tau_class_prior: float = 1.0,
        use_ess_beta: bool = False,
        tau_beta: float = 1.0,
        use_ess_label_dirichlet: bool = False,
        tau_label_dirichlet: float = 1.0,
        n_theta_samples: int = 1,
        sample_class_prior: bool = False,
        sample_label_dirichlet: bool = False,
        random_state=None,
        eps: float = 1e-12,
        log_base: float = 2.0,
        normalize: bool = True,
        batch_size: int | None = None,
    ):
        self.channel_variant = str(channel_variant)
        self.accuracy_mean = float(accuracy_mean)
        self.accuracy_strength = float(accuracy_strength)
        self.class_prior_strength = float(class_prior_strength)
        self.channel_label_dirichlet_strength = float(
            channel_label_dirichlet_strength
        )
        self.gamma_x = gamma_x
        self.gamma_x_scope = str(gamma_x_scope)
        self.use_ess_class_prior = bool(use_ess_class_prior)
        self.tau_class_prior = float(tau_class_prior)
        self.use_ess_beta = bool(use_ess_beta)
        self.tau_beta = float(tau_beta)
        self.use_ess_label_dirichlet = bool(use_ess_label_dirichlet)
        self.tau_label_dirichlet = float(tau_label_dirichlet)
        self.n_theta_samples = int(n_theta_samples)
        self.sample_class_prior = bool(sample_class_prior)
        self.sample_label_dirichlet = bool(sample_label_dirichlet)
        self.random_state = check_random_state(random_state)
        self.eps = float(eps)
        self.log_base = float(log_base)
        self.normalize = bool(normalize)
        self.batch_size = batch_size

        if self.channel_variant not in {
            "channel",
            "scalar_uniform_confusion",
            "full_confusion",
        }:
            raise ValueError(
                "channel_variant must be one of "
                "{'channel', 'scalar_uniform_confusion', 'full_confusion'}"
            )
        if not (0.0 < self.accuracy_mean < 1.0):
            raise ValueError("accuracy_mean must be in (0, 1)")
        if self.accuracy_strength <= 0:
            raise ValueError("accuracy_strength must be > 0")
        if self.class_prior_strength <= 0:
            raise ValueError("class_prior_strength must be > 0")
        if self.channel_label_dirichlet_strength <= 0:
            raise ValueError("channel_label_dirichlet_strength must be > 0")
        if self.gamma_x_scope not in {"global", "per_annotator"}:
            raise ValueError(
                "gamma_x_scope must be one of {'global', 'per_annotator'}"
            )
        if self.tau_class_prior <= 0:
            raise ValueError("tau_class_prior must be > 0")
        if self.tau_beta <= 0:
            raise ValueError("tau_beta must be > 0")
        if self.tau_label_dirichlet <= 0:
            raise ValueError("tau_label_dirichlet must be > 0")

    def _compute(
        self,
        X,
        y,
        sample_indices,
        annotator_indices,
        available_mask,
        clf,
        **kwargs,
    ):
        rng = kwargs.get("rng", None)
        if rng is None:
            rng = np.random.default_rng(
                self.random_state.randint(0, 2**32 - 1)
            )

        K = len(clf.classes_)
        if K < 2:
            raise ValueError("Information gain requires at least 2 classes.")

        n_sel_s = len(sample_indices)
        n_sel_a = len(annotator_indices)
        n_annotators_total = y.shape[1]
        X_cand = X[sample_indices]

        cand_extra_outputs = ["embeddings"]
        if self.channel_variant == "channel":
            cand_extra_outputs += ["annotator_perf", "annotator_class"]
        elif self.channel_variant == "scalar_uniform_confusion":
            cand_extra_outputs += ["annotator_perf"]
        else:
            cand_extra_outputs += ["annotator_confusion_matrices"]

        cand_out = clf.predict_proba(X_cand, extra_outputs=cand_extra_outputs)
        if not isinstance(cand_out, (tuple, list)):
            raise ValueError(
                "clf.predict_proba must return a tuple when extra_outputs are requested."
            )

        r_mean = np.asarray(cand_out[0], dtype=float)
        r_mean = np.clip(r_mean, self.eps, 1.0)
        r_mean = r_mean / np.maximum(r_mean.sum(axis=1, keepdims=True), self.eps)
        cand_named = {
            name: value for name, value in zip(cand_extra_outputs, cand_out[1:])
        }
        X_cand_emb = _l2_normalize(np.asarray(cand_named["embeddings"], dtype=float))

        is_lbld = is_labeled(y=y, missing_label=clf.missing_label)
        obs_sample_mask = np.any(is_lbld, axis=1)
        obs_samples = np.flatnonzero(obs_sample_mask)

        if obs_samples.size > 0:
            obs_out = clf.predict_proba(X[obs_samples], extra_outputs=["embeddings"])
            X_obs_emb = _l2_normalize(np.asarray(obs_out[1], dtype=float))
        else:
            X_obs_emb = np.empty((0, X_cand_emb.shape[1]), dtype=float)

        gamma_x_global = self._resolve_gamma_from_embeddings(
            X_obs_emb, self.gamma_x
        )

        class_K = self._kernel_matrix(
            X_obs_emb,
            X_cand_emb,
            gamma=gamma_x_global,
        )
        class_conc = self._kernel_concentration(
            class_K,
            use_ess=self.use_ess_class_prior,
            tau=self.tau_class_prior,
        )
        alpha_class = self._dirichlet_from_mean_and_concentration(
            mean=r_mean,
            concentration=class_conc,
            prior_strength=self.class_prior_strength,
        )

        if self.channel_variant == "channel":
            perf_mean = self._take_selected_annotators(
                np.asarray(cand_named["annotator_perf"], dtype=float),
                axis=1,
                annotator_indices=annotator_indices,
                n_annotators_total=n_annotators_total,
                name="annotator_perf",
            )
            annot_mean = self._take_selected_annotators(
                np.asarray(cand_named["annotator_class"], dtype=float),
                axis=1,
                annotator_indices=annotator_indices,
                n_annotators_total=n_annotators_total,
                name="annotator_class",
            )
            annot_mean = np.clip(annot_mean, self.eps, 1.0)
            annot_mean = annot_mean / np.maximum(
                annot_mean.sum(axis=2, keepdims=True), self.eps
            )
        elif self.channel_variant == "scalar_uniform_confusion":
            perf_mean = self._take_selected_annotators(
                np.asarray(cand_named["annotator_perf"], dtype=float),
                axis=1,
                annotator_indices=annotator_indices,
                n_annotators_total=n_annotators_total,
                name="annotator_perf",
            )
            annot_mean = None
        else:
            conf_mean = np.asarray(
                cand_named["annotator_confusion_matrices"], dtype=float
            )
            if conf_mean.ndim == 3:
                conf_mean = self._take_selected_annotators(
                    conf_mean,
                    axis=0,
                    annotator_indices=annotator_indices,
                    n_annotators_total=n_annotators_total,
                    name="annotator_confusion_matrices",
                )
            elif conf_mean.ndim == 4:
                conf_mean = self._take_selected_annotators(
                    conf_mean,
                    axis=1,
                    annotator_indices=annotator_indices,
                    n_annotators_total=n_annotators_total,
                    name="annotator_confusion_matrices",
                )
            else:
                raise ValueError(
                    "annotator_confusion_matrices must have shape "
                    "(n_annotators, K, K) or (n_samples, n_annotators, K, K)."
                )
            perf_mean = annot_mean = None

        obs_row_by_sample = np.full(y.shape[0], -1, dtype=int)
        obs_row_by_sample[obs_samples] = np.arange(obs_samples.size)

        U = np.empty((n_sel_s, n_sel_a), dtype=float)
        for j_a, a in enumerate(annotator_indices):
            obs_samples_a = np.flatnonzero(is_lbld[:, a])
            obs_rows_a = obs_row_by_sample[obs_samples_a]
            X_obs_a_emb = X_obs_emb[obs_rows_a] if obs_rows_a.size > 0 else X_obs_emb[:0]

            if self.gamma_x_scope == "per_annotator":
                gamma_x_a = self._resolve_gamma_from_embeddings(
                    X_obs_a_emb, self.gamma_x
                )
            else:
                gamma_x_a = gamma_x_global

            annot_K = self._kernel_matrix(
                X_obs_a_emb,
                X_cand_emb,
                gamma=gamma_x_a,
            )
            beta_conc = self._kernel_concentration(
                annot_K,
                use_ess=self.use_ess_beta,
                tau=self.tau_beta,
            )
            dir_conc = self._kernel_concentration(
                annot_K,
                use_ess=self.use_ess_label_dirichlet,
                tau=self.tau_label_dirichlet,
            )

            if self.channel_variant == "channel":
                alpha_acc, beta_acc = self._beta_from_mean_and_concentration(
                    mean=perf_mean[:, j_a],
                    concentration=beta_conc,
                )
                gamma_annot = self._dirichlet_from_mean_and_concentration(
                    mean=annot_mean[:, j_a, :],
                    concentration=dir_conc,
                    prior_strength=self.channel_label_dirichlet_strength,
                )
                U_col = self._ig_channel_batch(
                    alpha_class=alpha_class,
                    alpha_acc=alpha_acc,
                    beta_acc=beta_acc,
                    gamma_annot=gamma_annot,
                    rng=rng,
                )
            elif self.channel_variant == "scalar_uniform_confusion":
                alpha_acc, beta_acc = self._beta_from_mean_and_concentration(
                    mean=perf_mean[:, j_a],
                    concentration=beta_conc,
                )
                U_col = self._ig_scalar_uniform_confusion_batch(
                    alpha_class=alpha_class,
                    alpha_acc=alpha_acc,
                    beta_acc=beta_acc,
                    rng=rng,
                )
            else:
                C_mean = self._select_confusion_for_annotator(
                    C=conf_mean,
                    annotator_pos=j_a,
                    n_samples=n_sel_s,
                )
                delta = self._full_confusion_posterior(
                    C_mean=C_mean,
                    concentration=dir_conc,
                )
                U_col = self._ig_full_confusion_batch(
                    alpha_class=alpha_class,
                    delta=delta,
                    rng=rng,
                )

            if available_mask is not None:
                U_col = np.where(available_mask[:, j_a], U_col, np.nan)
            U[:, j_a] = U_col

        return U

    def _ig_channel_batch(
        self,
        *,
        alpha_class: np.ndarray,
        alpha_acc: np.ndarray,
        beta_acc: np.ndarray,
        gamma_annot: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        S, K = alpha_class.shape
        T = self._num_draws()

        r_draws = self._class_prior_draws(alpha_class=alpha_class, T=T, rng=rng)
        theta_draws = self._theta_draws(
            alpha=alpha_acc,
            beta=beta_acc,
            T=T,
            rng=rng,
        )
        g_draws = self._dirichlet_draws(
            alpha=gamma_annot,
            T=T,
            rng=rng,
            sample=self.sample_label_dirichlet,
        )

        ig_draws = information_gain(
            r_draws.reshape(-1, K),
            P_perf=theta_draws.reshape(-1, 1),
            P_annot=g_draws.reshape(-1, 1, K),
            eps=self.eps,
            log_base=self.log_base,
            normalize=self.normalize,
            batch_size=self.batch_size,
        ).reshape(S, T)
        return ig_draws.mean(axis=1)

    def _ig_scalar_uniform_confusion_batch(
        self,
        *,
        alpha_class: np.ndarray,
        alpha_acc: np.ndarray,
        beta_acc: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        S, K = alpha_class.shape
        T = self._num_draws()

        r_draws = self._class_prior_draws(alpha_class=alpha_class, T=T, rng=rng)
        theta_draws = self._theta_draws(
            alpha=alpha_acc,
            beta=beta_acc,
            T=T,
            rng=rng,
        )

        eye = np.eye(K, dtype=float)[None, None, :, :]
        off_base = (
            (np.ones((K, K), dtype=float) - np.eye(K, dtype=float)) / (K - 1)
        )[None, None, :, :]
        C = (1.0 - theta_draws)[..., None, None] * off_base + theta_draws[
            ..., None, None
        ] * eye

        ig_draws = information_gain(
            r_draws,
            C=C,
            eps=self.eps,
            log_base=self.log_base,
            normalize=self.normalize,
            batch_size=self.batch_size,
        )
        return ig_draws.mean(axis=1)

    def _ig_full_confusion_batch(
        self,
        *,
        alpha_class: np.ndarray,
        delta: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        S, K = alpha_class.shape
        T = self._num_draws()

        r_draws = self._class_prior_draws(alpha_class=alpha_class, T=T, rng=rng)
        if self.sample_label_dirichlet:
            alpha_bt = np.clip(delta[:, None, :, :], self.eps, None)
            if T != 1:
                alpha_bt = np.repeat(alpha_bt, T, axis=1)
            X = rng.gamma(shape=alpha_bt, scale=1.0)
            C_draws = X / np.maximum(X.sum(axis=3, keepdims=True), self.eps)
        else:
            C_mean = delta / np.maximum(delta.sum(axis=2, keepdims=True), self.eps)
            C_draws = np.repeat(C_mean[:, None, :, :], T, axis=1)

        ig_draws = information_gain(
            r_draws,
            C=C_draws,
            eps=self.eps,
            log_base=self.log_base,
            normalize=self.normalize,
            batch_size=self.batch_size,
        )
        return ig_draws.mean(axis=1)

    def _class_prior_draws(
        self,
        *,
        alpha_class: np.ndarray,
        T: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        alpha_class = np.asarray(alpha_class, dtype=float)
        mean = alpha_class / np.maximum(
            alpha_class.sum(axis=1, keepdims=True), self.eps
        )
        if not self.sample_class_prior:
            return np.repeat(mean[:, None, :], T, axis=1)

        alpha_bt = np.clip(alpha_class[:, None, :], self.eps, None)
        if T != 1:
            alpha_bt = np.repeat(alpha_bt, T, axis=1)
        X = rng.gamma(shape=alpha_bt, scale=1.0)
        return X / np.maximum(X.sum(axis=2, keepdims=True), self.eps)

    def _theta_draws(
        self,
        *,
        alpha: np.ndarray,
        beta: np.ndarray,
        T: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        alpha = np.asarray(alpha, dtype=float)
        beta = np.asarray(beta, dtype=float)
        if self.n_theta_samples <= 0:
            return (alpha / np.maximum(alpha + beta, self.eps))[:, None]
        return rng.beta(
            alpha[:, None],
            beta[:, None],
            size=(alpha.shape[0], T),
        ).astype(float)

    def _dirichlet_draws(
        self,
        *,
        alpha: np.ndarray,
        T: int,
        rng: np.random.Generator,
        sample: bool,
    ) -> np.ndarray:
        alpha = np.asarray(alpha, dtype=float)
        mean = alpha / np.maximum(alpha.sum(axis=1, keepdims=True), self.eps)
        if not sample:
            return np.repeat(mean[:, None, :], T, axis=1)

        alpha_bt = np.clip(alpha[:, None, :], self.eps, None)
        if T != 1:
            alpha_bt = np.repeat(alpha_bt, T, axis=1)
        X = rng.gamma(shape=alpha_bt, scale=1.0)
        return X / np.maximum(X.sum(axis=2, keepdims=True), self.eps)

    def _full_confusion_posterior(
        self,
        *,
        C_mean: np.ndarray,
        concentration: np.ndarray,
    ) -> np.ndarray:
        C_mean = np.asarray(C_mean, dtype=float)
        C_mean = np.clip(C_mean, self.eps, 1.0)
        C_mean = C_mean / np.maximum(C_mean.sum(axis=2, keepdims=True), self.eps)
        K = C_mean.shape[1]

        delta0 = self._full_confusion_dirichlet_prior(
            K=K,
            accuracy_mean=self.accuracy_mean,
            row_strength=self.accuracy_strength,
        )
        return delta0[None, :, :] + concentration[:, None, None] * C_mean

    def _beta_from_mean_and_concentration(
        self,
        *,
        mean: np.ndarray,
        concentration: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        mean = np.clip(np.asarray(mean, dtype=float), self.eps, 1.0 - self.eps)
        concentration = np.asarray(concentration, dtype=float)
        alpha0 = self.accuracy_mean * self.accuracy_strength
        beta0 = (1.0 - self.accuracy_mean) * self.accuracy_strength
        alpha = alpha0 + concentration * mean
        beta = beta0 + concentration * (1.0 - mean)
        return alpha, beta

    def _dirichlet_from_mean_and_concentration(
        self,
        *,
        mean: np.ndarray,
        concentration: np.ndarray,
        prior_strength: float,
    ) -> np.ndarray:
        mean = np.asarray(mean, dtype=float)
        mean = np.clip(mean, self.eps, 1.0)
        mean = mean / np.maximum(mean.sum(axis=1, keepdims=True), self.eps)
        concentration = np.asarray(concentration, dtype=float)
        K = mean.shape[1]
        gamma0 = np.full(K, prior_strength / K, dtype=float)
        return gamma0[None, :] + concentration[:, None] * mean

    def _num_draws(self) -> int:
        return 1 if self.n_theta_samples <= 0 else self.n_theta_samples

    @staticmethod
    def _kernel_concentration(
        K: np.ndarray,
        *,
        use_ess: bool,
        tau: float,
        eps: float = 1e-12,
    ) -> np.ndarray:
        K = np.asarray(K, dtype=float)
        if K.ndim != 2:
            raise ValueError(f"K must be 2D, got shape {K.shape}")
        if K.shape[0] == 0:
            return np.zeros(K.shape[1], dtype=float)
        mass = K.sum(axis=0)
        if not use_ess:
            return mass
        m2 = (K**2).sum(axis=0)
        n_eff = (mass**2) / np.maximum(m2, eps)
        return tau * n_eff

    @staticmethod
    def _kernel_matrix(
        X_obs: np.ndarray,
        X_cand: np.ndarray,
        *,
        gamma: float,
    ) -> np.ndarray:
        X_obs = np.asarray(X_obs, dtype=float)
        X_cand = np.asarray(X_cand, dtype=float)
        if X_obs.shape[0] == 0:
            return np.empty((0, X_cand.shape[0]), dtype=float)
        return rbf_kernel(X_obs, X_cand, gamma=gamma)

    @staticmethod
    def _resolve_gamma_from_embeddings(E: np.ndarray, mode):
        E = np.asarray(E, dtype=float)
        if E.shape[0] < 2:
            return 1.0

        norms = (E * E).sum(axis=1, keepdims=True)
        d2 = norms + norms.T - 2.0 * (E @ E.T)
        np.fill_diagonal(d2, np.nan)
        d = np.sqrt(np.maximum(d2, 0.0))

        if mode == "median":
            s = np.nanmedian(d)
            s = max(float(s), 1e-3)
            return s ** (-2)
        if mode == "mean":
            s = np.nanmean(d)
            s = max(float(s), 1e-3)
            return s ** (-2)
        if mode == "minimum":
            s = np.nanmin(d)
            s = max(float(s), 1e-3)
            return s ** (-2)
        return float(mode)

    @staticmethod
    def _full_confusion_dirichlet_prior(
        *,
        K: int,
        accuracy_mean: float,
        row_strength: float,
    ) -> np.ndarray:
        if K < 2:
            raise ValueError("K must be >= 2")
        off = (1.0 - accuracy_mean) / (K - 1)
        prior_mean = np.full((K, K), off, dtype=float)
        np.fill_diagonal(prior_mean, accuracy_mean)
        return row_strength * prior_mean

    @staticmethod
    def _take_selected_annotators(
        arr,
        *,
        axis,
        annotator_indices,
        n_annotators_total,
        name,
    ):
        if arr.shape[axis] == len(annotator_indices):
            return arr
        if arr.shape[axis] == n_annotators_total:
            return np.take(arr, indices=annotator_indices, axis=axis)
        raise ValueError(
            f"{name} has incompatible annotator axis size {arr.shape[axis]}; "
            f"expected {len(annotator_indices)} or {n_annotators_total}."
        )

    @staticmethod
    def _select_confusion_for_annotator(
        *,
        C: np.ndarray,
        annotator_pos: int,
        n_samples: int,
    ) -> np.ndarray:
        if C.ndim == 3:
            return np.broadcast_to(C[annotator_pos][None, :, :], (n_samples,) + C.shape[1:])
        if C.ndim == 4:
            return C[:, annotator_pos, :, :]
        raise ValueError(f"Unexpected confusion array shape {C.shape}.")
