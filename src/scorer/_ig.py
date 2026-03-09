import numpy as np

from ._base import PairScorer
from ._utils import information_gain


class InformationGainPairScorer(PairScorer):
    """
    Expected information gain for candidate ``(sample, annotator)`` pairs.

    The utility of a pair is the mutual information between the latent class
    of a sample and the annotator label that would be observed for that sample.
    The annotation channel is obtained from ``clf.predict_proba`` using one of
    the following explicit channel variants:

    - ``"full_confusion"``: confusion matrices returned via
      ``extra_outputs=["annotator_confusion_matrices"]``, either with shape
      ``(n_annotators, K, K)`` or ``(n_samples, n_annotators, K, K)``.
    - ``"channel"``: pair-specific correctness probabilities and pair-specific
      annotator label probabilities returned via
      ``extra_outputs=["annotator_perf", "annotator_class"]``.
    - ``"scalar_uniform_confusion"``: pair-specific correctness probabilities
      returned via ``extra_outputs=["annotator_perf"]`` only,
      using a uniform off-diagonal confusion model.
    The class prior used in the IG computation is controlled by
    ``class_prior``:

    - ``"classifier"``: use ``clf.predict_proba(X)``.
    - ``"uniform"``: ignore classifier class probabilities and use a uniform
      prior over classes.
    """

    def __init__(
        self,
        *,
        channel_variant: str = "channel",
        class_prior: str = "classifier",
        eps: float = 1e-12,
        log_base: float = 2.0,
        normalize: bool = True,
        batch_size: int | None = None,
    ):
        self.channel_variant = str(channel_variant)
        self.class_prior = str(class_prior)
        self.eps = float(eps)
        self.log_base = float(log_base)
        self.normalize = bool(normalize)
        self.batch_size = batch_size

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
        variant = self.channel_variant.lower()
        if variant not in {
            "full_confusion",
            "channel",
            "scalar_uniform_confusion",
        }:
            raise ValueError(
                "channel_variant must be one of "
                "{'full_confusion', 'channel', "
                "'scalar_uniform_confusion'}."
            )
        if self.class_prior not in {"classifier", "uniform"}:
            raise ValueError(
                "class_prior must be one of {'classifier', 'uniform'}."
            )

        n_sel_s = len(sample_indices)
        n_sel_a = len(annotator_indices)
        X_cand = X[sample_indices]
        n_annotators_total = y.shape[1]

        if clf is None:
            raise ValueError("`clf` must be provided.")

        P_clf, extra_map = self._predict_channel_outputs(
            clf=clf,
            X_cand=X_cand,
            variant=variant,
        )
        P_prior = self._resolve_class_prior(P_clf)

        if variant == "full_confusion":
            C = np.asarray(
                extra_map["annotator_confusion_matrices"], dtype=float
            )
            U = self._information_gain_from_confusions(
                P=P_prior,
                C=C,
                annotator_indices=annotator_indices,
                n_annotators_total=n_annotators_total,
                n_sel_a=n_sel_a,
            )
        elif variant == "channel":
            annotator_perf = self._take_selected_annotators(
                np.asarray(extra_map["annotator_perf"], dtype=float),
                axis=1,
                annotator_indices=annotator_indices,
                n_annotators_total=n_annotators_total,
                name="annotator_perf",
            )
            annotator_class = self._take_selected_annotators(
                np.asarray(extra_map["annotator_class"], dtype=float),
                axis=1,
                annotator_indices=annotator_indices,
                n_annotators_total=n_annotators_total,
                name="annotator_class",
            )
            U = information_gain(
                P_prior,
                P_perf=annotator_perf,
                P_annot=annotator_class,
                eps=self.eps,
                log_base=self.log_base,
                normalize=self.normalize,
                batch_size=self.batch_size,
            )
        else:
            annotator_perf = self._take_selected_annotators(
                np.asarray(extra_map["annotator_perf"], dtype=float),
                axis=1,
                annotator_indices=annotator_indices,
                n_annotators_total=n_annotators_total,
                name="annotator_perf",
            )
            U = self._information_gain_from_accuracy_only(
                P=P_prior,
                theta=annotator_perf,
            )

        if U.shape != (n_sel_s, n_sel_a):
            raise ValueError(
                f"Expected utilities of shape {(n_sel_s, n_sel_a)}, got {U.shape}."
            )
        if available_mask is not None:
            U = np.where(available_mask, U, np.nan)
        return U

    def _predict_channel_outputs(self, *, clf, X_cand, variant):
        requested_outputs = {
            "full_confusion": ["annotator_confusion_matrices"],
            "channel": ["annotator_perf", "annotator_class"],
            "scalar_uniform_confusion": ["annotator_perf"],
        }[variant]
        out = clf.predict_proba(X_cand, extra_outputs=requested_outputs)
        P = np.asarray(out[0], dtype=float)
        extra_map = {
            name: value for name, value in zip(requested_outputs, out[1:])
        }
        return P, extra_map

    def _resolve_class_prior(self, P: np.ndarray) -> np.ndarray:
        P = np.asarray(P, dtype=float)
        if P.ndim != 2:
            raise ValueError(
                f"Expected classifier probabilities of shape (n_samples, K), got {P.shape}."
            )
        if self.class_prior == "classifier":
            return P
        K = P.shape[1]
        if K < 2:
            raise ValueError("Information gain requires at least 2 classes.")
        return np.full_like(P, 1.0 / K, dtype=float)

    def _information_gain_from_confusions(
        self,
        *,
        P: np.ndarray,
        C: np.ndarray | None,
        annotator_indices,
        n_annotators_total: int,
        n_sel_a: int,
    ) -> np.ndarray:
        if C is None:
            raise ValueError(
                "channel_variant='full_confusion' requires "
                "`annotator_confusion_matrices`."
            )
        if C.ndim == 3:
            C_pair = self._take_selected_annotators(
                C,
                axis=0,
                annotator_indices=annotator_indices,
                n_annotators_total=n_annotators_total,
                name="annotator_confusion_matrices",
            )
            C_pair = np.broadcast_to(
                C_pair[None, :, :, :],
                (P.shape[0], n_sel_a) + C_pair.shape[1:],
            )
        elif C.ndim == 4:
            C_pair = self._take_selected_annotators(
                C,
                axis=1,
                annotator_indices=annotator_indices,
                n_annotators_total=n_annotators_total,
                name="annotator_confusion_matrices",
            )
        else:
            raise ValueError(
                "annotator_confusion_matrices must have shape "
                "(n_annotators, n_classes, n_classes) or "
                "(n_samples, n_annotators, n_classes, n_classes)."
            )
        r = np.broadcast_to(P[:, None, :], C_pair.shape[:-1])
        return information_gain(
            r,
            C=C_pair,
            eps=self.eps,
            log_base=self.log_base,
            normalize=self.normalize,
            batch_size=self.batch_size,
        )

    def _information_gain_from_accuracy_only(
        self,
        *,
        P: np.ndarray,
        theta: np.ndarray | None,
    ) -> np.ndarray:
        if theta is None:
            raise ValueError(
                "channel_variant='scalar_uniform_confusion' requires "
                "`annotator_perf`."
            )
        n_classes = P.shape[1]
        if n_classes < 2:
            raise ValueError("Information gain requires at least 2 classes.")

        theta = np.clip(np.asarray(theta, dtype=float), 0.0, 1.0)
        off = (1.0 - theta) / (n_classes - 1)
        C = np.repeat(off[..., None, None], n_classes, axis=-2)
        C = np.repeat(C, n_classes, axis=-1)
        diag_idx = np.arange(n_classes)
        C[..., diag_idx, diag_idx] = theta[..., None]
        r = np.broadcast_to(P[:, None, :], C.shape[:-1])
        return information_gain(
            r,
            C=C,
            eps=self.eps,
            log_base=self.log_base,
            normalize=self.normalize,
            batch_size=self.batch_size,
        )

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
