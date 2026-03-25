try:
    import numpy as np

    from sklearn.utils.validation import check_array
    from torch import nn

    from skactiveml.utils import (
        MISSING_LABEL,
        check_n_features,
        compute_vote_vectors,
        majority_vote,
    )
    from skactiveml.classifier.multiannotator._utils import (
        _SkorchMultiAnnotatorClassifier,
        _MultiAnnotatorClassificationModule,
    )


    class AggregateClassifier(_SkorchMultiAnnotatorClassifier):
        """Aggregate Classifier

        Aggregate Classifier is a classifier aggregating the noisy class
        labels from multiple annotators in a first stage as the basis
        for training a usual classifier in the second stage via cross-entropy.


        Parameters
        ----------
        clf_module : nn.Module or nn.Module.__class__
            A PyTorch module as classification model outputting logits for
            samples as input. In general, the uninstantiated class should
            be passed, although instantiated modules will also work. The
            `forward` module must return logits as first element and optional
            sample embeddings as second element. If no sample embeddings are
            returned, the implementation uses the original samples.
        aggregate_function : {"majority_voting", "soft_voting", "dawid_skene_voting"}, \
                default="majority_voting"
            Aggregation strategy used during fit:
            - "majority_voting": majority vote converted to one-hot labels.
            - "soft_voting": normalized vote vectors.
            - "dawid_skene_voting": Dawid-Skene posterior label estimates.
        dawid_skene_max_iter : int, default=100
            Maximum number of EM iterations for Dawid-Skene.
        dawid_skene_tol : float, default=1e-6
            Convergence tolerance for Dawid-Skene.
        dawid_skene_smoothing : float, default=1e-1
            Additive smoothing for Dawid-Skene confusion-matrix estimation.
        n_annotators : int, default=None
            Number of annotators. If `n_annotators=None`, the number of
            annotators is inferred from `y` when calling `fit`.
        neural_net_param_dict : dict, default=None
            Additional arguments for `skorch.net.NeuralNet`. If
            `neural_net_param_dict` is None, no additional arguments are added.
            `module`, `criterion`, `predict_nonlinearity`, and `train_split`
            are not allowed in this dictionary.
        sample_dtype : str or type, default=np.float32
            Dtype to which input samples are cast inside the estimator. If set
            to `None`, the input dtype is preserved.
        classes : array-like of shape (n_classes,), default=None
            Holds the label for each class. If `None`, the classes are
            determined during the fit.
        missing_label : scalar or string or np.nan or None, default=np.nan
            Value to represent a missing label.
        cost_matrix : array-like of shape (n_classes, n_classes), default=None
            Cost matrix with `cost_matrix[i,j]` indicating cost of predicting
            class `classes[j]` for a sample of class `classes[i]`. Can be only
            set, if `classes` is not `None`.
        random_state : int or RandomState sample or None, default=None
            Determines random number for `predict` method. Pass an int for
            reproducible results across multiple method calls.
        """

        _ALLOWED_EXTRA_OUTPUTS = {
            "logits",
            "embeddings",
            "annotator_perf",
            "annotator_class",
            "annotator_confusion_matrices",
        }
        _DS_EXTRA_OUTPUTS = {
            "annotator_perf",
            "annotator_class",
            "annotator_confusion_matrices",
        }

        def __init__(
            self,
            clf_module,
            aggregate_function="majority_voting",
            dawid_skene_max_iter=100,
            dawid_skene_tol=1e-6,
            dawid_skene_smoothing=1e-2,
            n_annotators=None,
            neural_net_param_dict=None,
            sample_dtype=np.float32,
            classes=None,
            cost_matrix=None,
            missing_label=MISSING_LABEL,
            random_state=None,
        ):
            super(AggregateClassifier, self).__init__(
                multi_annotator_module=_AggregateModule,
                clf_module=clf_module,
                n_annotators=n_annotators,
                criterion=nn.CrossEntropyLoss,
                sample_dtype=sample_dtype,
                classes=classes,
                missing_label=missing_label,
                cost_matrix=cost_matrix,
                random_state=random_state,
                neural_net_param_dict=neural_net_param_dict,
            )
            self.aggregate_function = aggregate_function
            self.dawid_skene_max_iter = dawid_skene_max_iter
            self.dawid_skene_tol = dawid_skene_tol
            self.dawid_skene_smoothing = dawid_skene_smoothing

        def _fit(self, fit_function, X, y, **fit_params):
            """Fit the classifier on a multi-annotator label matrix."""
            self._check_multiannotator_y(y)
            return super()._fit(fit_function, X, y, **fit_params)

        def _return_training_data(self, X, y):
            X_train, y_train = super()._return_training_data(X=X, y=y)
            if y_train is None:
                return X_train, y_train
            return X_train, self._aggregate_targets(y=y_train)

        @staticmethod
        def _check_multiannotator_y(y):
            if y is None:
                raise ValueError("`y` must not be None.")
            if np.asarray(y).ndim != 2:
                raise ValueError(
                    "`y` must have shape (n_samples, n_annotators) for "
                    "multi-annotator training."
                )

        def _aggregate_targets(self, y):
            if self.aggregate_function == "majority_voting":
                y_mv = majority_vote(
                    y=y,
                    missing_label=-1,
                    random_state=self.random_state,
                    classes=np.arange(len(self.classes_)),
                )
                return y_mv

            if self.aggregate_function == "soft_voting":
                votes = compute_vote_vectors(
                    y=y,
                    classes=np.arange(len(self.classes_)),
                    missing_label=-1,
                )
                return _normalize_rows(votes).astype(np.float32, copy=False)

            if self.aggregate_function == "dawid_skene_voting":
                posteriors, confusions, class_prior, info = (
                    dawid_skene(y=y, n_classes=len(self.classes_), max_iter=self.dawid_skene_max_iter, tol=self.dawid_skene_tol, smoothing=self.dawid_skene_smoothing)
                )
                self.dawid_skene_posteriors_ = posteriors
                self.annotator_confusion_matrices_ = confusions
                self.dawid_skene_class_prior_ = class_prior
                self.dawid_skene_n_iter_ = info["n_iter"]
                self.dawid_skene_converged_ = info["converged"]
                return posteriors.astype(np.float32, copy=False)

            raise ValueError("Unexpected aggregate function.")


        def _validate_ds_extra_outputs(self, extra_outputs):
            ds_requested = [
                name for name in extra_outputs if name in self._DS_EXTRA_OUTPUTS
            ]
            if not ds_requested:
                return
            if self.aggregate_function != "dawid_skene_voting":
                raise ValueError(
                    "Requested Dawid-Skene outputs "
                    f"{ds_requested}, but aggregate_function="
                    f"{self.aggregate_function!r} is not "
                    "'dawid_skene_voting'."
                )

        def predict_proba(
            self,
            X,
            extra_outputs=None,
        ):
            """Return class probability estimates for the test samples `X`.

            By default, this method returns only the class probabilities `P`.
            If `extra_outputs` is provided, a tuple is returned whose first
            element is `P` and whose remaining elements are the requested
            additional forward outputs, in the order specified by
            `extra_outputs`.

            Parameters
            ----------
            X : array-like of shape (n_samples, ...)
                Test samples.
            extra_outputs : None or str or sequence of str, default=None
                Names of additional outputs to return next to `P`. The names
                must be a subset of the following keys:

                - "logits" : Additionally return the class-membership logits
                  `L_class` for the samples in `X`.
                - "embeddings" : Additionally return the learned embeddings
                  `X_embed` for the samples in `X`.
                - "annotator_perf" : additionally return the estimated
                  annotator performance probabilities `P_perf` for each sample-
                  annotator pair (available only for `aggregate_function=
                  'dawid_skene'`).
                - "annotator_class" : Additionally return the annotator–class
                  probability estimates `P_annot` for each sample, class, and
                  annotator (available only for `aggregate_function=
                  'dawid_skene'`).
                - "annotator_confusion_matrices" : Additionally return the
                  Dawid-Skene annotator confusion matrices `C` of shape
                  `(n_annotators, n_classes, n_classes)`.

            Returns
            -------
            P : numpy.ndarray of shape (n_samples, n_classes)
                Class probabilities of the test samples. Classes are ordered
                according to `self.classes_`.
            *extras : numpy.ndarray, optional
                Only returned if `extra_outputs` is not `None`. In that
                case, the method returns a tuple whose first element is `P`
                and whose remaining elements correspond to the requested
                forward outputs in the order given by `extra_outputs`.
                Potential outputs are:

                - `L_class` : `np.ndarray` of shape `(n_samples, n_classes)`,
                  where `L_class[n, c]` is the logit for the class
                  `classes_[c]` of sample `X[n]`.
                - `X_embed` : `np.ndarray` of shape `(n_samples, ...)`, where
                  `X_embed[n]` refers to the learned embedding for sample
                  `X[n]`.
                - `P_perf` : `np.ndarray` of shape `(n_samples, n_annotators)`,
                  where `P_perf[n, m]` refers to the estimated label
                  correctness probability (performance) of annotator `m` when
                  labeling sample `X[n]`.
                - `P_annot` : `np.ndarray` of shape
                  `(n_samples, n_annotators, n_classes)`, where
                  `P_annot[n, m, c]` refers to the probability that annotator
                  `m` provides the class label `c` for sample `X[n]`.
            """
            # Check input parameters.
            self._validate_data_kwargs()
            X = check_array(X, **self.check_X_dict_)
            check_n_features(
                self, X, reset=not hasattr(self, "n_features_in_")
            )
            extra_outputs = self._normalize_extra_outputs(
                extra_outputs=extra_outputs,
                allowed_names=AggregateClassifier._ALLOWED_EXTRA_OUTPUTS,
            )
            self._validate_ds_extra_outputs(extra_outputs=extra_outputs)

            # Initialize module, if not done yet.
            if not hasattr(self, "neural_net_"):
                self.initialize()

            # Set forward options to obtain the different outputs required
            # by the input parameters.
            net = self.neural_net_.module_
            old_forward_return = net.forward_return
            forward_outputs = {"probas": (0, nn.Softmax(dim=-1))}
            forward_returns = ["logits_class"]
            model_extra_outputs = []
            out_idx = 1

            if "logits" in extra_outputs:
                forward_outputs["logits"] = (0, None)
                model_extra_outputs.append("logits")

            if "embeddings" in extra_outputs:
                forward_outputs["embeddings"] = (out_idx, None)
                forward_returns.append("x_embed")
                model_extra_outputs.append("embeddings")
                out_idx += 1

            # Compute predictions for the different outputs required
            # by the input parameters.
            try:
                net.set_forward_return(forward_returns)
                fw_out = self._forward_with_named_outputs(
                    X=X,
                    forward_outputs=forward_outputs,
                    extra_outputs=model_extra_outputs,
                )
            finally:
                net.set_forward_return(old_forward_return)

            if isinstance(fw_out, tuple):
                p_class = fw_out[0]
                model_named = {
                    name: value
                    for name, value in zip(model_extra_outputs, fw_out[1:])
                }
            else:
                p_class = fw_out
                model_named = {}

            ds_requested = [
                name
                for name in (
                    "annotator_perf",
                    "annotator_class",
                    "annotator_confusion_matrices",
                )
                if name in extra_outputs
            ]
            if ds_requested:
                confusions = getattr(self, "annotator_confusion_matrices_", None)
                if confusions is None:
                    raise RuntimeError(
                        "Dawid-Skene outputs requested, but no fitted "
                        "annotator confusion matrices are available."
                    )
                confusions = np.asarray(confusions, dtype=float)
                p_class_ds = None

                if "annotator_perf" in ds_requested:
                    p_class_ds = np.asarray(p_class, dtype=float)
                    conf_diag = np.diagonal(confusions, axis1=1, axis2=2)
                    p_perf = np.einsum(
                        "nk,mk->nm", p_class_ds, conf_diag, optimize=True
                    )
                    model_named["annotator_perf"] = p_perf.astype(
                        np.float32, copy=False
                    )

                if "annotator_class" in ds_requested:
                    if p_class_ds is None:
                        p_class_ds = np.asarray(p_class, dtype=float)
                    p_annot = np.einsum(
                        "nk,mkl->nml", p_class_ds, confusions, optimize=True
                    )
                    model_named["annotator_class"] = p_annot.astype(
                        np.float32, copy=False
                    )

                if "annotator_confusion_matrices" in ds_requested:
                    model_named["annotator_confusion_matrices"] = (
                        confusions.astype(np.float32, copy=False)
                    )

            # Initialize fallbacks if the classifier hasn't been fitted before.
            self._initialize_fallbacks(p_class)

            if not extra_outputs:
                return p_class
            ordered = [model_named[name] for name in extra_outputs]
            return (p_class, *ordered)

        def _build_neural_net_param_overrides(self, X, y):
            return {
                "criterion__reduction": "mean",
            }

    class _AggregateModule(_MultiAnnotatorClassificationModule):
        """Classification module wrapper used by :class:`AggregateClassifier`.

        Parameters
        ----------
        clf_module : nn.Module or nn.Module.__class__
            Classifier backbone/head that maps `x -> logits_class` or
            `(logits_class, x_embed)`. If it returns only logits, `x_embed` is
            set to the input `x` (or to `None` if `x` is not an embedding).
        clf_module_param_dict : dict
            Keyword args for constructing `clf_module` if a class is passed.

        """

        def __init__(
            self, clf_module, clf_module_param_dict
        ):
            super().__init__(
                clf_module=clf_module,
                clf_module_param_dict=clf_module_param_dict,
                default_forward_outputs="logits_class",
                full_forward_outputs=[
                    "logits_class",
                    "x_embed",
                ],
            )

        def forward(self, x):
            """
            Forward pass through the wrapped classification module.

            Parameters
            ----------
            x : torch.Tensor of shape (batch_size, ...)
                Input samples.

            Returns
            -------
            logits_class : torch.Tensor of shape (batch_size, n_classes)
                Class-membership logits.
            x_embed : torch.Tensor of shape (batch_size, ...), optional
                Learned embeddings of samples. Only returned if "x_embed" in
                `self.forward_return`.
            """
            # Inference of classification model.
            logits_class, x_embed = self.clf_module_forward(x)

            # Append classifier outputs to `out` if required.
            out = []
            if "logits_class" in self.forward_return:
                out.append(logits_class)
            if "x_embed" in self.forward_return:
                out.append(x_embed.flatten(start_dim=1))

            return out[0] if len(out) == 1 else tuple(out)


    def _normalize_rows(arr, eps=1e-12):
        arr = np.asarray(arr, dtype=float)
        row_sum = arr.sum(axis=1, keepdims=True)
        out = np.zeros_like(arr, dtype=float)
        valid = row_sum[:, 0] > eps
        if np.any(valid):
            out[valid] = arr[valid] / row_sum[valid]
        return out
    

    def _estimate_confusions(
        y: np.ndarray,
        observed: np.ndarray,
        posteriors: np.ndarray,
        n_classes: int,
        smoothing: float = 0.0,
    ) -> np.ndarray:
        """Estimate annotator confusion matrices from posterior class probabilities.

        Parameters
        ----------
        y : ndarray of shape (n_samples, n_annotators)
            Integer-encoded annotator labels for the covered samples. Missing
            entries must already be replaced by arbitrary dummy values because they
            are ignored using ``observed``.

        observed : ndarray of shape (n_samples, n_annotators)
            Boolean mask indicating which annotator labels are observed.

        posteriors : ndarray of shape (n_samples, n_classes)
            Posterior probabilities of the latent true classes for the covered
            samples.

        n_classes : int
            Number of classes.

        smoothing : float, default=0.0
            Additive smoothing applied to each row of each annotator confusion
            matrix before normalization.

        Returns
        -------
        confusions : ndarray of shape (n_annotators, n_classes, n_classes)
            Estimated annotator confusion matrices. Entry
            ``confusions[a, i, j]`` is the estimated probability that annotator
            ``a`` outputs observed label ``j`` when the latent true class is
            ``i``.

        Notes
        -----
        Each row of each annotator confusion matrix is estimated by aggregating the
        posterior responsibilities of all samples for the corresponding latent true
        class and normalizing over observed labels.
        """
        y = np.asarray(y, dtype=int)
        observed = np.asarray(observed)
        posteriors = np.asarray(posteriors)

        n_annotators = y.shape[1]
        confusions = np.empty((n_annotators, n_classes, n_classes), dtype=float)

        for annot in range(n_annotators):
            ann_obs = observed[:, annot]
            labels_ann = y[ann_obs, annot]

            for c in range(n_classes):
                row = np.full(n_classes, smoothing, dtype=float)
                if labels_ann.size > 0:
                    row += np.bincount(
                        labels_ann,
                        weights=posteriors[ann_obs, c],
                        minlength=n_classes,
                    )
                row_sum = row.sum()
                confusions[annot, c] = (
                    row / row_sum if row_sum > 0 else 1.0 / n_classes
                )

        return confusions


    def dawid_skene(
        y: np.ndarray,
        n_classes: int,
        max_iter: int = 100,
        tol: float = 1e-5,
        smoothing: float = 0.0,
        eps: float = 1e-12,
    ):
        """Estimate latent class posteriors and annotator confusion matrices
        with the Dawid-Skene algorithm.

        This function applies the expectation-maximization (EM) algorithm for
        multi-annotator classification. It estimates the posterior probabilities
        of the latent true class for each sample, the class prior, and one
        confusion matrix per annotator.

        Missing annotations are assumed to be encoded as ``-1``. Observed labels
        are assumed to be integer-encoded as ``0, ..., n_classes - 1``.

        Parameters
        ----------
        y : ndarray of shape (n_samples, n_annotators)
            Observed annotator labels. Each entry must either be an integer in
            ``{0, ..., n_classes - 1}`` or ``-1`` to indicate a missing label.

        n_classes : int
            Number of classes.

        max_iter : int, default=100
            Maximum number of EM iterations.

        tol : float, default=1e-5
            Convergence tolerance. The algorithm stops when the maximum absolute
            change in the posterior probabilities between two consecutive EM
            iterations is at most ``tol``.

        smoothing : float, default=0.0
            Additive smoothing applied to each row of each annotator confusion
            matrix during the M-step. This can improve numerical stability when
            some class-label combinations are rarely or never observed.

        eps : float, default=1e-12
            Small positive constant used for numerical stability, for example when
            normalizing probabilities or taking logarithms.

        Returns
        -------
        posteriors : ndarray of shape (n_samples, n_classes)
            Estimated posterior class probabilities for all samples. Samples with
            no observed annotations receive the estimated class prior.

        confusions : ndarray of shape (n_annotators, n_classes, n_classes)
            Estimated annotator confusion matrices. Entry
            ``confusions[a, i, j]`` corresponds to the estimated probability that
            annotator ``a`` assigns observed label ``j`` when the latent true class
            is ``i``.

        class_prior : ndarray of shape (n_classes,)
            Estimated prior probabilities of the latent classes.

        info : dict
            Dictionary with diagnostic information. It contains the following keys:

            - ``"n_iter"`` : int
            Number of EM iterations performed.
            - ``"converged"`` : bool
            Whether the convergence criterion was met.
            - ``"n_covered"`` : int
            Number of samples with at least one observed annotation.
            - ``"n_uncovered"`` : int
            Number of samples with no observed annotations.

        Notes
        -----
        The EM procedure is applied only to samples with at least one observed
        annotation. Samples without any annotation do not contribute to the
        parameter updates and are assigned the estimated class prior in the
        returned posterior matrix.

        The confusion matrices follow the convention

        ``confusions[a, i, j] = P(y^(a) = j | z = i)``,

        where ``z`` denotes the latent true class and ``y^(a)`` the label provided
        by annotator ``a``.
        """
        y = np.asarray(y)
        if y.ndim != 2:
            raise ValueError("y must have shape (n_samples, n_annotators).")
        if n_classes <= 0:
            raise ValueError("n_classes must be a positive integer.")

        observed_mask = y != -1
        invalid = observed_mask & ((y < 0) | (y >= n_classes))
        if np.any(invalid):
            invalid_labels = np.unique(y[invalid])
            raise ValueError(
                "Observed labels must be in {0, ..., n_classes - 1} and missing "
                f"labels must be -1, but found invalid labels "
                f"{invalid_labels.tolist()}."
            )

        n_samples, n_annotators = y.shape
        max_iter = int(max(max_iter, 1))
        tol = float(max(tol, 0.0))
        smoothing = float(max(smoothing, 0.0))

        # Keep only samples with at least one observed annotation for EM.
        covered = observed_mask.any(axis=1)
        n_covered = int(np.sum(covered))

        # Degenerate case: no observed labels anywhere.
        if n_covered == 0:
            class_prior = np.full(n_classes, 1.0 / n_classes, dtype=float)
            posteriors = np.tile(class_prior, (n_samples, 1))
            confusions = np.full(
                (n_annotators, n_classes, n_classes),
                1.0 / n_classes,
                dtype=float,
            )
            info = {
                "n_iter": 0,
                "converged": True,
                "n_covered": 0,
                "n_uncovered": int(n_samples),
            }
            return posteriors, confusions, class_prior, info

        y_cov = y[covered].copy()
        observed_cov = observed_mask[covered]

        # Replace missing entries by a harmless dummy label. These positions are
        # always ignored via `observed_cov`.
        y_cov[~observed_cov] = 0

        votes = compute_vote_vectors(
            y[covered], classes=np.arange(n_classes), missing_label=-1
        )
        posteriors_cov = _normalize_rows(votes, eps=eps)

        if not np.any(observed_cov.sum(axis=1) >= 2):
            y_mv = majority_vote(
                y=y[covered],
                missing_label=-1,
                classes=np.arange(n_classes),
            )
            posteriors_cov = np.eye(n_classes, dtype=float)[y_mv]
            class_prior = posteriors_cov.mean(axis=0)
            class_prior = class_prior / np.maximum(class_prior.sum(), eps)
            confusions = _estimate_confusions(
                y=y_cov,
                observed=observed_cov,
                posteriors=posteriors_cov,
                n_classes=n_classes,
                smoothing=smoothing,
            )
            posteriors = np.tile(class_prior, (n_samples, 1))
            posteriors[covered] = posteriors_cov
            info = {
                "n_iter": 0,
                "converged": True,
                "n_covered": n_covered,
                "n_uncovered": int(n_samples - n_covered),
                "fallback": "majority_voting",
            }
            return posteriors, confusions, class_prior, info

        class_prior = posteriors_cov.mean(axis=0)
        class_prior = class_prior / np.maximum(class_prior.sum(), eps)

        confusions = _estimate_confusions(
            y=y_cov,
            observed=observed_cov,
            posteriors=posteriors_cov,
            n_classes=n_classes,
            smoothing=smoothing,
        )

        converged = False
        n_iter = 0

        for it in range(max_iter):
            old = posteriors_cov.copy()
            log_post = np.tile(
                np.log(np.clip(class_prior, eps, 1.0)),
                (n_covered, 1),
            )

            for annot in range(n_annotators):
                ann_obs = observed_cov[:, annot]
                if not np.any(ann_obs):
                    continue

                labels_ann = np.ravel(y_cov[ann_obs, annot]).astype(
                    np.int64, copy=False
                )
                ann_likelihood = confusions[annot].T[labels_ann]
                log_post[ann_obs] += np.log(
                    np.clip(ann_likelihood, eps, 1.0)
                )

            log_post -= log_post.max(axis=1, keepdims=True)
            posteriors_cov = np.exp(log_post)
            posteriors_cov = _normalize_rows(posteriors_cov, eps=eps)

            class_prior = posteriors_cov.mean(axis=0)
            class_prior = class_prior / np.maximum(class_prior.sum(), eps)

            confusions = _estimate_confusions(
                y=y_cov,
                observed=observed_cov,
                posteriors=posteriors_cov,
                n_classes=n_classes,
                smoothing=smoothing,
            )

            n_iter = it + 1
            if np.max(np.abs(posteriors_cov - old)) <= tol:
                converged = True
                break

        # Expand posteriors back to the full sample set.
        posteriors = np.tile(class_prior, (n_samples, 1))
        posteriors[covered] = posteriors_cov

        info = {
            "n_iter": n_iter,
            "converged": converged,
            "n_covered": n_covered,
            "n_uncovered": int(n_samples - n_covered),
        }
        return posteriors, confusions, class_prior, info

except ImportError:  # pragma: no cover
    pass
