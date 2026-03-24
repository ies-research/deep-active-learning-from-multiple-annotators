try:
    import numpy as np

    from sklearn.utils.validation import check_array
    from torch import nn

    from skactiveml.base import SkactivemlClassifier
    from skactiveml.classifier.multiannotator._utils import (
        _MultiAnnotatorClassificationModule,
        _SkorchMultiAnnotatorClassifier,
    )
    from skactiveml.utils import (
        MISSING_LABEL,
        check_n_features,
        compute_vote_vectors,
        is_labeled,
    )


    class CrowdEMClassifier(_SkorchMultiAnnotatorClassifier):
        """Crowd EM Classifier

        Crowd EM Classifier is a neural-network-based multi-annotator
        classifier trained with expectation-maximization (EM). It jointly
        estimates latent class posteriors for the training samples, one
        confusion matrix per annotator, and the parameters of a neural
        classifier producing class probabilities for the samples.

        The model follows the Raykar-style learning-from-crowds setup in
        which annotators are represented by class-conditional confusion
        matrices and the classifier is updated in the M-step using the current
        latent class posteriors as soft targets.

        Parameters
        ----------
        clf_module : nn.Module or nn.Module.__class__
            A PyTorch module used as classification model. The module must
            return class logits as first output and may optionally return
            sample embeddings as second output. If no embeddings are returned,
            only the class logits are used by the estimator.
        n_annotators : int, default=None
            Number of annotators. If `n_annotators=None`, the number of
            annotators is inferred from `y` when calling `fit`.
        tol : float, default=1e-3
            Convergence tolerance for the EM objective. The algorithm stops
            when the absolute change in the objective value between two
            consecutive EM iterations is smaller than `tol`.
        max_iter : int, default=10
            Maximum number of EM iterations.
        annot_prior_full : float or array-like of shape (n_annotators,), \
                default=1.0
            Additive prior mass applied uniformly to each entry of every
            annotator confusion matrix before row normalization.
        annot_prior_diag : float or array-like of shape (n_annotators,), \
                default=0.0
            Additional additive prior mass applied to the diagonal entries of
            each annotator confusion matrix before row normalization.
        warm_start_m_step : bool, default=True
            If `True`, the neural classifier is reused across EM iterations.
            If `False`, the neural network is reinitialized before each M-step
            after the first iteration.
        neural_net_param_dict : dict, default=None
            Additional arguments for `skorch.net.NeuralNet`. If
            `neural_net_param_dict` is `None`, no additional arguments are
            added. `module`, `criterion`, `predict_nonlinearity`, and
            `train_split` are not allowed in this dictionary.
        sample_dtype : str or type, default=np.float32
            Dtype to which input samples are cast inside the estimator. If set
            to `None`, the input dtype is preserved.
        classes : array-like of shape (n_classes,), default=None
            Holds the label for each class. If `None`, the classes are
            determined during fitting from the observed annotator labels.
        cost_matrix : array-like of shape (n_classes, n_classes), default=None
            Cost matrix with `cost_matrix[i, j]` indicating the cost of
            predicting class `classes[j]` for a sample of class `classes[i]`.
            Can only be set if `classes` is not `None`.
        missing_label : scalar or string or np.nan or None, default=np.nan
            Value used to represent a missing label in the original annotator
            label matrix.
        random_state : int or RandomState or None, default=None
            Determines random numbers for methods relying on randomized
            decisions. Pass an int for reproducible results across multiple
            method calls.
        """

        _ALLOWED_EXTRA_OUTPUTS = {
            "logits",
            "embeddings",
            "annotator_perf",
            "annotator_class",
        }

        def __init__(
            self,
            clf_module,
            n_annotators=None,
            tol=1e-3,
            max_iter=10,
            annot_prior_full=1.0,
            annot_prior_diag=0.0,
            warm_start_m_step=True,
            neural_net_param_dict=None,
            sample_dtype=np.float32,
            classes=None,
            cost_matrix=None,
            missing_label=MISSING_LABEL,
            random_state=None,
        ):
            super(CrowdEMClassifier, self).__init__(
                multi_annotator_module=_RaykarModule,
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
            self.tol = tol
            self.max_iter = max_iter
            self.annot_prior_full = annot_prior_full
            self.annot_prior_diag = annot_prior_diag
            self.warm_start_m_step = warm_start_m_step

        def _fit(self, fit_function, X, y, **fit_params):
            """Fit the classifier and annotator model with EM.

            This method validates the multi-annotator training data, encodes
            annotator labels, initializes latent class posteriors, and then
            alternates between updating annotator confusion matrices and
            fitting the neural classifier on the current posterior estimates.

            Parameters
            ----------
            fit_function : {'fit', 'partial_fit'}
                Name of the caller, used to decide whether to reinitialize the
                neural network before training.
            X : array-like of shape (n_samples, ...)
                Training samples.
            y : array-like of shape (n_samples, n_annotators)
                Annotator label matrix. Missing entries must follow
                `self.missing_label`.
            **fit_params : dict
                Extra keyword arguments forwarded to
                `self.neural_net_.partial_fit`.

            Returns
            -------
            self : CrowdEMClassifier
                The fitted estimator.
            """
            X, y = self._validate_em_training_inputs(X=X, y=y)
            if not hasattr(self, "classes_"):
                self.classes_ = self._infer_classes(y)
            y = self._encode_annotator_labels(y)
            self._validate_em_hyperparameters()

            need_reinit = fit_function == "fit" or not hasattr(self, "neural_net_")
            if need_reinit:
                _, X, y = self.initialize(X=X, y=y, enforce_check_X_y=True)
            else:
                self._ensure_partial_fit_compatibility(y)

            X_train, y_train = self._extract_labeled_training_data(X=X, y=y)
            n_classes = len(self.classes_)
            prior_matrices, default_alpha = self._build_annotator_prior(
                n_classes=n_classes,
                n_annotators=self.n_annotators_,
            )
            self.objective_history_ = []
            self.n_iter_ = 0

            if X_train is None or y_train is None:
                self.Alpha_ = default_alpha
                self.Mu_ = np.empty((0, n_classes), dtype=np.float32)
                self.objective_history_ = np.asarray([], dtype=np.float64)
                return self

            mu = self._initialize_latent_posteriors(y=y_train)
            current_objective = -np.inf

            for iteration_idx in range(self.max_iter):
                self.n_iter_ = iteration_idx + 1
                self.Alpha_ = self._update_annotator_confusions(
                    y=y_train,
                    posteriors=mu,
                    prior_matrices=prior_matrices,
                )
                self._fit_classifier_m_step(
                    X=X_train,
                    y_posteriors=mu,
                    full_X=X,
                    full_y=y,
                    fit_params=fit_params,
                    iteration_idx=iteration_idx,
                )
                p_class = np.asarray(self.predict_proba(X_train), dtype=float)
                mu, annotator_likelihood = self._e_step(y=y_train, p_class=p_class)
                objective = self._compute_objective(
                    p_class=p_class,
                    annotator_likelihood=annotator_likelihood,
                    posteriors=mu,
                    alpha=self.Alpha_,
                    prior_matrices=prior_matrices,
                )
                self.objective_history_.append(float(objective))
                if np.abs(objective - current_objective) < self.tol:
                    break
                current_objective = objective

            self.Mu_ = mu.astype(np.float32, copy=False)
            self.objective_history_ = np.asarray(
                self.objective_history_, dtype=np.float64
            )
            return self

        def predict(self, X, extra_outputs=None):
            """Return class predictions for the test samples `X`.

            By default, this method returns only the class predictions
            `y_pred`. If `extra_outputs` is provided, a tuple is returned
            whose first element is `y_pred` and whose remaining elements are
            the requested additional outputs, in the order specified by
            `extra_outputs`.

            Parameters
            ----------
            X : array-like of shape (n_samples, ...)
                Test samples.
            extra_outputs : None or str or sequence of str, default=None
                Names of additional outputs to return next to `y_pred`. The
                names must be a subset of the following keys:

                - "logits" : Additionally return the class-membership logits
                  `L_class` for the samples in `X`.
                - "embeddings" : Additionally return the learned embeddings
                  `X_embed` for the samples in `X`.
                - "annotator_perf" : Additionally return the estimated
                  annotator correctness probabilities `P_perf` for each
                  sample-annotator pair.
                - "annotator_class" : Additionally return the annotator-class
                  probability estimates `P_annot` for each sample, annotator,
                  and class.

            Returns
            -------
            y_pred : numpy.ndarray of shape (n_samples,)
                Class predictions of the test samples.
            *extras : numpy.ndarray, optional
                Only returned if `extra_outputs` is not `None`. In that case,
                the method returns a tuple whose first element is `y_pred`
                and whose remaining elements correspond to the requested
                outputs in the order given by `extra_outputs`. Potential
                outputs are:

                - `L_class` : `np.ndarray` of shape `(n_samples, n_classes)`,
                  where `L_class[n, c]` is the logit for class
                  `classes_[c]` of sample `X[n]`.
                - `X_embed` : `np.ndarray` of shape `(n_samples, ...)`, where
                  `X_embed[n]` refers to the learned embedding for sample
                  `X[n]`.
                - `P_perf` : `np.ndarray` of shape `(n_samples, n_annotators)`,
                  where `P_perf[n, m]` is the estimated probability that
                  annotator `m` labels sample `X[n]` correctly.
                - `P_annot` : `np.ndarray` of shape
                  `(n_samples, n_annotators, n_classes)`, where
                  `P_annot[n, m, c]` is the probability that annotator `m`
                  outputs class `classes_[c]` for sample `X[n]`.
            """
            return SkactivemlClassifier.predict(
                self,
                X=X,
                extra_outputs=extra_outputs,
            )

        def predict_proba(self, X, extra_outputs=None):
            """Return class probability estimates for the test samples `X`.

            By default, this method returns only the class probabilities `P`.
            If `extra_outputs` is provided, a tuple is returned whose first
            element is `P` and whose remaining elements are the requested
            additional outputs, in the order specified by `extra_outputs`.

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
                - "annotator_perf" : Additionally return the estimated
                  annotator correctness probabilities `P_perf` for each
                  sample-annotator pair.
                - "annotator_class" : Additionally return the annotator-class
                  probability estimates `P_annot` for each sample, annotator,
                  and class.

            Returns
            -------
            P : numpy.ndarray of shape (n_samples, n_classes)
                Class probabilities of the test samples. Classes are ordered
                according to `self.classes_`.
            *extras : numpy.ndarray, optional
                Only returned if `extra_outputs` is not `None`. In that case,
                the method returns a tuple whose first element is `P` and
                whose remaining elements correspond to the requested outputs
                in the order given by `extra_outputs`. Potential outputs are:

                - `L_class` : `np.ndarray` of shape `(n_samples, n_classes)`,
                  where `L_class[n, c]` is the logit for class
                  `classes_[c]` of sample `X[n]`.
                - `X_embed` : `np.ndarray` of shape `(n_samples, ...)`, where
                  `X_embed[n]` refers to the learned embedding for sample
                  `X[n]`.
                - `P_perf` : `np.ndarray` of shape `(n_samples, n_annotators)`,
                  where `P_perf[n, m]` is the estimated probability that
                  annotator `m` labels sample `X[n]` correctly.
                - `P_annot` : `np.ndarray` of shape
                  `(n_samples, n_annotators, n_classes)`, where
                  `P_annot[n, m, c]` is the probability that annotator `m`
                  outputs class `classes_[c]` for sample `X[n]`.
            """
            self._validate_data_kwargs()
            X = check_array(X, **self.check_X_dict_)
            check_n_features(
                self, X, reset=not hasattr(self, "n_features_in_")
            )
            extra_outputs = self._normalize_extra_outputs(
                extra_outputs=extra_outputs,
                allowed_names=type(self)._ALLOWED_EXTRA_OUTPUTS,
            )

            if not hasattr(self, "neural_net_"):
                self.initialize()

            net = self.neural_net_.module_
            old_forward_return = net.forward_return
            forward_outputs = {"proba": (0, nn.Softmax(dim=-1))}
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
                named_outputs = {
                    name: value
                    for name, value in zip(model_extra_outputs, fw_out[1:])
                }
            else:
                p_class = fw_out
                named_outputs = {}

            if any(
                name in extra_outputs
                for name in ("annotator_perf", "annotator_class")
            ):
                if not hasattr(self, "Alpha_"):
                    raise RuntimeError(
                        "Annotator outputs requested, but the classifier has "
                        "no fitted annotator confusion matrices."
                    )
                p_class_float = np.asarray(p_class, dtype=float)
                if "annotator_perf" in extra_outputs:
                    conf_diag = np.diagonal(self.Alpha_, axis1=1, axis2=2)
                    named_outputs["annotator_perf"] = np.einsum(
                        "nk,mk->nm", p_class_float, conf_diag, optimize=True
                    ).astype(np.float32, copy=False)
                if "annotator_class" in extra_outputs:
                    named_outputs["annotator_class"] = np.einsum(
                        "nk,mkl->nml",
                        p_class_float,
                        self.Alpha_,
                        optimize=True,
                    ).astype(np.float32, copy=False)

            self._initialize_fallbacks(p_class)
            if not extra_outputs:
                return p_class
            return (p_class, *[named_outputs[name] for name in extra_outputs])

        def _build_neural_net_param_overrides(self, X, y):
            del X, y
            return {"criterion__reduction": "mean"}

        def _validate_em_training_inputs(self, X, y):
            vd_kwargs = self._validate_data_kwargs()
            X, y, _ = self._validate_data(X=X, y=y, **vd_kwargs)
            if y is None:
                raise ValueError("`y` must not be None.")
            if y.ndim != 2:
                raise ValueError(
                    "`y` must have shape (n_samples, n_annotators) for "
                    "multi-annotator training."
                )
            return X, y

        def _infer_classes(self, y):
            labeled = y[is_labeled(y, missing_label=-1)]
            if labeled.size == 0:
                raise ValueError(
                    "Cannot infer classes from `y` because all labels are missing."
                )
            return np.unique(labeled)

        def _encode_annotator_labels(self, y):
            y_encoded = np.full(y.shape, fill_value=-1, dtype=np.int64)
            observed = is_labeled(y, missing_label=-1)
            for class_idx, class_label in enumerate(np.asarray(self.classes_)):
                y_encoded[y == class_label] = class_idx
            if np.any((y_encoded < 0) & observed):
                unknown = np.unique(y[(y_encoded < 0) & observed])
                raise ValueError(
                    f"Observed labels {unknown.tolist()} are not in classes_."
                )
            return y_encoded

        def _ensure_partial_fit_compatibility(self, y):
            if not hasattr(self, "n_annotators_"):
                self.n_annotators_ = y.shape[1]
            if self.n_annotators_ != y.shape[1]:
                raise ValueError(
                    f"`n_annotators={self.n_annotators_}` does not match "
                    f"{y.shape[1]} as the number of columns in `y`."
                )

        def _validate_em_hyperparameters(self):
            if not isinstance(self.max_iter, int) or self.max_iter < 1:
                raise ValueError("`max_iter` must be a positive integer.")
            if not isinstance(self.tol, (int, float)) or self.tol <= 0:
                raise ValueError("`tol` must be a positive scalar.")
            if not isinstance(self.warm_start_m_step, bool):
                raise TypeError("`warm_start_m_step` must be a bool.")

        def _build_annotator_prior(self, n_classes, n_annotators):
            full = self._coerce_annotator_prior(
                self.annot_prior_full,
                name="annot_prior_full",
                n_annotators=n_annotators,
                allow_zero=False,
            )
            diag = self._coerce_annotator_prior(
                self.annot_prior_diag,
                name="annot_prior_diag",
                n_annotators=n_annotators,
                allow_zero=True,
            )
            prior = np.ones((n_annotators, n_classes, n_classes), dtype=float)
            for annotator_idx in range(n_annotators):
                prior[annotator_idx] *= full[annotator_idx]
                prior[annotator_idx] += np.eye(n_classes) * diag[annotator_idx]
            alpha_counts = prior - 1.0
            alpha_sums = alpha_counts.sum(axis=-1, keepdims=True)
            default_alpha = np.divide(
                alpha_counts,
                alpha_sums,
                out=np.full_like(alpha_counts, 1.0 / n_classes, dtype=float),
                where=alpha_sums != 0,
            )
            return prior, default_alpha

        def _coerce_annotator_prior(
            self, prior, name, n_annotators, allow_zero
        ):
            if np.isscalar(prior):
                prior = np.full(n_annotators, prior, dtype=float)
            else:
                prior = np.asarray(prior, dtype=float).reshape(-1)
            if prior.shape[0] != n_annotators:
                raise ValueError(
                    f"`{name}` must have shape (n_annotators,), got {prior}."
                )
            if allow_zero:
                is_invalid = np.any(prior < 0)
            else:
                is_invalid = np.any(prior <= 0)
            if is_invalid:
                comparator = "non-negative" if allow_zero else "positive"
                raise ValueError(f"`{name}` must contain {comparator} values.")
            return prior

        def _extract_labeled_training_data(self, X, y):
            observed = is_labeled(y, missing_label=-1)
            covered = observed.any(axis=1)
            if not np.any(covered):
                return None, None
            return X[covered], y[covered].astype(np.int64, copy=False)

        def _initialize_latent_posteriors(self, y):
            mu = compute_vote_vectors(
                y=y,
                classes=np.arange(len(self.classes_)),
                missing_label=-1,
            )
            return _normalize_rows(mu).astype(np.float32, copy=False)

        def _update_annotator_confusions(self, y, posteriors, prior_matrices):
            n_classes = len(self.classes_)
            observed = is_labeled(y, missing_label=-1)
            alpha = np.zeros(
                (self.n_annotators_, n_classes, n_classes), dtype=float
            )
            eye = np.eye(n_classes, dtype=float)
            for annotator_idx in range(self.n_annotators_):
                if np.any(observed[:, annotator_idx]):
                    labels = y[observed[:, annotator_idx], annotator_idx].astype(int)
                    y_onehot = eye[labels]
                    alpha[annotator_idx] = (
                        posteriors[observed[:, annotator_idx]].T @ y_onehot
                    ) + prior_matrices[annotator_idx] - 1.0
                else:
                    alpha[annotator_idx] = prior_matrices[annotator_idx] - 1.0
            alpha_sums = alpha.sum(axis=-1, keepdims=True)
            return np.divide(
                alpha,
                alpha_sums,
                out=np.full_like(alpha, 1.0 / n_classes, dtype=float),
                where=alpha_sums != 0,
            )

        def _fit_classifier_m_step(
            self,
            X,
            y_posteriors,
            full_X,
            full_y,
            fit_params,
            iteration_idx,
        ):
            if iteration_idx > 0 and not self.warm_start_m_step:
                _, _, _ = self.initialize(
                    X=full_X, y=full_y, enforce_check_X_y=True
                )
            self.neural_net_.partial_fit(
                X,
                np.asarray(y_posteriors, dtype=np.float32),
                **fit_params,
            )

        def _e_step(self, y, p_class):
            n_samples = y.shape[0]
            n_classes = len(self.classes_)
            observed = is_labeled(y, missing_label=-1)
            annotator_likelihood = np.ones((n_samples, n_classes), dtype=float)
            for annotator_idx in range(self.n_annotators_):
                obs_idx = observed[:, annotator_idx]
                if not np.any(obs_idx):
                    continue
                labels = y[obs_idx, annotator_idx].astype(int)
                annotator_likelihood[obs_idx] *= self.Alpha_[
                    annotator_idx, :, labels
                ]
            mu = _normalize_rows(p_class * annotator_likelihood)
            return mu.astype(np.float32, copy=False), annotator_likelihood

        def _compute_objective(
            self,
            p_class,
            annotator_likelihood,
            posteriors,
            alpha,
            prior_matrices,
        ):
            eps = np.finfo(float).eps
            prior_alpha = np.sum((prior_matrices - 1.0) * np.log(alpha + eps))
            log_likelihood = np.sum(
                posteriors * np.log((p_class * annotator_likelihood) + eps)
            )
            return log_likelihood + prior_alpha


    class _RaykarModule(_MultiAnnotatorClassificationModule):
        def __init__(self, clf_module, clf_module_param_dict):
            super().__init__(
                clf_module=clf_module,
                clf_module_param_dict=clf_module_param_dict,
                default_forward_outputs="logits_class",
                full_forward_outputs=["logits_class", "x_embed"],
            )

        def forward(self, x):
            logits_class, x_embed = self.clf_module_forward(x)
            out = []
            if "logits_class" in self.forward_return:
                out.append(logits_class)
            if "x_embed" in self.forward_return:
                out.append(x_embed.flatten(start_dim=1))
            return out[0] if len(out) == 1 else tuple(out)


    def _normalize_rows(arr, eps=1e-12):
        arr = np.asarray(arr, dtype=float)
        row_sum = arr.sum(axis=1, keepdims=True)
        return np.divide(
            arr,
            row_sum,
            out=np.full_like(arr, 1.0 / arr.shape[1], dtype=float),
            where=row_sum > eps,
        )

    __all__ = ["CrowdEMClassifier"]
except ImportError:
    pass
