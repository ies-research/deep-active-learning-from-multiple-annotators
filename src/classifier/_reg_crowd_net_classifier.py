try:
    import numpy as np
    import torch

    from sklearn.utils.validation import check_array
    from torch import nn
    from torch.nn import functional as F

    from skactiveml.classifier.multiannotator._utils import (
        _MultiAnnotatorCollate,
        _MultiAnnotatorClassificationModule,
        _SkorchMultiAnnotatorClassifier,
    )
    from skactiveml.utils import (
        MISSING_LABEL,
        check_n_features,
    )


    class RegCrowdNetClassifier(_SkorchMultiAnnotatorClassifier):
        """Regularized crowd network classifier.

        RegCrowdNet jointly learns a classifier for the latent class
        distribution and one confusion matrix per annotator. Training is
        carried out end-to-end on observed sample-annotator pairs while the
        confusion matrices are regularized according to one of the variants
        proposed in the literature.

        Parameters
        ----------
        clf_module : nn.Module or nn.Module.__class__
            A PyTorch module used as classification backbone. The module must
            return class logits as first output and may optionally return
            sample embeddings as second output.
        regularization : {"trace-reg", "geo-reg-f", "geo-reg-w"}, \
                default="trace-reg"
            Regularization applied to the annotator confusion model.
        lmbda : float, default=0.01
            Weight of the regularization term.
        n_annotators : int, default=None
            Number of annotators. If `n_annotators=None`, it is inferred from
            `y` during fitting.
        neural_net_param_dict : dict, default=None
            Additional keyword arguments for the underlying skorch
            `NeuralNet`.
        sample_dtype : str or type, default=np.float32
            Dtype to which input samples are cast inside the estimator.
        classes : array-like of shape (n_classes,), default=None
            Class labels. If `None`, classes are inferred from the observed
            annotator labels at fit time.
        cost_matrix : array-like of shape (n_classes, n_classes), default=None
            Cost matrix passed through to `SkactivemlClassifier`.
        missing_label : scalar or string or np.nan or None, default=np.nan
            Value representing a missing label in the original annotator label
            matrix.
        random_state : int or RandomState or None, default=None
            Random seed used for reproducibility.
        """

        _ALLOWED_EXTRA_OUTPUTS = {
            "logits",
            "embeddings",
            "annotator_perf",
            "annotator_class",
            "annotator_confusion_matrices",
        }
        _VALID_REGULARIZATIONS = {"trace-reg", "geo-reg-f", "geo-reg-w"}

        def __init__(
            self,
            clf_module,
            regularization="trace-reg",
            lmbda=0.01,
            n_annotators=None,
            neural_net_param_dict=None,
            sample_dtype=np.float32,
            classes=None,
            cost_matrix=None,
            missing_label=MISSING_LABEL,
            random_state=None,
        ):
            super(RegCrowdNetClassifier, self).__init__(
                multi_annotator_module=_RegCrowdNetModule,
                clf_module=clf_module,
                n_annotators=n_annotators,
                criterion=_RegCrowdNetLoss,
                sample_dtype=sample_dtype,
                classes=classes,
                missing_label=missing_label,
                cost_matrix=cost_matrix,
                random_state=random_state,
                neural_net_param_dict=neural_net_param_dict,
            )
            self.regularization = regularization
            self.lmbda = lmbda

        def _fit(self, fit_function, X, y, **fit_params):
            """Fit the classifier on a multi-annotator label matrix."""
            self._check_multiannotator_y(y)
            super()._fit(fit_function, X, y, **fit_params)
            if hasattr(self, "neural_net_"):
                self.annotator_confusion_matrices_ = (
                    self._current_confusion_matrices()
                )
            return self

        def predict_proba(self, X, extra_outputs=None):
            """Return class probability estimates for the test samples `X`."""
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

            if "annotator_perf" in extra_outputs:
                forward_outputs["annotator_perf"] = (out_idx, None)
                forward_returns.append("p_annot_perf")
                model_extra_outputs.append("annotator_perf")
                out_idx += 1

            if "annotator_class" in extra_outputs:
                forward_outputs["annotator_class"] = (out_idx, torch.exp)
                forward_returns.append("log_p_annot")
                model_extra_outputs.append("annotator_class")
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

            if "annotator_confusion_matrices" in extra_outputs:
                named_outputs["annotator_confusion_matrices"] = (
                    self._current_confusion_matrices()
                )

            self._initialize_fallbacks(p_class)
            if not extra_outputs:
                return p_class
            return (p_class, *[named_outputs[name] for name in extra_outputs])

        def _build_neural_net_param_overrides(self, X, y):
            self._validate_reg_crowd_net_hyperparameters()
            collate_fn = _MultiAnnotatorCollate(missing_label=-1)
            return {
                "criterion__regularization": self.regularization,
                "criterion__lmbda": float(self.lmbda),
                "criterion__reduction": "mean",
                "module__n_classes": len(self.classes_),
                "module__n_annotators": self.n_annotators_,
                "module__regularization": self.regularization,
                "iterator_train__collate_fn": collate_fn,
            }

        @staticmethod
        def _check_multiannotator_y(y):
            if y is None:
                raise ValueError("`y` must not be None.")
            if np.asarray(y).ndim != 2:
                raise ValueError(
                    "`y` must have shape (n_samples, n_annotators) for "
                    "multi-annotator training."
                )

        def _validate_reg_crowd_net_hyperparameters(self):
            if self.regularization not in self._VALID_REGULARIZATIONS:
                raise ValueError(
                    "`regularization` must be one of "
                    f"{sorted(self._VALID_REGULARIZATIONS)}."
                )
            if not isinstance(
                self.lmbda, (int, float, np.integer, np.floating)
            ) or self.lmbda < 0:
                raise ValueError("`lmbda` must be a non-negative scalar.")

        def _current_confusion_matrices(self):
            confusion = F.softmax(
                self.neural_net_.module_.ap_confs.detach(), dim=-1
            )
            confusion = confusion.cpu().numpy().astype(np.float32, copy=False)
            return confusion


    class _RegCrowdNetLoss(nn.Module):
        """Loss for RegCrowdNet training."""

        def __init__(
            self,
            regularization="trace-reg",
            lmbda=0.01,
            reduction="mean",
        ):
            super().__init__()
            self.regularization = regularization
            self.lmbda = float(lmbda)
            self.reduction = reduction

        def forward(self, y_pred, y_true):
            if not isinstance(y_pred, tuple) or len(y_pred) != 3:
                raise ValueError(
                    "RegCrowdNet loss expects "
                    "(logits_class, log_p_annot, annotator_confusions)."
                )

            logits_class, log_p_annot, annotator_confusions = y_pred
            loss = F.nll_loss(log_p_annot, y_true, reduction=self.reduction)

            if self.lmbda <= 0:
                return loss

            reg_term = self._regularization_term(
                logits_class=logits_class,
                annotator_confusions=annotator_confusions,
            )
            return loss + self.lmbda * reg_term

        def _regularization_term(self, logits_class, annotator_confusions):
            if self.regularization == "trace-reg":
                reg_term = annotator_confusions.diagonal(
                    offset=0, dim1=-2, dim2=-1
                ).sum(-1).mean()
            elif self.regularization == "geo-reg-f":
                p_class = F.softmax(logits_class, dim=-1)
                reg_term = -torch.logdet(p_class.T @ p_class)
                if (not torch.isfinite(reg_term)) or reg_term > 100:
                    reg_term = logits_class.new_zeros(())
            elif self.regularization == "geo-reg-w":
                confusion_rows = annotator_confusions.transpose(1, 2).flatten(
                    start_dim=0, end_dim=1
                )
                reg_term = -torch.logdet(confusion_rows.T @ confusion_rows)
                if (not torch.isfinite(reg_term)) or reg_term > 100:
                    reg_term = logits_class.new_zeros(())
            else:
                raise ValueError(
                    "`regularization` must be one of "
                    "['trace-reg', 'geo-reg-f', 'geo-reg-w']."
                )
            return reg_term


    class _RegCrowdNetModule(_MultiAnnotatorClassificationModule):
        """Auxiliary module for :class:`RegCrowdNetClassifier`."""

        def __init__(
            self,
            n_classes,
            n_annotators,
            clf_module,
            clf_module_param_dict,
            regularization,
        ):
            super().__init__(
                clf_module=clf_module,
                clf_module_param_dict=clf_module_param_dict,
                default_forward_outputs=[
                    "log_p_annot",
                    "logits_class",
                    "annotator_confusion_matrices",
                ],
                full_forward_outputs=[
                    "logits_class",
                    "x_embed",
                    "p_annot_perf",
                    "log_p_annot",
                    "annotator_confusion_matrices",
                ],
            )
            self.n_classes = int(n_classes)
            self.n_annotators = int(n_annotators)
            self.regularization = regularization
            self.ap_confs = nn.Parameter(
                self._initial_confusions(
                    n_classes=self.n_classes,
                    n_annotators=self.n_annotators,
                    regularization=self.regularization,
                )
            )

        def forward(self, x, input_ids=None):
            logits_class, x_embed = self.clf_module_forward(x)

            out = []
            if "logits_class" in self.forward_return:
                out.append(logits_class)
            if "x_embed" in self.forward_return:
                out.append(x_embed.flatten(start_dim=1))

            need_annotator_output = any(
                name in self.forward_return
                for name in (
                    "p_annot_perf",
                    "log_p_annot",
                    "annotator_confusion_matrices",
                )
            )
            if need_annotator_output:
                log_p_class = F.log_softmax(logits_class, dim=-1)
                annotator_log_confusions = F.log_softmax(self.ap_confs, dim=-1)
                annotator_confusions = annotator_log_confusions.exp()

                if "p_annot_perf" in self.forward_return:
                    diag_idx = torch.arange(
                        self.n_classes, device=annotator_confusions.device
                    )
                    perf_per_class = annotator_confusions[
                        :, diag_idx, diag_idx
                    ]
                    p_annot_perf = torch.einsum(
                        "nc,ac->na", log_p_class.exp(), perf_per_class
                    )
                    out.append(p_annot_perf)

                if "log_p_annot" in self.forward_return:
                    if isinstance(input_ids, torch.Tensor):
                        log_p_class_sel = log_p_class.index_select(
                            0, input_ids[:, 0]
                        )
                        log_conf_sel = annotator_log_confusions.index_select(
                            0, input_ids[:, 1]
                        )
                        log_p_annot = torch.logsumexp(
                            log_p_class_sel[:, :, None] + log_conf_sel,
                            dim=1,
                        )
                    else:
                        log_p_annot = torch.logsumexp(
                            log_p_class[:, None, :, None]
                            + annotator_log_confusions[None, :, :, :],
                            dim=2,
                        )
                    out.append(log_p_annot)

                if "annotator_confusion_matrices" in self.forward_return:
                    out.append(annotator_confusions)

            return out[0] if len(out) == 1 else tuple(out)

        @staticmethod
        def _initial_confusions(n_classes, n_annotators, regularization):
            if regularization == "trace-reg":
                init = 6.0 * torch.eye(n_classes, dtype=torch.float32) - 5.0
            elif regularization in {"geo-reg-f", "geo-reg-w"}:
                init = torch.eye(n_classes, dtype=torch.float32)
            else:
                raise ValueError(
                    "`regularization` must be one of "
                    "['trace-reg', 'geo-reg-f', 'geo-reg-w']."
                )
            return init.unsqueeze(0).repeat(n_annotators, 1, 1)


    __all__ = ["RegCrowdNetClassifier"]
except ImportError:  # pragma: no cover
    pass
