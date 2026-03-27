try:
    import numpy as np
    import torch

    from sklearn.utils.validation import check_array
    from torch import nn
    from torch.nn import functional as F

    from skactiveml.classifier.multiannotator._utils import (
        _MultiAnnotatorClassificationModule,
        _MultiAnnotatorCollate,
        _SkorchMultiAnnotatorClassifier,
    )
    from skactiveml.utils import (
        MISSING_LABEL,
        check_n_features,
    )


    class DALCLikeClassifier(_SkorchMultiAnnotatorClassifier):
        """DALC-like multi-annotator classifier.

        This classifier follows the central modeling idea of DALC:
        sample-specific annotator expertise is represented via a low-rank
        interaction between a learned annotator embedding ``u_j`` and a linear
        projection ``F x_i`` of the sample representation. In contrast to the
        original Bayesian DALC formulation, this implementation trains the
        model end-to-end with a standard neural objective by marginalizing the
        latent true label and using a multiclass extension with a scalar
        correctness probability and uniform off-diagonal error mass.

        Parameters
        ----------
        clf_module : nn.Module or nn.Module.__class__
            Backbone classification module producing class logits and
            optionally sample embeddings.
        annotator_embed_dim : int, default=16
            Dimensionality of the learned annotator embeddings.
        sample_projection_bias : bool, default=False
            Whether the linear projection that maps sample embeddings into the
            annotator-expertise space uses a bias term.
        annotator_bias : bool, default=False
            Whether to add an annotator-specific scalar bias to the expertise
            logits.
        n_annotators : int, default=None
            Number of annotators. If `None`, inferred from `y` during fit.
        neural_net_param_dict : dict, default=None
            Additional skorch neural-net parameters.
        sample_dtype : str or type, default=np.float32
            Dtype to which input samples are cast.
        classes : array-like of shape (n_classes,), default=None
            Class labels. If `None`, inferred during fit.
        cost_matrix : array-like of shape (n_classes, n_classes), default=None
            Misclassification cost matrix.
        missing_label : scalar or string or np.nan or None, default=np.nan
            Missing-label marker in the multi-annotator label matrix.
        random_state : int or RandomState or None, default=None
            Random seed used by the estimator.
        """

        _ALLOWED_EXTRA_OUTPUTS = {
            "logits",
            "embeddings",
            "annotator_perf",
            "annotator_class",
            "annotator_embeddings",
        }

        def __init__(
            self,
            clf_module,
            annotator_embed_dim=16,
            sample_projection_bias=False,
            annotator_bias=False,
            n_annotators=None,
            neural_net_param_dict=None,
            sample_dtype=np.float32,
            classes=None,
            cost_matrix=None,
            missing_label=MISSING_LABEL,
            random_state=None,
        ):
            super(DALCLikeClassifier, self).__init__(
                multi_annotator_module=_DALCLikeModule,
                clf_module=clf_module,
                n_annotators=n_annotators,
                criterion=nn.NLLLoss,
                sample_dtype=sample_dtype,
                classes=classes,
                missing_label=missing_label,
                cost_matrix=cost_matrix,
                random_state=random_state,
                neural_net_param_dict=neural_net_param_dict,
            )
            self.annotator_embed_dim = int(annotator_embed_dim)
            self.sample_projection_bias = bool(sample_projection_bias)
            self.annotator_bias = bool(annotator_bias)

        def _fit(self, fit_function, X, y, **fit_params):
            self._check_multiannotator_y(y)
            self._validate_hyperparameters()
            return super()._fit(fit_function, X, y, **fit_params)

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
                forward_outputs["annotator_class"] = (out_idx, None)
                forward_returns.append("p_annot")
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

            if "annotator_embeddings" in extra_outputs:
                named_outputs["annotator_embeddings"] = (
                    net.annotator_embeddings_.detach().cpu().numpy()
                )

            self._initialize_fallbacks(p_class)
            if not extra_outputs:
                return p_class
            return (p_class, *[named_outputs[name] for name in extra_outputs])

        def _build_neural_net_param_overrides(self, X, y):
            collate_fn = _MultiAnnotatorCollate(missing_label=-1)
            return {
                "criterion__reduction": "mean",
                "module__n_classes": len(self.classes_),
                "module__n_annotators": self.n_annotators_,
                "module__annotator_embed_dim": self.annotator_embed_dim,
                "module__sample_projection_bias": self.sample_projection_bias,
                "module__annotator_bias": self.annotator_bias,
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

        def _validate_hyperparameters(self):
            if self.annotator_embed_dim <= 0:
                raise ValueError("`annotator_embed_dim` must be a positive integer.")


    class _DALCLikeModule(_MultiAnnotatorClassificationModule):
        """Auxiliary module for :class:`DALCLikeClassifier`."""

        def __init__(
            self,
            n_classes,
            n_annotators,
            annotator_embed_dim,
            sample_projection_bias,
            annotator_bias,
            clf_module,
            clf_module_param_dict,
        ):
            super().__init__(
                clf_module=clf_module,
                clf_module_param_dict=clf_module_param_dict,
                default_forward_outputs="log_p_annot",
                full_forward_outputs=[
                    "log_p_annot",
                    "logits_class",
                    "x_embed",
                    "p_annot_perf",
                    "p_annot",
                    "annotator_embeddings",
                ],
            )
            self.n_classes = int(n_classes)
            self.n_annotators = int(n_annotators)
            self.annotator_embed_dim = int(annotator_embed_dim)

            if self.n_classes < 2:
                raise ValueError("DALCLikeClassifier requires at least two classes.")

            self.sample_projection = nn.LazyLinear(
                self.annotator_embed_dim,
                bias=bool(sample_projection_bias),
            )
            self.annotator_embeddings_ = nn.Parameter(
                torch.empty(self.n_annotators, self.annotator_embed_dim)
            )
            if annotator_bias:
                self.annotator_bias_ = nn.Parameter(
                    torch.zeros(self.n_annotators, dtype=torch.float32)
                )
            else:
                self.register_parameter("annotator_bias_", None)

            nn.init.normal_(self.annotator_embeddings_, mean=0.0, std=0.02)

        def forward(self, x, input_ids=None):
            logits_class, x_embed = self.clf_module_forward(x)
            x_embed = x_embed.flatten(start_dim=1)
            x_topics = self.sample_projection(x_embed)

            out = []
            if "log_p_annot" in self.forward_return:
                log_p_annot = self._annotation_log_probs(
                    logits_class=logits_class,
                    x_topics=x_topics,
                    input_ids=input_ids,
                )
                out.append(log_p_annot)
            if "logits_class" in self.forward_return:
                out.append(logits_class)
            if "x_embed" in self.forward_return:
                out.append(x_embed)

            need_pair_outputs = any(
                name in self.forward_return
                for name in ("p_annot_perf", "p_annot")
            )
            if need_pair_outputs:
                p_class = F.softmax(logits_class, dim=-1)
                p_annot_perf = self._annotator_correctness(x_topics=x_topics)
                if "p_annot_perf" in self.forward_return:
                    out.append(p_annot_perf)
                if "p_annot" in self.forward_return:
                    out.append(
                        self._annotator_label_distribution(
                            p_class=p_class,
                            p_annot_perf=p_annot_perf,
                        )
                    )

            if "annotator_embeddings" in self.forward_return:
                out.append(self.annotator_embeddings_)

            return out[0] if len(out) == 1 else tuple(out)

        def _annotator_correctness(self, x_topics):
            eta_logits = x_topics @ self.annotator_embeddings_.T
            if self.annotator_bias_ is not None:
                eta_logits = eta_logits + self.annotator_bias_[None, :]
            return torch.sigmoid(eta_logits)

        def _annotator_label_distribution(self, p_class, p_annot_perf):
            p_class = p_class[:, None, :]
            p_annot_perf = p_annot_perf[:, :, None]
            off_diag = (1.0 - p_annot_perf) / float(self.n_classes - 1)
            return p_class * p_annot_perf + (1.0 - p_class) * off_diag

        def _annotation_log_probs(self, logits_class, x_topics, input_ids):
            p_class = F.softmax(logits_class, dim=-1)
            p_annot_perf = self._annotator_correctness(x_topics=x_topics)
            if isinstance(input_ids, torch.Tensor):
                p_class_sel = p_class.index_select(0, input_ids[:, 0])
                p_annot_perf_sel = p_annot_perf[
                    input_ids[:, 0], input_ids[:, 1]
                ]
                off_diag = (1.0 - p_annot_perf_sel) / float(self.n_classes - 1)
                p_annot = (
                    p_class_sel * p_annot_perf_sel[:, None]
                    + (1.0 - p_class_sel) * off_diag[:, None]
                )
                return torch.log(torch.clamp(p_annot, min=1e-12))

            p_annot = self._annotator_label_distribution(
                p_class=p_class,
                p_annot_perf=p_annot_perf,
            )
            return torch.log(torch.clamp(p_annot, min=1e-12))


    __all__ = ["DALCLikeClassifier"]
except ImportError:  # pragma: no cover
    pass
