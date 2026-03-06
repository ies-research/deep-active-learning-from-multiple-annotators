try:
    import math
    import warnings
    import numpy as np
    import torch

    from sklearn.utils.validation import check_array
    from torch import nn
    from torch.nn import KLDivLoss
    from torch.nn import functional as F

    from skactiveml.base import SkactivemlClassifier
    from skactiveml.classifier.multiannotator._utils import (
        _MultiAnnotatorClassificationModule,
        _SkorchMultiAnnotatorClassifier,
    )
    from skactiveml.utils import MISSING_LABEL, check_n_features, check_scalar

    from skactiveml.classifier.multiannotator._annot_mix_classifier import (
        _MixUpCollate,
    )

    class GatingClassifier(_SkorchMultiAnnotatorClassifier):
        """Gated multi-annotator classifier with structured confusion modeling.

        This classifier extends a standard multi-annotator model by decomposing
        annotator confusion for a sample-annotator pair `(x, a)` into:

        - an annotator-only base confusion term `B_a`, and
        - an optional sample-dependent residual term `R_{x,a}`.

        The confusion logits are modeled as:

        `C_{x,a} = B_a + g_{x,a} * R_{x,a}`,

        where `g_{x,a} in [0, gate_max]` is one of:

        - a single learned global scalar (`gate_mode="global"`),
        - a learned annotator-specific gate (`gate_mode="annotator"`), or
        - a learned sample-annotator pair gate (`gate_mode="pair"`).

        This lets the model keep instance-dependent confusion effects
        conservative, while still adapting confusion structure when data
        supports it.

        Parameters
        ----------
        clf_module : nn.Module or nn.Module.__class__
            A PyTorch module used as backbone classifier. The `forward` method
            must return class logits as first output and may optionally return
            sample embeddings as second output. If no embeddings are returned,
            the raw input representation is used by downstream components.
        alpha : float, default=0.5
            MixUp concentration parameter. The mixing coefficient is sampled
            from `Beta(alpha, alpha)`. Set `alpha=0` to disable MixUp.
        sample_embed_dim : int, default=0
            Dimensionality of the optional sample embedding used by the
            annotator confusion model. If `sample_embed_dim <= 0`,
            sample-dependent residuals are disabled and only
            annotator-dependent confusion is modeled.
        annotator_embed_dim : int, default=16
            Dimensionality of the annotator embedding.
        eta : float, default=0.9
            Prior annotator performance used to initialize diagonal confusion
            bias. Must be in `(1 / n_classes, 1)`.
        residual_rank : int, default=8
            Rank of the low-rank residual interaction between sample and
            annotator embeddings. Set to `0` to disable residual modeling.
            Ignored if `sample_embed_dim <= 0`.
        gate_mode : {"global", "annotator", "pair"}, default="pair"
            Gate parametrization:

            - `"global"`: one shared scalar gate.
            - `"annotator"`: one gate value per annotator, independent of
              sample features.
            - `"pair"`: one gate value per sample-annotator pair.
        gate_max : float, default=0.5
            Upper bound for residual scaling coefficient.
        gate_hidden_dim : int, default=16
            Hidden dimension of the gate network used when
            `gate_mode in {"annotator", "pair"}`.
        gate_bias_init : float, default=-2.0
            Initial bias for the last gate layer. Lower values initialize the
            gate closer to zero residual influence.
        gamma_init : float, default=0.1
            Initial global residual scale when `gate_mode="global"`. Must lie
            in `(0, gate_max)`. Ignored if `gate_mode != "global"` or if
            residual modeling is disabled.
        residual_dropout : float, default=0.0
            Dropout probability applied to residual interaction features.
        center_residual_rows : bool, default=True
            If `True`, row-centers residual confusion logits before combining
            with base confusion.
        detach_sample_embed : bool, default=True
            If `True`, detaches sample embeddings before feeding them into the
            annotator confusion residual path. This prevents residual confusion
            gradients from flowing back into the backbone sample
            representation. Set to `False` to co-train sample representations
            with residual confusion modeling.
        n_annotators : int, default=None
            Number of annotators. If `None`, inferred during fitting from the
            shape of `y`.
        neural_net_param_dict : dict, default=None
            Additional parameters passed to `skorch.net.NeuralNet`. Keys that
            override internal essentials (`module`, `criterion`,
            `predict_nonlinearity`, `train_split`) are not allowed.
        sample_dtype : str or type, default=np.float32
            Dtype used to cast input samples internally. If `None`, input dtype
            is preserved.
        classes : array-like of shape (n_classes,), default=None
            Class labels. If `None`, classes are inferred during `fit`.
        cost_matrix : array-like of shape (n_classes, n_classes), default=None
            Misclassification cost matrix, where `cost_matrix[i, j]` is the
            cost of predicting class `j` for true class `i`.
        missing_label : scalar or string or np.nan or None, default=np.nan
            Marker used for missing annotations. Internally, training labels are
            expected to use the encoded missing-label convention of the
            multi-annotator base classes (typically `-1`).
        random_state : int or RandomState instance or None, default=None
            Controls the randomization used during prediction.
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
            alpha=0.5,
            sample_embed_dim=0,
            annotator_embed_dim=16,
            eta=0.9,
            residual_rank=8,
            gate_mode="pair",
            gate_max=0.5,
            gate_hidden_dim=16,
            gate_bias_init=-2.0,
            gamma_init=0.1,
            residual_dropout=0.0,
            center_residual_rows=True,
            detach_sample_embed=True,
            n_annotators=None,
            neural_net_param_dict=None,
            sample_dtype=np.float32,
            classes=None,
            cost_matrix=None,
            missing_label=MISSING_LABEL,
            random_state=None,
        ):
            super().__init__(
                multi_annotator_module=_GatingModule,
                clf_module=clf_module,
                criterion=KLDivLoss,
                classes=classes,
                missing_label=missing_label,
                cost_matrix=cost_matrix,
                random_state=random_state,
                neural_net_param_dict=neural_net_param_dict,
                sample_dtype=sample_dtype,
            )
            self.clf_module = clf_module
            self.alpha = alpha
            self.sample_embed_dim = sample_embed_dim
            self.annotator_embed_dim = annotator_embed_dim
            self.eta = eta
            self.residual_rank = residual_rank
            self.gate_mode = gate_mode
            self.gate_max = gate_max
            self.gate_hidden_dim = gate_hidden_dim
            self.gate_bias_init = gate_bias_init
            self.gamma_init = gamma_init
            self.residual_dropout = residual_dropout
            self.center_residual_rows = center_residual_rows
            self.detach_sample_embed = detach_sample_embed
            self.n_annotators = n_annotators

        def predict(self, X, extra_outputs=None):
            """Return class predictions for the input samples.

            Parameters
            ----------
            X : array-like of shape (n_samples, ...)
                Input samples.
            extra_outputs : None or str or sequence of str, default=None
                Optional additional outputs to return next to predicted class
                labels. Allowed names are:
                `"logits"`, `"embeddings"`, `"annotator_perf"`,
                `"annotator_class"`, and `"annotator_embeddings"`.

            Returns
            -------
            y_pred : np.ndarray of shape (n_samples,)
                Predicted class labels.
            *extras : np.ndarray, optional
                Returned only if `extra_outputs` is provided. Extra outputs are
                returned in the same order as requested:

                - `logits` with shape `(n_samples, n_classes)`,
                - `embeddings` with shape `(n_samples, ...)`,
                - `annotator_perf` with shape `(n_samples, n_annotators)`,
                - `annotator_class` with shape
                  `(n_samples, n_annotators, n_classes)`,
                - `annotator_embeddings` with shape
                  `(n_annotators, annotator_embed_dim)`.
            """
            return SkactivemlClassifier.predict(
                self,
                X=X,
                extra_outputs=extra_outputs,
            )

        def predict_proba(self, X, extra_outputs=None):
            """Return class-membership probabilities for input samples.

            Parameters
            ----------
            X : array-like of shape (n_samples, ...)
                Input samples.
            extra_outputs : None or str or sequence of str, default=None
                Optional additional outputs to return next to class
                probabilities. Allowed names are:
                `"logits"`, `"embeddings"`, `"annotator_perf"`,
                `"annotator_class"`, and `"annotator_embeddings"`.

            Returns
            -------
            P : np.ndarray of shape (n_samples, n_classes)
                Predicted class probabilities ordered according to
                `self.classes_`.
            *extras : np.ndarray, optional
                Returned only if `extra_outputs` is provided. Extra outputs are
                returned in the same order as requested:

                - `logits` with shape `(n_samples, n_classes)`,
                - `embeddings` with shape `(n_samples, ...)`,
                - `annotator_perf` with shape `(n_samples, n_annotators)`,
                - `annotator_class` with shape
                  `(n_samples, n_annotators, n_classes)`,
                - `annotator_embeddings` with shape
                  `(n_annotators, annotator_embed_dim)`.
            """
            self._validate_data_kwargs()
            X = check_array(X, **self.check_X_dict_)
            check_n_features(self, X, reset=not hasattr(self, "n_features_in_"))
            extra_outputs = self._normalize_extra_outputs(
                extra_outputs=extra_outputs,
                allowed_names=self._ALLOWED_EXTRA_OUTPUTS,
            )

            if not hasattr(self, "neural_net_"):
                self.initialize()

            net = self.neural_net_.module_
            old_forward_return = net.forward_return
            forward_outputs = {"probas": (0, nn.Softmax(dim=-1))}
            forward_returns = ["logits_class"]
            out_idx = 1

            if "logits" in extra_outputs:
                forward_outputs["logits"] = (0, None)

            if "embeddings" in extra_outputs:
                forward_outputs["embeddings"] = (out_idx, None)
                forward_returns.append("x_embed")
                out_idx += 1

            if "annotator_perf" in extra_outputs:

                def _transform_annotator_perf(p_perf):
                    p_perf = p_perf.exp()
                    return p_perf.reshape(-1, self.n_annotators_)

                forward_outputs["annotator_perf"] = (
                    out_idx,
                    _transform_annotator_perf,
                )
                forward_returns.append("log_p_annotator_perf")
                out_idx += 1

            if "annotator_class" in extra_outputs:

                def _transform_annotator_class(p_annot):
                    p_annot = p_annot.exp()
                    return p_annot.reshape(
                        -1,
                        self.n_annotators_,
                        len(self.classes_),
                    )

                forward_outputs["annotator_class"] = (
                    out_idx,
                    _transform_annotator_class,
                )
                forward_returns.append("log_p_annotator_class")
                out_idx += 1

            if "annotator_embeddings" in extra_outputs:

                def _transform_annotator_embeddings(a_embed):
                    return a_embed[: self.n_annotators_]

                forward_outputs["annotator_embeddings"] = (
                    out_idx,
                    _transform_annotator_embeddings,
                )
                forward_returns.append("a_embed")

            try:
                net.set_forward_return(forward_returns)
                fw_out = self._forward_with_named_outputs(
                    X=X,
                    forward_outputs=forward_outputs,
                    extra_outputs=extra_outputs,
                )
            finally:
                net.set_forward_return(old_forward_return)

            self._initialize_fallbacks(fw_out[0] if isinstance(fw_out, tuple) else fw_out)
            return fw_out

        def _build_neural_net_param_overrides(self, X, y):
            # Validate core scalar parameters.
            check_scalar(
                self.alpha,
                name="alpha",
                target_type=float,
                min_val=0.0,
                min_inclusive=True,
            )
            check_scalar(
                self.sample_embed_dim,
                name="sample_embed_dim",
                target_type=int,
                min_val=0,
                min_inclusive=True,
            )
            check_scalar(
                self.annotator_embed_dim,
                name="annotator_embed_dim",
                target_type=int,
                min_val=1,
                min_inclusive=True,
            )
            check_scalar(
                self.eta,
                name="eta",
                target_type=float,
                min_val=1 / len(self.classes_),
                min_inclusive=False,
                max_val=1.0,
                max_inclusive=False,
            )
            check_scalar(
                self.residual_rank,
                name="residual_rank",
                target_type=int,
                min_val=0,
                min_inclusive=True,
            )

            if not isinstance(self.gate_mode, str):
                raise TypeError("`gate_mode` must be str.")
            gate_mode = self.gate_mode.lower()
            if gate_mode not in {"global", "annotator", "pair"}:
                raise ValueError(
                    "`gate_mode` must be one of {'global', 'annotator', 'pair'}."
                )
            if not isinstance(self.center_residual_rows, bool):
                raise TypeError("`center_residual_rows` must be bool.")
            if not isinstance(self.detach_sample_embed, bool):
                raise TypeError("`detach_sample_embed` must be bool.")

            residual_active = self.sample_embed_dim > 0 and self.residual_rank > 0
            if self.sample_embed_dim <= 0 and self.residual_rank > 0:
                warnings.warn(
                    "`residual_rank` is ignored because `sample_embed_dim <= 0`.",
                    stacklevel=2,
                )
            if self.sample_embed_dim <= 0 and gate_mode != "global":
                warnings.warn(
                    "`gate_mode` is ignored because `sample_embed_dim <= 0`.",
                    stacklevel=2,
                )
            if self.residual_rank <= 0 and gate_mode != "global":
                warnings.warn(
                    "`gate_mode` is ignored because `residual_rank <= 0`.",
                    stacklevel=2,
                )

            if residual_active:
                check_scalar(
                    self.gate_max,
                    name="gate_max",
                    target_type=float,
                    min_val=0.0,
                    min_inclusive=False,
                )
                check_scalar(
                    self.residual_dropout,
                    name="residual_dropout",
                    target_type=float,
                    min_val=0.0,
                    min_inclusive=True,
                    max_val=1.0,
                    max_inclusive=False,
                )

                if gate_mode in {"annotator", "pair"}:
                    check_scalar(
                        self.gate_hidden_dim,
                        name="gate_hidden_dim",
                        target_type=int,
                        min_val=1,
                        min_inclusive=True,
                    )
                    check_scalar(
                        self.gate_bias_init,
                        name="gate_bias_init",
                        target_type=float,
                    )
                elif gate_mode == "global":
                    if self.gate_max <= 2e-6:
                        raise ValueError(
                            "`gate_max` must be > 2e-6 when "
                            "`gate_mode='global'` and residual modeling is "
                            "enabled."
                        )
                    check_scalar(
                        self.gamma_init,
                        name="gamma_init",
                        target_type=float,
                        min_val=0.0,
                        min_inclusive=False,
                        max_val=self.gate_max,
                        max_inclusive=False,
                    )

            collate_fn = _MixUpCollate(
                n_classes=len(self.classes_),
                n_annotators=self.n_annotators_,
                alpha=self.alpha,
                missing_label=-1,
            )
            return {
                "criterion__reduction": "batchmean",
                "module__n_classes": len(self.classes_),
                "module__n_annotators": self.n_annotators_,
                "module__sample_embed_dim": self.sample_embed_dim,
                "module__annotator_embed_dim": self.annotator_embed_dim,
                "module__eta": self.eta,
                "module__residual_rank": self.residual_rank,
                "module__gate_mode": gate_mode,
                "module__gate_max": self.gate_max,
                "module__gate_hidden_dim": self.gate_hidden_dim,
                "module__gate_bias_init": self.gate_bias_init,
                "module__gamma_init": self.gamma_init,
                "module__residual_dropout": self.residual_dropout,
                "module__center_residual_rows": self.center_residual_rows,
                "module__detach_sample_embed": self.detach_sample_embed,
                "iterator_train__collate_fn": collate_fn,
            }

    class _GatingModule(_MultiAnnotatorClassificationModule):
        OUTPUTS = (
            "logits_class",
            "x_embed",
            "a_embed",
            "log_p_annotator_class",
            "log_p_annotator_perf",
            "gate",
        )

        def __init__(
            self,
            n_classes,
            n_annotators,
            clf_module,
            clf_module_param_dict,
            sample_embed_dim,
            annotator_embed_dim,
            eta,
            residual_rank=8,
            gate_mode="pair",
            gate_max=0.5,
            gate_hidden_dim=16,
            gate_bias_init=-2.0,
            gamma_init=0.1,
            residual_dropout=0.0,
            center_residual_rows=True,
            detach_sample_embed=True,
        ):
            super().__init__(
                clf_module=clf_module,
                clf_module_param_dict=clf_module_param_dict,
                default_forward_outputs="log_p_annotator_class",
                full_forward_outputs=list(self.OUTPUTS),
            )

            self.n_classes = int(n_classes)
            self.n_annotators = int(n_annotators)
            self.annotator_embed_dim = int(annotator_embed_dim)

            self.sample_embed_dim = int(sample_embed_dim)
            self.residual_rank = int(residual_rank)
            self.gate_mode = str(gate_mode).lower()
            if self.gate_mode not in {"global", "annotator", "pair"}:
                raise ValueError(
                    "`gate_mode` must be one of {'global', 'annotator', 'pair'}."
                )
            self.gate_max = float(gate_max)
            self.residual_dropout = float(residual_dropout)
            self.center_residual_rows = bool(center_residual_rows)
            self.detach_sample_embed = bool(detach_sample_embed)

            if self.sample_embed_dim <= 0:
                self.residual_rank = 0
                self.gate_mode = "global"

            self.register_buffer("a", torch.eye(n_annotators, dtype=torch.float32))

            self.sample_embed = None
            if self.sample_embed_dim > 0:
                self.sample_embed = nn.Sequential(
                    nn.LazyLinear(out_features=self.sample_embed_dim),
                    nn.SiLU(),
                )

            self.annotator_embed = nn.Linear(
                in_features=n_annotators,
                out_features=self.annotator_embed_dim,
            )

            eta_logit = math.log(eta / (1.0 - eta)) + math.log(self.n_classes - 1.0)
            prior_bias = eta_logit * torch.eye(
                self.n_classes,
                dtype=torch.float32,
            ).flatten()

            self.base_head = nn.Linear(
                self.annotator_embed_dim,
                self.n_classes * self.n_classes,
            )
            with torch.no_grad():
                self.base_head.bias.copy_(prior_bias)

            self.res_s = None
            self.res_a = None
            self.res_out = None
            self.res_norm = None

            if self.residual_rank > 0:
                self.res_s = nn.Linear(self.sample_embed_dim, self.residual_rank)
                self.res_a = nn.Linear(self.annotator_embed_dim, self.residual_rank)
                self.res_norm = nn.LayerNorm(self.residual_rank)
                self.res_out = nn.Linear(
                    self.residual_rank,
                    self.n_classes * self.n_classes,
                )
                nn.init.zeros_(self.res_out.weight)
                nn.init.zeros_(self.res_out.bias)

            self.gate = None
            self.gate_annot = None
            self.gamma_logit = None
            if self.residual_rank > 0:
                if self.gate_mode == "pair":
                    gate_in_dim = (
                        self.sample_embed_dim
                        + self.annotator_embed_dim
                        + self.residual_rank
                    )
                    self.gate = nn.Sequential(
                        nn.Linear(gate_in_dim, int(gate_hidden_dim)),
                        nn.SiLU(),
                        nn.Linear(int(gate_hidden_dim), 1),
                    )
                    nn.init.constant_(self.gate[-1].bias, float(gate_bias_init))
                elif self.gate_mode == "annotator":
                    self.gate_annot = nn.Sequential(
                        nn.Linear(self.annotator_embed_dim, int(gate_hidden_dim)),
                        nn.SiLU(),
                        nn.Linear(int(gate_hidden_dim), 1),
                    )
                    nn.init.constant_(
                        self.gate_annot[-1].bias, float(gate_bias_init)
                    )
                else:
                    eps = 1e-6
                    if self.gate_max <= 2 * eps:
                        raise ValueError(
                            "`gate_max` must be > 2e-6 when "
                            "`gate_mode='global'` and residual modeling is "
                            "enabled."
                        )

                    # Parameterize gamma in (0, gate_max) via a stable
                    # fractional logit to avoid log-domain errors.
                    gamma_frac = float(gamma_init) / self.gate_max
                    gamma_frac = max(eps, min(gamma_frac, 1.0 - eps))
                    gamma_logit = math.log(gamma_frac / (1.0 - gamma_frac))
                    self.gamma_logit = nn.Parameter(
                        torch.tensor(gamma_logit, dtype=torch.float32)
                    )

        def _resolve_annotator_input(self, a, device):
            if a is None:
                return self.a.to(device)

            if a.dim() == 0:
                a = a.unsqueeze(0)

            if a.dtype in (torch.int8, torch.int16, torch.int32, torch.int64):
                a = F.one_hot(
                    a.to(device=device, dtype=torch.long),
                    num_classes=self.n_annotators,
                ).float()
            else:
                a = a.to(device=device, dtype=torch.float32)

            if a.dim() == 1:
                a = a.unsqueeze(0)
            if a.shape[-1] != self.n_annotators:
                raise ValueError(
                    f"`a` must have last dimension {self.n_annotators}, got "
                    f"{tuple(a.shape)}."
                )
            return a

        def forward(self, x, a=None):
            """Forward pass of the gated confusion model.

            Parameters
            ----------
            x : torch.Tensor of shape (n_samples, ...)
                Input samples.
            a : torch.Tensor or None, default=None
                Annotator specification:

                - `None`: use all annotators (identity one-hot matrix).
                - integer tensor: annotator ids, one-hot encoded internally.
                - float tensor: annotator feature vectors / one-hot vectors.

            Returns
            -------
            out : torch.Tensor or tuple
                Outputs requested via `set_forward_return`.

            Notes
            -----
            In evaluation mode (`self.training == False`):

            - if `a is None`, all sample-annotator combinations are built.
            - if `a` is provided and `len(a) == n_samples`, inputs are treated
              as sample-aligned `(x_i, a_i)` pairs.
            - otherwise, all combinations of samples and provided annotators are
              built.
            """
            logits_class, x_embed = self.clf_module_forward(x)

            out = []
            if "logits_class" in self.forward_return:
                out.append(logits_class)
            if "x_embed" in self.forward_return:
                out.append(x_embed)

            need_annotator_output = any(
                k in self.forward_return
                for k in (
                    "a_embed",
                    "log_p_annotator_class",
                    "log_p_annotator_perf",
                    "gate",
                )
            )
            if not need_annotator_output:
                return out[0] if len(out) == 1 else tuple(out)

            a_provided = a is not None
            a = self._resolve_annotator_input(a, device=x.device)
            a_embed = self.annotator_embed(a)

            s = None
            if self.sample_embed is not None:
                x_for_residual = (
                    x_embed.detach() if self.detach_sample_embed else x_embed
                )
                s = self.sample_embed(x_for_residual.flatten(start_dim=1))

            if not self.training:
                if a_provided and a_embed.dim() >= 2 and len(a_embed) == len(x):
                    pair_x_idx = torch.arange(len(x), device=x.device)
                    pair_a_idx = torch.arange(len(a_embed), device=a_embed.device)
                else:
                    combs = torch.cartesian_prod(
                        torch.arange(len(x), device=x.device),
                        torch.arange(len(a_embed), device=a_embed.device),
                    )
                    pair_x_idx = combs[:, 0]
                    pair_a_idx = combs[:, 1]

                logits_class_pairs = logits_class[pair_x_idx]
                a_embed_return = a_embed.clone().detach()
                a_embed_pairs = a_embed[pair_a_idx]
                s_pairs = s[pair_x_idx] if s is not None else None
            else:
                logits_class_pairs = logits_class
                a_embed_return = a_embed
                a_embed_pairs = a_embed
                s_pairs = s

            base = self.base_head(a_embed_pairs).view(
                -1,
                self.n_classes,
                self.n_classes,
            )

            gate = None
            residual = None
            if self.residual_rank > 0:
                if s_pairs is None:
                    residual = torch.zeros_like(base)
                    gate = torch.zeros(
                        base.shape[0],
                        1,
                        1,
                        device=base.device,
                        dtype=base.dtype,
                    )
                else:
                    us = self.res_s(s_pairs)
                    ue = self.res_a(a_embed_pairs)
                    h = self.res_norm(us * ue)
                    h = F.dropout(
                        h,
                        p=self.residual_dropout,
                        training=self.training,
                    )

                    residual = self.res_out(h).view(
                        -1,
                        self.n_classes,
                        self.n_classes,
                    )
                    if self.center_residual_rows:
                        residual = residual - residual.mean(dim=-1, keepdim=True)

                    if self.gate_mode == "pair":
                        gate_in = torch.cat([s_pairs, a_embed_pairs, h], dim=-1)
                        gate = self.gate_max * torch.sigmoid(self.gate(gate_in))
                    elif self.gate_mode == "annotator":
                        gate = self.gate_max * torch.sigmoid(
                            self.gate_annot(a_embed_pairs)
                        )
                    else:
                        gate = self.gate_max * torch.sigmoid(self.gamma_logit).expand(
                            base.shape[0],
                            1,
                        )
                    gate = gate.unsqueeze(-1)

            p_conf_log = F.log_softmax(base, dim=-1)
            if residual is not None:
                log_p_conf_res = F.log_softmax(base + residual, dim=-1)
                log_gate = torch.log(gate.clamp_min(1e-12))
                log_one_minus_gate = torch.log1p((-gate).clamp_max(-1e-12))
                p_conf_log = torch.logsumexp(
                    torch.stack(
                        [
                            log_one_minus_gate + p_conf_log,
                            log_gate + log_p_conf_res,
                        ],
                        dim=0,
                    ),
                    dim=0,
                )
            else:
                gate = torch.zeros(
                    base.shape[0],
                    1,
                    1,
                    device=base.device,
                    dtype=base.dtype,
                )
            p_class_log = F.log_softmax(logits_class_pairs, dim=-1)

            if "log_p_annotator_perf" in self.forward_return:
                log_diag_conf = torch.diagonal(p_conf_log, dim1=-2, dim2=-1)
                log_p_perf = torch.logsumexp(p_class_log + log_diag_conf, dim=-1)
                out.append(log_p_perf)

            if "log_p_annotator_class" in self.forward_return:
                log_p_annotator_class = torch.logsumexp(
                    p_class_log[:, :, None] + p_conf_log,
                    dim=1,
                )
                out.append(log_p_annotator_class)

            if "gate" in self.forward_return:
                out.append(gate.squeeze(-1).squeeze(-1))

            if "a_embed" in self.forward_return:
                out.append(a_embed_return)

            return out[0] if len(out) == 1 else tuple(out)

except ImportError:  # pragma: no cover
    pass
