from __future__ import annotations

import os
import time

import numpy as np

from scipy.special import digamma
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.utils import check_random_state

from skactiveml.utils import is_labeled
from ._base import PairScorer
from ._utils import (
    _channel_confusion_from_theta_g_batch,
    expected_score_gain,
)


def _l2_normalize(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(n, eps)


class KernelSmoothedBayesianAnnotatorGain(PairScorer):
    """
    Pair scorer using expected uncertainty reduction under a
    kernel-smoothed annotator model.

    For a candidate pair (x, a), this scorer supports multiple channel
    variants:

    - "channel":
        Z ~ r(·) = p(Z|x)                 from clf
        theta ~ Beta(alpha(x,a), beta(x,a))
        g ~ Dirichlet(gamma(x,a))
        p(Y=Z | Z) = theta
        p(Y=y!=Z | Z) ∝ g_y
      i.e., `g` is conditioned on being incorrect.

    - "pi_mixture_channel":
        build the original "channel" confusion matrix K_orig, then mix it with
        a class-independent response component:
            K_mix[y | z] = (1-pi) K_orig[y | z] + pi g_y
        where pi is estimated from local response collapse in `g`, shrunk by
        the local effective sample size.

    - "scalar_uniform_confusion":
        estimate a single accuracy scalar theta and define a proper confusion
        matrix with uniform off-diagonal mass:
            C[z,z] = theta
            C[z,y!=z] = (1-theta)/(K-1)

    - "diag_uniform_confusion":
        estimate per-class diagonal accuracies theta_z and define rows
            C[z,z] = theta_z
            C[z,y!=z] = (1-theta_z)/(K-1)

    - "full_confusion":
        estimate the full row-stochastic confusion matrix C[z,y].

    Kernel-smoothed posterior parameters are built from observed annotations
    (x_i, a_i, y_i):

    - Beta correctness model (soft counts using configurable correctness
      evidence)
        m_i = p_obs(y_i | x_i)                                    if
              correctness_mode="classifier"
        m_i = p_clf(worker a_i correct | x_i)                     if
              correctness_mode="annotator_perf"
        m_i = p(Y=y_i | x_i, worker a_i, observed label y_i)      if
              correctness_mode="annotator_perf_posterior"
        m_i = p(Y=y_i | x_i, worker a_i, observed label y_i)      if
              correctness_mode="confusion_posterior"
        s(x,a) = sum_i w_i(x,a) * m_i
        f(x,a) = sum_i w_i(x,a) * (1 - m_i)
        alpha = alpha0 + s,  beta = beta0 + f
      (ESS-scaled when `use_ess_beta=True`)

    - Dirichlet label model (kernel-weighted label counts)
        gamma_k(x,a) = gamma0_k + sum_i w_i(x,a) * 1[y_i = k]
      (ESS-scaled when `use_ess_label_dirichlet=True`)

    where the pair weight factorizes as:
        w_i(x,a) = k_x(x_i, x) * k_a(a_i, a)

    `k_a` uses annotator embeddings when available. If annotator embeddings are
    not available (or are not global), the scorer falls back to exact annotator
    identity weighting: k_a(a_i, a) = 1[a_i = a].

    Utility is an expected reduction in predictive uncertainty, evaluated via
    Monte Carlo samples from Beta/Dirichlet (or using posterior means). The
    uncertainty functional is controlled by ``gain_type``.

    Parameters
    ----------
    accuracy_mean : float or {"global_observed", "per_annotator_observed"}, \
            default="global_observed"
        Accuracy-prior specification. A float uses a fixed prior mean. The
        observed modes infer the prior mean from the average soft correctness:
        - "global_observed": use the average soft correctness across all
          observed annotations.
        - "per_annotator_observed": use the average soft correctness over all
          instances labeled by the respective annotator; if an annotator has no
          observations, fallback to the global observed average.
    accuracy_strength : float, default=10.0
        Prior strength for accuracy parameters. Used for Beta priors in
        variants with scalar/diagonal accuracies and as the total concentration
        per confusion row in `channel_variant="full_confusion"`.
    gamma_x : float or {"median","mean","minimum"}, default="median"
        Bandwidth selection for the sample-embedding RBF kernel.
    gamma_x_scope : {"global","per_annotator"}, default="global"
        Scope used to resolve `gamma_x`:
        - "global": estimate a single bandwidth from all observed sample
          embeddings.
        - "per_annotator": estimate bandwidth separately per target annotator
          from that annotator's observed sample embeddings (fallback to global
          if <2 points).
    gamma_a : float or {"median","mean","minimum"}, default="median"
        Bandwidth selection for the annotator-embedding RBF kernel (if used).
    use_annotator_embeddings : bool, default=True
        If True, request and use global annotator embeddings (when provided by
        `clf`) to smooth across annotators. If False, always use exact
        annotator identity weighting.
    annotator_lambda : float, default=0.0
        Convex weight on an instance-independent sample kernel in the
        annotator-side posterior updates. `0.0` keeps the local kernel only,
        while `1.0` yields instance-independent annotator estimates.
    channel_label_dirichlet_strength : float, default=1.0
        Symmetric Dirichlet prior concentration for the fallback label
        distribution `g` in `channel_variant` values that use a local response
        distribution (`"channel"` and `"pi_mixture_channel"`).
    gain_type : {"entropy", "margin", "brier", "confidence"}, default="entropy"
        Uncertainty functional reduced in expectation after observing an
        annotator label. ``"entropy"`` recovers standard information gain.
        ``"confidence"`` is the expected increase in maximum posterior
        class probability.
    entropy_response_cap : bool, default=False
        If True and `gain_type="entropy"`, cap each entropy-gain draw by the
        upper bound ``min(H(class), H(response))``. For `channel`, response
        entropy is computed from the raw sampled label distribution `g`, not
        from the misspecified conditional channel matrix. For confusion
        variants, response entropy is computed from the model-implied response
        distribution. Ignored for non-entropy gains.
    response_entropy_cap : bool, default=False
        If True and `gain_type="entropy"`, additionally cap channel gains by
        ``response_entropy_cap_lambda * H(g)``. Unlike
        `entropy_response_cap`, this is a response-collapse regularizer, not a
        mutual-information bound.
    response_entropy_cap_lambda : float, default=1.0
        Multiplier for the optional ``H(g)`` response-collapse cap.
    channel_variant : {"channel", "pi_mixture_channel", /
            "scalar_uniform_confusion", "diag_uniform_confusion", /
            "full_confusion"}, default="channel"
        Annotator noise parameterization used for gain computation.
    full_confusion_prior_source : {"accuracy", "channel"}, default="accuracy"
        Prior used for `channel_variant="full_confusion"`:
        - "accuracy": use the existing accuracy-prior mean with uniform
          off-diagonal mass.
        - "channel": use the local channel posterior mean as a
          candidate-specific full-confusion Dirichlet prior mean.
    full_confusion_channel_prior_strength : float, default=1.0
        Dirichlet row concentration used after converting the local channel
        posterior mean into a full-confusion prior. Used only when
        `channel_variant="full_confusion"` and
        `full_confusion_prior_source="channel"`.
    correctness_mode : {"classifier", "annotator_perf", /
            "annotator_perf_posterior", "confusion_posterior", "auto"}, \
            default="classifier"
        Source of observed-annotation correctness evidence used in the
        kernel-smoothed posteriors:
        - "classifier": use ``p_clf(y_i | x_i)``.
        - "annotator_perf": use classifier-provided annotator correctness
          probabilities for the observed annotator.
        - "annotator_perf_posterior": build a synthetic confusion matrix from
          classifier-provided annotator correctness probabilities with uniform
          off-diagonal error mass, then compute posterior latent-class
          responsibilities conditioned on the observed label.
        - "confusion_posterior": use classifier-provided confusion matrices to
          compute posterior latent-class responsibilities conditioned on the
          observed label.
        - "auto": prefer confusion matrices, else annotator correctness
          probabilities, else fallback to ``p_clf(y_i | x_i)``.
    observed_class_prior : {"classifier", "kernel"}, default="classifier"
        Class posterior source used for observed-annotation correctness
        evidence and latent-class responsibilities:
        - "classifier": use direct classifier posteriors ``p_clf(Y|x_i)``.
        - "kernel": use kernel-smoothed classifier posteriors at the observed
          sample embeddings, with the same kernel hyperparameters as
          ``class_prior="kernel"``.
    observed_class_prior_leave_one_out : bool, default=False
        If True and ``observed_class_prior="kernel"``, remove the queried
        observed sample from the kernel smoother used to compute its observed
        class prior. This avoids using an annotation's own instance posterior
        as local evidence for its correctness target.
    class_prior : {"classifier", "uniform", "kernel"}, default="classifier"
        Prior used for the latent class in the IG computation:
        - "classifier": use the classifier posterior ``p(Y|x)``.
        - "uniform": use a uniform prior over classes.
        - "kernel": use a kernel-smoothed Dirichlet prior built from the
          classifier posteriors on labeled samples.
    class_prior_strength : float, default=1.0
        Symmetric Dirichlet prior concentration for the kernelized class prior.
        Used only when `class_prior="kernel"`.
    class_prior_lambda : float, default=0.0
        Convex weight on an instance-independent sample kernel in the
        kernelized class prior. `0.0` keeps the local kernel only, while
        `1.0` collapses the class prior to a global sample smoother.
    use_ess_class_prior : bool, default=False
        If True, map kernel-weighted class evidence to the class-prior
        Dirichlet posterior using ESS-based concentration instead of raw
        weighted counts.
    tau_class_prior : float, default=1.0
        Discount factor for ESS-based class-prior Dirichlet concentration
        (only used if `use_ess_class_prior=True`).
    sample_class_prior : bool, default=False
        If True and `class_prior="kernel"` together with
        `n_mc_samples > 0`, sample the class prior from the kernelized
        Dirichlet; otherwise use its posterior mean.
    use_ess_beta : bool, default=False
        If True, map the kernel-weighted correctness evidence to a Beta 
        posterior using ESS-based concentration instead of raw weighted counts.
    tau_beta : float, default=1.0
        Discount factor for ESS-based Beta concentration
        (only used if `use_ess_beta=True`).
    use_ess_label_dirichlet : bool, default=False
        If True, map kernel-weighted label evidence to a Dirichlet posterior
        using ESS-based concentration instead of raw weighted counts.
    tau_label_dirichlet : float, default=1.0
        Discount factor for ESS-based Dirichlet concentration
        (only used if `use_ess_label_dirichlet=True`).
    top_m : int or None, default=None
        If not None, approximate entropy gain in top-M + "other" reduced label
        space. Currently supported only for `gain_type="entropy"` with
        `channel_variant="channel"` together with
        `class_prior` in `{"classifier", "kernel"}`.
    n_mc_samples : int, default=1
        Shared number of Monte Carlo draws for all sampled latent variables:
        Beta accuracies, Dirichlet label parameters, and the kernelized class
        prior. If <=0, deterministic point estimates / posterior means are
        used instead.
    gain_ucb_quantile : float or None, default=None
        Empirical optimistic quantile for Monte Carlo gain samples when
        `n_mc_samples > 0`. If provided, the scorer returns the empirical
        gain quantile `q` across latent-variable draws instead of the
        posterior-mean gain.
    theta_ucb_quantile : float or None, default=None
        Deprecated alias for `gain_ucb_quantile`. If provided, it is mapped
        to `gain_ucb_quantile`.
    sample_label_dirichlet : bool, default=False
        If True, sample Dirichlet-distributed label parameters
        (`g` in `channel`, confusion rows in `full_confusion`) using
        `n_mc_samples` draws; otherwise use posterior means.
    pi_mixture_kappa : float, default=5.0
        ESS shrinkage constant for `channel_variant="pi_mixture_channel"`.
        Larger values require more local evidence before response collapse is
        converted into a large class-independent mixture weight.
    pi_mixture_gamma : float, default=2.0
        Exponent applied to the normalized response-collapse score for
        `channel_variant="pi_mixture_channel"`.
    pi_mixture_max : float, default=1.0
        Maximum class-independent mixture probability for
        `channel_variant="pi_mixture_channel"`.
    full_confusion_reduce_top_m : int or None, default=None
        If not None, approximate full-confusion entropy gain in a reduced
        label space consisting of the candidate-specific top-M latent classes
        plus one aggregated "other" class. This preserves response mass for
        low-probability labels in the "other" output column, which is important
        for single-class spammer behavior. Supported only for
        `channel_variant="full_confusion"` and `gain_type="entropy"`.
    confusion_parameter_gain_weight : float, default=0.0
        Additive exploration bonus for `channel_variant="full_confusion"`.
        The bonus is the expected one-step information gain about the
        annotator's row-wise Dirichlet confusion model, weighted by the
        candidate class posterior. A value of ``0.0`` disables the bonus.
    gain_batch_size : int or None, default=None
        Optional candidate chunk size used while evaluating full-confusion
        gains. ``None`` keeps the fully vectorized behavior.
    store_utility_intervals : bool, default=False
        If True, store pair-level lower/upper utility bounds from raw
        full-confusion Monte Carlo gain draws in ``last_utility_lcb_`` and
        ``last_utility_ucb_``. Currently populated only for
        ``channel_variant="full_confusion"`` with ``n_mc_samples > 0``.
    utility_interval_lower, utility_interval_upper : float, default=(0.0, 1.0)
        Quantiles used for stored utility bounds. The default stores min/max
        over the available Monte Carlo draws.
    profile_timing : bool, default=False
        If True, print lightweight scorer timing information. The same
        profiling output can be enabled with ``KS_BAG_PROFILE=1``.
    channel_wrong_label_mode : {"normalize", "sample_dirichlet_wrong"}, /
            default="normalize"
        Wrong-label construction for `channel_variant="channel"`:
        - "normalize": use shared `g` and condition on being wrong by removing
          the assumed true class and renormalizing.
        - "sample_dirichlet_wrong": for each assumed true class z, draw
          (or use mean of) a Dirichlet over wrong labels only,
          `Dir(gamma_{-z})`.
    random_state : None or int, default=None
        Seed for reproducibility.
    """

    def __init__(
        self,
        *,
        accuracy_mean: float | str = "global_observed",
        accuracy_strength: float = 10.0,
        gamma_x="median",
        gamma_x_scope: str = "global",
        gamma_a="median",
        use_annotator_embeddings: bool = True,
        annotator_lambda: float = 0.0,
        channel_label_dirichlet_strength: float = 1.0,
        gain_type: str = "entropy",
        entropy_response_cap: bool = False,
        response_entropy_cap: bool = False,
        response_entropy_cap_lambda: float = 1.0,
        channel_variant: str = "channel",
        full_confusion_prior_source: str = "accuracy",
        full_confusion_channel_prior_strength: float = 1.0,
        correctness_mode: str = "classifier",
        observed_class_prior: str = "classifier",
        observed_class_prior_leave_one_out: bool = False,
        class_prior: str = "classifier",
        class_prior_strength: float = 1.0,
        class_prior_lambda: float = 0.0,
        use_ess_class_prior: bool = False,
        tau_class_prior: float = 1.0,
        sample_class_prior: bool = False,
        use_ess_beta: bool = False,
        tau_beta: float = 1.0,
        use_ess_label_dirichlet: bool = False,
        tau_label_dirichlet: float = 1.0,
        top_m: int | None = None,
        n_mc_samples: int = 1,
        gain_ucb_quantile: float | None = None,
        theta_ucb_quantile: float | None = None,
        sample_label_dirichlet: bool = False,
        pi_mixture_kappa: float = 5.0,
        pi_mixture_gamma: float = 2.0,
        pi_mixture_max: float = 1.0,
        full_confusion_reduce_top_m: int | None = None,
        confusion_parameter_gain_weight: float = 0.0,
        gain_batch_size: int | None = None,
        store_utility_intervals: bool = False,
        utility_interval_lower: float = 0.0,
        utility_interval_upper: float = 1.0,
        profile_timing: bool = True,
        channel_wrong_label_mode: str = "normalize",
        random_state=None,
    ):
        if isinstance(accuracy_mean, str):
            self.accuracy_mean = str(accuracy_mean)
            self._accuracy_mean_mode = self.accuracy_mean
        else:
            self.accuracy_mean = float(accuracy_mean)
            self._accuracy_mean_mode = "fixed"
        self.accuracy_strength = float(accuracy_strength)
        self.gamma_x = gamma_x
        self.gamma_x_scope = str(gamma_x_scope)
        self.gamma_a = gamma_a
        self.use_annotator_embeddings = bool(use_annotator_embeddings)
        self.annotator_lambda = float(annotator_lambda)
        self.channel_label_dirichlet_strength = float(
            channel_label_dirichlet_strength
        )
        self.gain_type = str(gain_type)
        self.entropy_response_cap = bool(entropy_response_cap)
        self.response_entropy_cap = bool(response_entropy_cap)
        self.response_entropy_cap_lambda = float(response_entropy_cap_lambda)
        self.channel_variant = str(channel_variant)
        self.full_confusion_prior_source = str(full_confusion_prior_source)
        self.full_confusion_channel_prior_strength = float(
            full_confusion_channel_prior_strength
        )
        self.correctness_mode = str(correctness_mode)
        self.observed_class_prior = str(observed_class_prior)
        self.observed_class_prior_leave_one_out = bool(
            observed_class_prior_leave_one_out
        )
        self.class_prior = str(class_prior)
        self.class_prior_strength = float(class_prior_strength)
        self.class_prior_lambda = float(class_prior_lambda)
        self.use_ess_class_prior = bool(use_ess_class_prior)
        self.tau_class_prior = float(tau_class_prior)
        self.sample_class_prior = bool(sample_class_prior)
        self.use_ess_beta = bool(use_ess_beta)
        self.tau_beta = float(tau_beta)
        self.use_ess_label_dirichlet = bool(use_ess_label_dirichlet)
        self.tau_label_dirichlet = float(tau_label_dirichlet)
        self.top_m = None if top_m is None else int(top_m)
        self.n_mc_samples = int(n_mc_samples)
        if gain_ucb_quantile is not None and theta_ucb_quantile is not None:
            raise ValueError(
                "Use only one of gain_ucb_quantile or theta_ucb_quantile."
            )
        if gain_ucb_quantile is None and theta_ucb_quantile is not None:
            gain_ucb_quantile = theta_ucb_quantile
        self.gain_ucb_quantile = (
            None
            if gain_ucb_quantile is None
            else float(gain_ucb_quantile)
        )
        self.sample_label_dirichlet = bool(sample_label_dirichlet)
        self.pi_mixture_kappa = float(pi_mixture_kappa)
        self.pi_mixture_gamma = float(pi_mixture_gamma)
        self.pi_mixture_max = float(pi_mixture_max)
        self.full_confusion_reduce_top_m = (
            None
            if full_confusion_reduce_top_m is None
            else int(full_confusion_reduce_top_m)
        )
        self.confusion_parameter_gain_weight = float(
            confusion_parameter_gain_weight
        )
        self.gain_batch_size = (
            None if gain_batch_size is None else int(gain_batch_size)
        )
        self.store_utility_intervals = bool(store_utility_intervals)
        self.utility_interval_lower = float(utility_interval_lower)
        self.utility_interval_upper = float(utility_interval_upper)
        self.profile_timing = bool(profile_timing)
        self.channel_wrong_label_mode = str(channel_wrong_label_mode)
        self.random_state = check_random_state(random_state)
        self.last_utility_mean_ = None
        self.last_utility_lcb_ = None
        self.last_utility_ucb_ = None

        if self._accuracy_mean_mode == "fixed":
            if not (0.0 < self.accuracy_mean < 1.0):
                raise ValueError("accuracy_mean must be in (0, 1)")
        elif self._accuracy_mean_mode not in {
            "global_observed",
            "per_annotator_observed",
        }:
            raise ValueError(
                "accuracy_mean must be a float in (0, 1) or one of "
                "{'global_observed', 'per_annotator_observed'}"
            )
        if self.accuracy_strength <= 0:
            raise ValueError("accuracy_strength must be > 0")
        if self.gain_ucb_quantile is not None and not (
            0.0 < self.gain_ucb_quantile < 1.0
        ):
            raise ValueError("gain_ucb_quantile must be in (0, 1)")
        if self.gain_batch_size is not None and self.gain_batch_size <= 0:
            raise ValueError("gain_batch_size must be positive or None")
        if not (
            0.0
            <= self.utility_interval_lower
            <= self.utility_interval_upper
            <= 1.0
        ):
            raise ValueError(
                "Require 0 <= utility_interval_lower <= "
                "utility_interval_upper <= 1."
            )
        if self.channel_label_dirichlet_strength <= 0:
            raise ValueError("channel_label_dirichlet_strength must be > 0")
        if self.response_entropy_cap_lambda < 0:
            raise ValueError("response_entropy_cap_lambda must be >= 0")
        if self.pi_mixture_kappa <= 0:
            raise ValueError("pi_mixture_kappa must be > 0")
        if self.pi_mixture_gamma <= 0:
            raise ValueError("pi_mixture_gamma must be > 0")
        if not (0.0 <= self.pi_mixture_max <= 1.0):
            raise ValueError("pi_mixture_max must be in [0, 1]")
        if self.confusion_parameter_gain_weight < 0.0:
            raise ValueError("confusion_parameter_gain_weight must be >= 0")
        if (
            self.confusion_parameter_gain_weight > 0.0
            and self.channel_variant != "full_confusion"
        ):
            raise ValueError(
                "confusion_parameter_gain_weight requires "
                "channel_variant='full_confusion'"
            )
        if self.full_confusion_reduce_top_m is not None:
            if self.full_confusion_reduce_top_m <= 0:
                raise ValueError("full_confusion_reduce_top_m must be positive or None")
            if self.channel_variant != "full_confusion":
                raise ValueError(
                    "full_confusion_reduce_top_m requires "
                    "channel_variant='full_confusion'"
                )
            if self.gain_type != "entropy":
                raise ValueError(
                    "full_confusion_reduce_top_m requires gain_type='entropy'"
                )
        if self.gamma_x_scope not in {"global", "per_annotator"}:
            raise ValueError(
                "gamma_x_scope must be one of {'global', 'per_annotator'}"
            )
        if not (0.0 <= self.annotator_lambda <= 1.0):
            raise ValueError("annotator_lambda must be in [0, 1]")
        if not (0.0 <= self.class_prior_lambda <= 1.0):
            raise ValueError("class_prior_lambda must be in [0, 1]")
        if self.channel_wrong_label_mode not in {
            "normalize",
            "sample_dirichlet_wrong",
        }:
            raise ValueError(
                "channel_wrong_label_mode must be one of "
                "{'normalize', 'sample_dirichlet_wrong'}"
            )
        if self.channel_variant not in {
            "channel",
            "pi_mixture_channel",
            "scalar_uniform_confusion",
            "diag_uniform_confusion",
            "full_confusion",
        }:
            raise ValueError(
                "channel_variant must be one of "
                "{'channel', 'pi_mixture_channel', 'scalar_uniform_confusion', "
                "'diag_uniform_confusion', 'full_confusion'}"
            )
        if self.full_confusion_prior_source not in {"accuracy", "channel"}:
            raise ValueError(
                "full_confusion_prior_source must be one of "
                "{'accuracy', 'channel'}"
            )
        if (
            self.full_confusion_prior_source == "channel"
            and self.channel_variant != "full_confusion"
        ):
            raise ValueError(
                "full_confusion_prior_source='channel' requires "
                "channel_variant='full_confusion'"
            )
        if self.full_confusion_channel_prior_strength <= 0:
            raise ValueError("full_confusion_channel_prior_strength must be > 0")
        if (
            self.full_confusion_prior_source != "channel"
            and self.full_confusion_channel_prior_strength != 1.0
        ):
            raise ValueError(
                "full_confusion_channel_prior_strength is only used when "
                "full_confusion_prior_source='channel'"
            )
        if self.correctness_mode not in {
            "classifier",
            "annotator_perf",
            "annotator_perf_posterior",
            "confusion_posterior",
            "auto",
        }:
            raise ValueError(
                "correctness_mode must be one of "
                "{'classifier', 'annotator_perf', "
                "'annotator_perf_posterior', "
                "'confusion_posterior', 'auto'}"
            )
        if self.class_prior not in {"classifier", "uniform", "kernel"}:
            raise ValueError(
                "class_prior must be one of {'classifier', 'uniform', 'kernel'}"
            )
        if self.observed_class_prior not in {"classifier", "kernel"}:
            raise ValueError(
                "observed_class_prior must be one of {'classifier', 'kernel'}"
            )
        if (
            self.observed_class_prior_leave_one_out
            and self.observed_class_prior != "kernel"
        ):
            raise ValueError(
                "observed_class_prior_leave_one_out=True requires "
                "observed_class_prior='kernel'"
            )
        if self.gain_type not in {"entropy", "margin", "brier", "confidence"}:
            raise ValueError(
                "gain_type must be one of "
                "{'entropy', 'margin', 'brier', 'confidence'}"
            )
        if self.class_prior_strength <= 0:
            raise ValueError("class_prior_strength must be > 0")
        if self.tau_class_prior <= 0:
            raise ValueError("tau_class_prior must be > 0")
        if self.tau_beta <= 0:
            raise ValueError("tau_beta must be > 0")
        if self.tau_label_dirichlet <= 0:
            raise ValueError("tau_label_dirichlet must be > 0")
        if self.sample_class_prior and self.class_prior != "kernel":
            raise ValueError(
                "sample_class_prior=True requires class_prior='kernel'"
            )
        if self.top_m is not None:
            if self.top_m <= 0:
                raise ValueError("top_m must be positive or None")
            if self.channel_variant != "channel":
                raise ValueError(
                    "top_m is only supported with channel_variant='channel'"
                )
            if self.gain_type != "entropy":
                raise ValueError(
                    "top_m is only supported with gain_type='entropy'"
                )
            if self.class_prior not in {"classifier", "kernel"}:
                raise ValueError(
                    "top_m is only supported with class_prior in "
                    "{'classifier', 'kernel'}"
                )

        uses_channel_prior = (
            self.channel_variant == "full_confusion"
            and self.full_confusion_prior_source == "channel"
        )
        uses_beta = self.channel_variant in {
            "channel",
            "pi_mixture_channel",
            "scalar_uniform_confusion",
            "diag_uniform_confusion",
        } or uses_channel_prior
        if not uses_beta:
            if self.use_ess_beta:
                raise ValueError(
                    "use_ess_beta is only supported for channel variants "
                    "with Beta accuracy posteriors"
                )
            if self.tau_beta != 1.0:
                raise ValueError(
                    "tau_beta is only used for channel variants with Beta "
                    "accuracy posteriors"
                )

        if (
            self.gain_ucb_quantile is not None
            and self.n_mc_samples <= 0
        ):
            raise ValueError(
                "gain_ucb_quantile requires n_mc_samples > 0"
            )

        uses_label_dirichlet = self.channel_variant in {
            "channel",
            "pi_mixture_channel",
            "full_confusion",
        }
        if not uses_label_dirichlet:
            if self.use_ess_label_dirichlet:
                raise ValueError(
                    "use_ess_label_dirichlet is only supported for "
                    "channel, pi_mixture_channel, and full_confusion variants"
                )
            if self.tau_label_dirichlet != 1.0:
                raise ValueError(
                    "tau_label_dirichlet is only used for channel and "
                    "pi_mixture_channel/full_confusion variants"
                )
            if self.sample_label_dirichlet:
                raise ValueError(
                    "sample_label_dirichlet is only supported for "
                    "channel, pi_mixture_channel, and full_confusion variants"
                )

        uses_channel_style_params = self.channel_variant in {
            "channel",
            "pi_mixture_channel",
        } or uses_channel_prior
        if not uses_channel_style_params:
            if self.channel_label_dirichlet_strength != 1.0:
                raise ValueError(
                    "channel_label_dirichlet_strength is only used for "
                    "channel-style variants or full_confusion channel priors"
                )
            if self.channel_wrong_label_mode != "normalize":
                raise ValueError(
                    "channel_wrong_label_mode is only used for "
                    "channel-style variants or full_confusion channel priors"
                )
        if (
            self.channel_variant == "pi_mixture_channel"
            and self.channel_wrong_label_mode != "normalize"
        ):
            raise ValueError(
                "pi_mixture_channel currently requires "
                "channel_wrong_label_mode='normalize'"
            )
        if self.channel_variant != "pi_mixture_channel":
            if self.pi_mixture_kappa != 5.0:
                raise ValueError(
                    "pi_mixture_kappa is only used for "
                    "channel_variant='pi_mixture_channel'"
                )
            if self.pi_mixture_gamma != 2.0:
                raise ValueError(
                    "pi_mixture_gamma is only used for "
                    "channel_variant='pi_mixture_channel'"
                )
            if self.pi_mixture_max != 1.0:
                raise ValueError(
                    "pi_mixture_max is only used for "
                    "channel_variant='pi_mixture_channel'"
                )

        if self.class_prior != "kernel" and self.observed_class_prior != "kernel":
            if self.class_prior_strength != 1.0:
                raise ValueError(
                    "class_prior_strength is only used when "
                    "class_prior='kernel' or observed_class_prior='kernel'"
                )
            if self.use_ess_class_prior:
                raise ValueError(
                    "use_ess_class_prior is only supported when "
                    "class_prior='kernel' or observed_class_prior='kernel'"
                )
            if self.tau_class_prior != 1.0:
                raise ValueError(
                    "tau_class_prior is only used when class_prior='kernel' "
                    "or observed_class_prior='kernel'"
                )
            if self.class_prior_lambda != 0.0:
                raise ValueError(
                    "class_prior_lambda is only used when "
                    "class_prior='kernel' or observed_class_prior='kernel'"
                )

    @staticmethod
    def _env_flag(name: str) -> bool:
        value = os.environ.get(name, "")
        return value.lower() in {"1", "true", "yes", "on"}

    def _profile_enabled(self) -> bool:
        return self.profile_timing or self._env_flag("KS_BAG_PROFILE")

    @staticmethod
    def _profile_add(timings: dict[str, float], name: str, start: float):
        timings[name] = timings.get(name, 0.0) + (time.perf_counter() - start)

    @staticmethod
    def _profile_print(timings: dict[str, float]):
        pieces = [f"{name}={seconds:.4f}s" for name, seconds in timings.items()]
        print("[ks_bag profile] " + ", ".join(pieces))

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
        self.last_utility_mean_ = None
        self.last_utility_lcb_ = None
        self.last_utility_ucb_ = None
        profile = self._profile_enabled()
        timings: dict[str, float] = {}
        t_total = time.perf_counter()

        classes = clf.classes_
        K = len(classes)
        if K < 2:
            raise ValueError("IG requires at least 2 classes.")
        valid_variants = {
            "channel",
            "pi_mixture_channel",
            "scalar_uniform_confusion",
            "diag_uniform_confusion",
            "full_confusion",
        }
        if self.channel_variant not in valid_variants:
            raise ValueError(
                f"Unknown channel_variant={self.channel_variant!r}. "
                f"Expected one of {sorted(valid_variants)}."
            )
        if self.top_m is not None and self.channel_variant != "channel":
            raise ValueError(
                "top_m is only supported with channel_variant='channel'."
            )
        if self.top_m is not None and self.class_prior == "uniform":
            raise ValueError(
                "top_m is only supported with class_prior in "
                "{'classifier', 'kernel'} for channel_variant='channel'."
            )
        if self.top_m is not None and self.gain_type != "entropy":
            raise ValueError(
                "top_m is only supported with gain_type='entropy' "
                "for channel_variant='channel'."
            )

        # Candidate sample posteriors/embeddings and (optionally) annotator embeddings.
        cand_extra_outputs = ["embeddings"]
        if self.use_annotator_embeddings:
            cand_extra_outputs.append("annotator_embeddings")
        t0 = time.perf_counter()
        cand_out = clf.predict_proba(
            X[sample_indices],
            extra_outputs=cand_extra_outputs,
        )
        if profile:
            self._profile_add(timings, "candidate_predict_proba", t0)
        if not isinstance(cand_out, (tuple, list)):
            raise ValueError(
                "clf.predict_proba must return a tuple when extra_outputs are requested."
            )
        if len(cand_out) < 2:
            raise ValueError(
                "clf.predict_proba returned too few outputs for requested embeddings."
            )
        r_cand = cand_out[0]
        X_cand_emb = cand_out[1]
        A_pred = cand_out[2] if (self.use_annotator_embeddings and len(cand_out) > 2) else None
        r_cand = np.asarray(r_cand, dtype=float)
        r_cand = np.clip(r_cand, 1e-15, 1.0)
        r_cand = r_cand / np.maximum(r_cand.sum(axis=1, keepdims=True), 1e-15)
        X_cand_emb = _l2_normalize(np.asarray(X_cand_emb, dtype=float))

        is_lbld = is_labeled(y=y, missing_label=clf.missing_label)
        obs_s, obs_a = np.where(is_lbld)
        if obs_s.size == 0:
            U = np.full(
                (len(sample_indices), len(annotator_indices)),
                np.nan,
                dtype=float,
            )
            if available_mask is not None:
                U = np.where(available_mask, U, np.nan)
            self.last_utility_mean_ = U.copy()
            return U

        classes = np.asarray(clf.classes_)
        class_to_idx = {c: i for i, c in enumerate(classes)}

        y_obs_raw = np.asarray(y[obs_s, obs_a])
        try:
            y_obs_idx = np.array([class_to_idx[v] for v in y_obs_raw], dtype=int)
        except KeyError as e:
            raise ValueError(f"Observed label {e.args[0]!r} not found in clf.classes_")

        y_obs_oh = np.eye(K, dtype=float)[y_obs_idx]

        # Observed sample embeddings + mode-specific correctness evidence.
        n_annotators_total = y.shape[1]
        t0 = time.perf_counter()
        (
            r_obs,
            X_obs_emb,
            m_obs,
            responsibility_obs,
            _resolved_correctness_mode,
        ) = self._resolve_observed_annotation_evidence(
            clf=clf,
            X_obs=X[obs_s],
            obs_s=obs_s,
            obs_a=obs_a,
            y_obs_idx=y_obs_idx,
            n_annotators_total=n_annotators_total,
        )
        if profile:
            self._profile_add(timings, "observed_evidence", t0)
        _, obs_first_idx = np.unique(obs_s, return_index=True)
        X_obs_cls_emb = X_obs_emb[obs_first_idx]
        r_obs_cls = r_obs[obs_first_idx]

        eps_prior = 1e-6
        global_obs_acc_mean = float(np.mean(m_obs))
        global_obs_acc_mean = float(
            np.clip(global_obs_acc_mean, eps_prior, 1.0 - eps_prior)
        )

        obs_count_by_annotator = np.bincount(
            obs_a, minlength=n_annotators_total
        ).astype(float)
        obs_sum_by_annotator = np.bincount(
            obs_a, weights=m_obs, minlength=n_annotators_total
        ).astype(float)
        obs_mean_by_annotator = np.divide(
            obs_sum_by_annotator,
            np.maximum(obs_count_by_annotator, 1.0),
        )
        obs_mean_by_annotator = np.clip(
            obs_mean_by_annotator, eps_prior, 1.0 - eps_prior
        )

        # Global annotator embeddings are optional. If unavailable, use exact annotator identity weights.
        A_all = None
        A_obs_emb = None
        use_annotator_kernel = False
        gamma_a_val = None
        if A_pred is not None:
            A_pred = np.asarray(A_pred, dtype=float)
            if A_pred.ndim == 2 and A_pred.shape[0] == y.shape[1]:
                A_all = _l2_normalize(A_pred)
                A_obs_emb = A_all[obs_a]
                use_annotator_kernel = True

        t0 = time.perf_counter()
        gamma_x_global = self._resolve_gamma_from_embeddings(
            X_obs_emb, self.gamma_x
        )
        if use_annotator_kernel:
            gamma_a_val = self._resolve_gamma_from_embeddings(
                A_all, self.gamma_a
            )
        if profile:
            self._profile_add(timings, "kernel_bandwidth", t0)
        t0 = time.perf_counter()
        r_cand_prior = self._resolve_class_prior(
            r=r_cand,
            X_cand_emb=X_cand_emb,
            X_obs_cls_emb=X_obs_cls_emb,
            r_obs_cls=r_obs_cls,
            gamma_x=gamma_x_global,
            rng=rng,
        )
        threshold = 0.95

        cs = np.cumsum(-np.sort(-r_cand_prior.mean(axis=1), axis=1), axis=1)
        reached = cs >= threshold

        n_elements = np.where(
            reached.any(axis=1),
            reached.argmax(axis=1) + 1,
            -1,  # threshold never reached
        )
        print(np.mean(n_elements))
        if profile:
            self._profile_add(timings, "class_prior", t0)

        # Local sample-kernel weights from observed pairs to candidate samples.
        t0 = time.perf_counter()
        Kx_obs_cand_local_global = rbf_kernel(
            X_obs_emb, X_cand_emb, gamma=gamma_x_global
        )
        if profile:
            self._profile_add(timings, "sample_kernel", t0)

        if self._accuracy_mean_mode == "fixed":
            prior_acc_global = float(self.accuracy_mean)
        else:
            prior_acc_global = global_obs_acc_mean
        prior_acc_global = float(
            np.clip(prior_acc_global, eps_prior, 1.0 - eps_prior)
        )
        alpha0_global = prior_acc_global * self.accuracy_strength
        beta0_global = (1.0 - prior_acc_global) * self.accuracy_strength
        gamma0 = np.full(
            K, self.channel_label_dirichlet_strength / K, dtype=float
        )
        delta0_full_global = self._full_confusion_dirichlet_prior(
            K=K,
            accuracy_mean=prior_acc_global,
            row_strength=self.accuracy_strength,
        )

        U = np.empty(
            (len(sample_indices), len(annotator_indices)), dtype=float
        )
        store_intervals = (
            self.store_utility_intervals
            and self.channel_variant == "full_confusion"
            and self.n_mc_samples > 0
        )
        U_lcb = np.full_like(U, np.nan) if store_intervals else None
        U_ucb = np.full_like(U, np.nan) if store_intervals else None
        obs_indices_by_annotator = None
        if self.channel_variant == "full_confusion" and not use_annotator_kernel:
            obs_indices_by_annotator = [
                np.flatnonzero(obs_a == idx)
                for idx in range(n_annotators_total)
            ]

        for j_a, a in enumerate(annotator_indices):
            a = int(a)
            if a < 0 or a >= n_annotators_total:
                raise ValueError(
                    f"Annotator index {a} out of bounds for y with "
                    f"{n_annotators_total} annotators."
                )
            if self._accuracy_mean_mode == "per_annotator_observed":
                if obs_count_by_annotator[a] > 0:
                    prior_acc = float(obs_mean_by_annotator[a])
                else:
                    prior_acc = global_obs_acc_mean
                prior_acc = float(np.clip(prior_acc, eps_prior, 1.0 - eps_prior))
                alpha0 = prior_acc * self.accuracy_strength
                beta0 = (1.0 - prior_acc) * self.accuracy_strength
                delta0_full = self._full_confusion_dirichlet_prior(
                    K=K,
                    accuracy_mean=prior_acc,
                    row_strength=self.accuracy_strength,
                )
            else:
                alpha0 = alpha0_global
                beta0 = beta0_global
                delta0_full = delta0_full_global

            if self.gamma_x_scope == "per_annotator":
                obs_mask_a = obs_a == a
                if np.count_nonzero(obs_mask_a) >= 1:
                    t0 = time.perf_counter()
                    gamma_x_a = self._resolve_gamma_from_embeddings(
                        X_obs_emb[obs_mask_a], self.gamma_x
                    )
                    if profile:
                        self._profile_add(timings, "kernel_bandwidth", t0)
                else:
                    gamma_x_a = gamma_x_global
                t0 = time.perf_counter()
                Kx_obs_cand_local = rbf_kernel(
                    X_obs_emb, X_cand_emb, gamma=gamma_x_a
                )
                if profile:
                    self._profile_add(timings, "sample_kernel", t0)
            else:
                Kx_obs_cand_local = Kx_obs_cand_local_global

            Kx_obs_cand = self._mix_with_global_sample_kernel(
                Kx_obs_cand_local,
                lam=self.annotator_lambda,
            )

            if use_annotator_kernel:
                t0 = time.perf_counter()
                Ka_obs = rbf_kernel(
                    A_obs_emb, A_all[[a]], gamma=gamma_a_val
                ).reshape(-1)
                if profile:
                    self._profile_add(timings, "annotator_kernel", t0)
            else:
                Ka_obs = (obs_a == a).astype(float)

            K_obs_cand = Kx_obs_cand * Ka_obs[:, None]
            uses_channel_prior = (
                self.channel_variant == "full_confusion"
                and self.full_confusion_prior_source == "channel"
            )

            if self.channel_variant in {
                "channel",
                "pi_mixture_channel",
                "scalar_uniform_confusion",
                "diag_uniform_confusion",
            } or uses_channel_prior:
                alpha, beta, _ = self.parzen_beta_posterior(
                    K=K_obs_cand,
                    p=m_obs,
                    alpha0=alpha0,
                    beta0=beta0,
                    use_ess=self.use_ess_beta,
                    tau=self.tau_beta,
                )
            else:
                alpha = beta = None

            gamma_cand = None
            if (
                self.channel_variant in {"channel", "pi_mixture_channel"}
                or uses_channel_prior
            ):
                gamma_cand, _ = self.parzen_dirichlet_posterior(
                    K=K_obs_cand,
                    Y=y_obs_oh,
                    gamma0=gamma0,
                    use_ess=self.use_ess_label_dirichlet,
                    tau=self.tau_label_dirichlet,
                )

                # Deterministic full-K path: reuse the shared closed-form gain helper.
                if (
                    self.channel_variant == "channel"
                    and self.n_mc_samples <= 0
                    and (self.top_m is None or self.top_m >= K)
                ):
                    U_col = self._ig_channel_full_batch(
                        r=r_cand_prior,
                        alpha=alpha,
                        beta=beta,
                        gamma=gamma_cand,
                        rng=rng,
                    )
                    if available_mask is not None:
                        U_col = np.where(available_mask[:, j_a], U_col, np.nan)
                    U[:, j_a] = U_col
                    continue

            if self.channel_variant == "diag_uniform_confusion":
                alpha_diag = np.empty((len(sample_indices), K), dtype=float)
                beta_diag = np.empty((len(sample_indices), K), dtype=float)
                y_eq = y_obs_oh
                row_responsibility_obs = responsibility_obs
                for z in range(K):
                    K_row = K_obs_cand * row_responsibility_obs[:, [z]]
                    a_z, b_z, _ = self.parzen_beta_posterior(
                        K=K_row,
                        p=y_eq[:, z],
                        alpha0=alpha0,
                        beta0=beta0,
                        use_ess=self.use_ess_beta,
                        tau=self.tau_beta,
                    )
                    alpha_diag[:, z] = a_z
                    beta_diag[:, z] = b_z
            else:
                alpha_diag = beta_diag = None

            if self.channel_variant == "full_confusion":
                t0 = time.perf_counter()
                row_responsibility_obs = responsibility_obs
                if obs_indices_by_annotator is not None:
                    obs_idx_a = obs_indices_by_annotator[a]
                    K_full = Kx_obs_cand[obs_idx_a]
                    Y_full = y_obs_oh[obs_idx_a]
                    responsibility_full = row_responsibility_obs[obs_idx_a]
                else:
                    K_full = K_obs_cand
                    Y_full = y_obs_oh
                    responsibility_full = row_responsibility_obs
                if uses_channel_prior:
                    delta0_full_for_candidates = (
                        self._channel_prior_full_confusion_dirichlet_prior(
                            alpha=alpha,
                            beta=beta,
                            gamma=gamma_cand,
                            row_strength=(
                                self.full_confusion_channel_prior_strength
                            ),
                        )
                    )
                else:
                    delta0_full_for_candidates = delta0_full
                confusion_rows = self.full_confusion_dirichlet_posterior(
                    K=K_full,
                    Y=Y_full,
                    row_responsibility=responsibility_full,
                    delta0=delta0_full_for_candidates,
                    use_ess=self.use_ess_label_dirichlet,
                    tau=self.tau_label_dirichlet,
                )
                if profile:
                    self._profile_add(timings, "full_confusion_posterior", t0)
            else:
                confusion_rows = None

            # Vectorized fast paths for full-K variants.
            if (
                self.channel_variant == "channel"
                and (self.top_m is None or self.top_m >= K)
            ):
                U_col = self._ig_channel_full_batch(
                    r=r_cand_prior,
                    alpha=alpha,
                    beta=beta,
                    gamma=gamma_cand,
                    rng=rng,
                )
                if available_mask is not None:
                    U_col = np.where(available_mask[:, j_a], U_col, np.nan)
                U[:, j_a] = U_col
                continue

            if (
                self.channel_variant == "channel"
                and self.top_m is not None
                and self.top_m < K
            ):
                U_col = self._ig_channel_topm_batch(
                    r=r_cand_prior,
                    alpha=alpha,
                    beta=beta,
                    gamma=gamma_cand,
                    top_m=int(self.top_m),
                    rng=rng,
                )
                if available_mask is not None:
                    U_col = np.where(available_mask[:, j_a], U_col, np.nan)
                U[:, j_a] = U_col
                continue

            if self.channel_variant == "pi_mixture_channel":
                U_col = self._ig_pi_mixture_channel_batch(
                    r=r_cand_prior,
                    alpha=alpha,
                    beta=beta,
                    gamma=gamma_cand,
                    K_pair=K_obs_cand,
                    rng=rng,
                )
                if available_mask is not None:
                    U_col = np.where(available_mask[:, j_a], U_col, np.nan)
                U[:, j_a] = U_col
                continue

            if self.channel_variant == "scalar_uniform_confusion":
                U_col = self._ig_scalar_uniform_confusion_batch(
                    r=r_cand_prior,
                    alpha=alpha,
                    beta=beta,
                    rng=rng,
                )
                if available_mask is not None:
                    U_col = np.where(available_mask[:, j_a], U_col, np.nan)
                U[:, j_a] = U_col
                continue

            if self.channel_variant == "diag_uniform_confusion":
                U_col = self._ig_diag_uniform_confusion_batch(
                    r=r_cand_prior,
                    alpha=alpha_diag,
                    beta=beta_diag,
                    rng=rng,
                )
                if available_mask is not None:
                    U_col = np.where(available_mask[:, j_a], U_col, np.nan)
                U[:, j_a] = U_col
                continue

            if self.channel_variant == "full_confusion":
                t0 = time.perf_counter()
                if store_intervals:
                    U_col, U_lcb_col, U_ucb_col = self._ig_full_confusion_batch(
                        r=r_cand_prior,
                        delta=confusion_rows,
                        rng=rng,
                        return_interval_bounds=True,
                    )
                else:
                    U_col = self._ig_full_confusion_batch(
                        r=r_cand_prior,
                        delta=confusion_rows,
                        rng=rng,
                    )
                    U_lcb_col = U_ucb_col = None
                if self.confusion_parameter_gain_weight > 0.0:
                    bonus = (
                        self.confusion_parameter_gain_weight
                        * self._full_confusion_parameter_gain_batch(
                            r=r_cand_prior,
                            delta=confusion_rows,
                        )
                    )
                    U_col = U_col + bonus
                    if U_lcb_col is not None:
                        U_lcb_col = U_lcb_col + bonus
                        U_ucb_col = U_ucb_col + bonus
                if profile:
                    self._profile_add(timings, "gain_computation", t0)
                if available_mask is not None:
                    U_col = np.where(available_mask[:, j_a], U_col, np.nan)
                    if U_lcb_col is not None:
                        U_lcb_col = np.where(
                            available_mask[:, j_a],
                            U_lcb_col,
                            np.nan,
                        )
                        U_ucb_col = np.where(
                            available_mask[:, j_a],
                            U_ucb_col,
                            np.nan,
                        )
                U[:, j_a] = U_col
                if U_lcb is not None:
                    U_lcb[:, j_a] = U_lcb_col
                    U_ucb[:, j_a] = U_ucb_col
                continue

            raise RuntimeError(
                "Unhandled channel variant branch in fast-path computation."
            )

        if available_mask is not None:
            U = np.where(available_mask, U, np.nan)

        self.last_utility_mean_ = U.copy()
        self.last_utility_lcb_ = None if U_lcb is None else U_lcb.copy()
        self.last_utility_ucb_ = None if U_ucb is None else U_ucb.copy()

        if profile:
            self._profile_add(timings, "total", t_total)
            self._profile_print(timings)

        return U

    def _resolve_observed_annotation_evidence(
        self,
        *,
        clf,
        X_obs,
        obs_s: np.ndarray | None = None,
        obs_a: np.ndarray,
        y_obs_idx: np.ndarray,
        n_annotators_total: int,
        eps: float = 1e-15,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
        obs_out = clf.predict_proba(X_obs, extra_outputs=["embeddings"])
        if not isinstance(obs_out, (tuple, list)) or len(obs_out) < 2:
            raise ValueError(
                "clf.predict_proba must return class probabilities and "
                "embeddings for observed annotation evidence."
            )

        r_obs = np.asarray(obs_out[0], dtype=float)
        r_obs = np.clip(r_obs, eps, 1.0)
        r_obs = r_obs / np.maximum(r_obs.sum(axis=1, keepdims=True), eps)
        X_obs_emb = _l2_normalize(np.asarray(obs_out[1], dtype=float))

        observed_prior_obs = self._resolve_observed_class_prior_probabilities(
            r_obs=r_obs,
            X_obs_emb=X_obs_emb,
            obs_s=obs_s,
            eps=eps,
        )
        responsibility_obs = observed_prior_obs.copy()
        m_classifier = np.clip(
            observed_prior_obs[np.arange(obs_a.size), y_obs_idx],
            0.0,
            1.0,
        )

        if self.correctness_mode == "classifier":
            return (
                r_obs,
                X_obs_emb,
                m_classifier,
                responsibility_obs,
                "classifier",
            )

        m_perf = None

        if self.correctness_mode in {"confusion_posterior", "auto"}:
            confusion = self._predict_optional_extra_output(
                clf=clf,
                X=X_obs,
                output_name="annotator_confusion_matrices",
                allow_missing=self.correctness_mode == "auto",
            )
            if confusion is not None:
                pair_confusions = self._take_observed_pair_confusions(
                    confusion=confusion,
                    obs_a=obs_a,
                    n_annotators_total=n_annotators_total,
                    eps=eps,
                )
                responsibility_obs = self._confusion_posterior_responsibilities(
                    r_obs=observed_prior_obs,
                    pair_confusions=pair_confusions,
                    y_obs_idx=y_obs_idx,
                    eps=eps,
                )
                m_conf = np.clip(
                    responsibility_obs[np.arange(obs_a.size), y_obs_idx],
                    0.0,
                    1.0,
                )
                return (
                    r_obs,
                    X_obs_emb,
                    m_conf,
                    responsibility_obs,
                    "confusion_posterior",
                )
            if self.correctness_mode == "confusion_posterior":
                raise RuntimeError(
                    "correctness_mode='confusion_posterior' requires "
                    "`annotator_confusion_matrices` from clf.predict_proba."
                )

        if self.correctness_mode in {
            "annotator_perf",
            "annotator_perf_posterior",
            "auto",
        }:
            annotator_perf = self._predict_optional_extra_output(
                clf=clf,
                X=X_obs,
                output_name="annotator_perf",
                allow_missing=self.correctness_mode == "auto",
            )
            if annotator_perf is not None:
                m_perf = self._take_observed_annotator_perf(
                    annotator_perf=annotator_perf,
                    obs_a=obs_a,
                    n_annotators_total=n_annotators_total,
                )
                if self.correctness_mode == "annotator_perf_posterior":
                    synthetic_confusions = (
                        self._uniform_confusion_from_annotator_perf(
                            theta=m_perf,
                            n_classes=r_obs.shape[1],
                        )
                    )
                    responsibility_obs = (
                        self._confusion_posterior_responsibilities(
                            r_obs=observed_prior_obs,
                            pair_confusions=synthetic_confusions,
                            y_obs_idx=y_obs_idx,
                            eps=eps,
                        )
                    )
                    m_perf_post = np.clip(
                        responsibility_obs[np.arange(obs_a.size), y_obs_idx],
                        0.0,
                        1.0,
                    )
                    return (
                        r_obs,
                        X_obs_emb,
                        m_perf_post,
                        responsibility_obs,
                        "annotator_perf_posterior",
                    )
                if self.correctness_mode == "annotator_perf":
                    return (
                        r_obs,
                        X_obs_emb,
                        m_perf,
                        responsibility_obs,
                        "annotator_perf",
                    )
            elif self.correctness_mode in {
                "annotator_perf",
                "annotator_perf_posterior",
            }:
                raise RuntimeError(
                    f"correctness_mode='{self.correctness_mode}' requires "
                    "`annotator_perf` from clf.predict_proba."
                )

        if m_perf is not None:
            return (
                r_obs,
                X_obs_emb,
                m_perf,
                responsibility_obs,
                "annotator_perf",
            )
        return (
            r_obs,
            X_obs_emb,
            m_classifier,
            responsibility_obs,
            "classifier",
        )

    def _resolve_observed_class_prior_probabilities(
        self,
        *,
        r_obs: np.ndarray,
        X_obs_emb: np.ndarray,
        obs_s: np.ndarray | None,
        eps: float = 1e-15,
    ) -> np.ndarray:
        r_obs = np.asarray(r_obs, dtype=float)
        if self.observed_class_prior == "classifier":
            return r_obs
        if self.observed_class_prior != "kernel":
            raise ValueError(
                "observed_class_prior must be one of {'classifier', 'kernel'}"
            )
        if obs_s is None:
            raise ValueError(
                "observed_class_prior='kernel' requires observed sample "
                "indices `obs_s`."
            )

        obs_s = np.asarray(obs_s)
        if obs_s.ndim != 1 or obs_s.shape[0] != r_obs.shape[0]:
            raise ValueError(
                "`obs_s` must be a 1D array with one entry per observed "
                "annotation."
            )

        _, obs_first_idx = np.unique(obs_s, return_index=True)
        obs_cls_s = obs_s[obs_first_idx]
        X_obs_cls_emb = X_obs_emb[obs_first_idx]
        r_obs_cls = r_obs[obs_first_idx]
        gamma_x = self._resolve_gamma_from_embeddings(X_obs_emb, self.gamma_x)
        alpha = self._kernel_class_dirichlet_posterior(
            X_query_emb=X_obs_emb,
            X_obs_cls_emb=X_obs_cls_emb,
            r_obs_cls=r_obs_cls,
            gamma_x=gamma_x,
            query_sample_ids=obs_s,
            support_sample_ids=obs_cls_s,
            leave_one_out=self.observed_class_prior_leave_one_out,
        )
        return alpha / np.maximum(alpha.sum(axis=1, keepdims=True), eps)

    def _kernel_class_dirichlet_posterior(
        self,
        *,
        X_query_emb: np.ndarray,
        X_obs_cls_emb: np.ndarray,
        r_obs_cls: np.ndarray,
        gamma_x: float,
        query_sample_ids: np.ndarray | None = None,
        support_sample_ids: np.ndarray | None = None,
        leave_one_out: bool = False,
    ) -> np.ndarray:
        K_cls_local = rbf_kernel(
            X_obs_cls_emb,
            X_query_emb,
            gamma=float(gamma_x),
        )
        same_sample = None
        if leave_one_out:
            if query_sample_ids is None or support_sample_ids is None:
                raise ValueError(
                    "leave_one_out=True requires query and support sample ids."
                )
            query_sample_ids = np.asarray(query_sample_ids)
            support_sample_ids = np.asarray(support_sample_ids)
            if (
                query_sample_ids.ndim != 1
                or query_sample_ids.shape[0] != K_cls_local.shape[1]
            ):
                raise ValueError(
                    "query_sample_ids must match the number of query samples."
                )
            if (
                support_sample_ids.ndim != 1
                or support_sample_ids.shape[0] != K_cls_local.shape[0]
            ):
                raise ValueError(
                    "support_sample_ids must match the number of support samples."
                )
            same_sample = support_sample_ids[:, None] == query_sample_ids[None, :]
        K_cls = self._mix_with_global_sample_kernel(
            K_cls_local,
            lam=self.class_prior_lambda,
        )
        if same_sample is not None:
            K_cls = np.where(same_sample, 0.0, K_cls)
        K = r_obs_cls.shape[1]
        alpha0 = np.full(K, self.class_prior_strength / K, dtype=float)
        alpha, _ = self.parzen_dirichlet_posterior(
            K=K_cls,
            Y=r_obs_cls,
            gamma0=alpha0,
            use_ess=self.use_ess_class_prior,
            tau=self.tau_class_prior,
        )
        return alpha

    @staticmethod
    def _predict_optional_extra_output(
        *,
        clf,
        X,
        output_name: str,
        allow_missing: bool,
    ):
        try:
            out = clf.predict_proba(X, extra_outputs=[output_name])
        except Exception:
            if allow_missing:
                return None
            raise

        if not isinstance(out, (tuple, list)) or len(out) < 2:
            raise ValueError(
                f"clf.predict_proba must return `{output_name}` when requested."
            )
        return out[1]

    @staticmethod
    def _take_observed_annotator_perf(
        *,
        annotator_perf: np.ndarray,
        obs_a: np.ndarray,
        n_annotators_total: int,
    ) -> np.ndarray:
        annotator_perf = np.asarray(annotator_perf, dtype=float)
        n_obs = obs_a.size
        if annotator_perf.ndim != 2:
            raise ValueError(
                "`annotator_perf` must have shape "
                "(n_obs_samples, n_annotators)."
            )
        if annotator_perf.shape[0] != n_obs:
            raise ValueError(
                "`annotator_perf` must match the number of observed pairs."
            )
        if annotator_perf.shape[1] != n_annotators_total:
            raise ValueError(
                "`annotator_perf` must include one column per annotator."
            )
        return np.clip(
            annotator_perf[np.arange(n_obs), obs_a],
            0.0,
            1.0,
        )

    @staticmethod
    def _uniform_confusion_from_annotator_perf(
        *,
        theta: np.ndarray,
        n_classes: int,
    ) -> np.ndarray:
        theta = np.clip(np.asarray(theta, dtype=float), 0.0, 1.0)
        if theta.ndim != 1:
            raise ValueError("theta must be 1D for observed annotator pairs.")
        if n_classes < 2:
            raise ValueError("n_classes must be at least 2.")

        n_obs = theta.shape[0]
        off = (1.0 - theta) / (n_classes - 1)
        C = np.full((n_obs, n_classes, n_classes), 0.0, dtype=float)
        C[:] = off[:, None, None]
        idx = np.arange(n_classes)
        C[:, idx, idx] = theta[:, None]
        return C

    @staticmethod
    def _take_observed_pair_confusions(
        *,
        confusion: np.ndarray,
        obs_a: np.ndarray,
        n_annotators_total: int,
        eps: float,
    ) -> np.ndarray:
        confusion = np.asarray(confusion, dtype=float)
        n_obs = obs_a.size
        if confusion.ndim == 3:
            if confusion.shape[0] != n_annotators_total:
                raise ValueError(
                    "`annotator_confusion_matrices` must include one matrix "
                    "per annotator."
                )
            pair_confusions = confusion[obs_a]
        elif confusion.ndim == 4:
            if confusion.shape[0] != n_obs:
                raise ValueError(
                    "Sample-specific `annotator_confusion_matrices` must "
                    "match the number of observed pairs."
                )
            if confusion.shape[1] != n_annotators_total:
                raise ValueError(
                    "Sample-specific `annotator_confusion_matrices` must "
                    "include one matrix per annotator."
                )
            pair_confusions = confusion[np.arange(n_obs), obs_a]
        else:
            raise ValueError(
                "`annotator_confusion_matrices` must have shape "
                "(n_annotators, K, K) or (n_obs_samples, n_annotators, K, K)."
            )

        if pair_confusions.ndim != 3 or pair_confusions.shape[1] != pair_confusions.shape[2]:
            raise ValueError(
                "Observed annotator confusions must have shape (n_obs, K, K)."
            )
        pair_confusions = np.clip(pair_confusions, 0.0, None)
        return pair_confusions / np.maximum(
            pair_confusions.sum(axis=2, keepdims=True),
            eps,
        )

    @staticmethod
    def _confusion_posterior_responsibilities(
        *,
        r_obs: np.ndarray,
        pair_confusions: np.ndarray,
        y_obs_idx: np.ndarray,
        eps: float,
    ) -> np.ndarray:
        emission = np.take_along_axis(
            pair_confusions,
            y_obs_idx[:, None, None],
            axis=2,
        ).squeeze(axis=2)
        rho_num = r_obs * emission
        rho_den = rho_num.sum(axis=1, keepdims=True)
        rho = np.divide(
            rho_num,
            np.maximum(rho_den, eps),
            out=np.zeros_like(rho_num),
            where=rho_den > eps,
        )
        zero_mass = np.squeeze(rho_den, axis=1) <= eps
        if np.any(zero_mass):
            rho[zero_mass] = r_obs[zero_mass]
        return rho

    def _resolve_class_prior(
        self,
        r: np.ndarray,
        *,
        X_cand_emb: np.ndarray | None = None,
        X_obs_cls_emb: np.ndarray | None = None,
        r_obs_cls: np.ndarray | None = None,
        gamma_x: float | None = None,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        r = np.asarray(r, dtype=float)
        if r.ndim != 2:
            raise ValueError(
                f"r must have shape (n_samples, n_classes), got {r.shape}."
            )
        T = self._mc_draw_count()

        if self.class_prior == "classifier":
            return np.repeat(r[:, None, :], T, axis=1)
        K = r.shape[1]
        if K < 2:
            raise ValueError("IG requires at least 2 classes.")
        if self.class_prior == "uniform":
            return np.full((r.shape[0], T, K), 1.0 / K, dtype=float)

        if (
            X_cand_emb is None
            or X_obs_cls_emb is None
            or r_obs_cls is None
            or gamma_x is None
        ):
            raise ValueError(
                "class_prior='kernel' requires candidate embeddings, "
                "deduplicated labeled-sample embeddings, labeled-sample "
                "posteriors, and gamma_x."
            )

        X_cand_emb = np.asarray(X_cand_emb, dtype=float)
        X_obs_cls_emb = np.asarray(X_obs_cls_emb, dtype=float)
        r_obs_cls = np.asarray(r_obs_cls, dtype=float)
        if X_cand_emb.ndim != 2 or X_obs_cls_emb.ndim != 2:
            raise ValueError("X_cand_emb and X_obs_cls_emb must be 2D.")
        if r_obs_cls.ndim != 2 or r_obs_cls.shape[1] != K:
            raise ValueError(
                f"r_obs_cls must have shape (n_obs_samples, {K}), got {r_obs_cls.shape}."
            )
        if X_obs_cls_emb.shape[0] != r_obs_cls.shape[0]:
            raise ValueError(
                "X_obs_cls_emb and r_obs_cls must have the same number of rows."
            )
        if X_cand_emb.shape[0] != r.shape[0]:
            raise ValueError(
                "X_cand_emb must have the same number of rows as r."
            )

        K_cls_local = rbf_kernel(
            X_obs_cls_emb, X_cand_emb, gamma=float(gamma_x)
        )
        K_cls = self._mix_with_global_sample_kernel(
            K_cls_local,
            lam=self.class_prior_lambda,
        )
        alpha0 = np.full(K, self.class_prior_strength / K, dtype=float)
        alpha, _ = self.parzen_dirichlet_posterior(
            K=K_cls,
            Y=r_obs_cls,
            gamma0=alpha0,
            use_ess=self.use_ess_class_prior,
            tau=self.tau_class_prior,
        )

        if self.sample_class_prior and self.n_mc_samples > 0:
            if rng is None:
                raise ValueError(
                    "sample_class_prior=True requires an RNG."
                )
            alpha_bt = np.clip(alpha[:, None, :], 1e-12, None)
            if T != 1:
                alpha_bt = np.repeat(alpha_bt, T, axis=1)
            X = rng.gamma(shape=alpha_bt, scale=1.0)
            return X / np.maximum(X.sum(axis=2, keepdims=True), 1e-12)

        mean = alpha / np.maximum(alpha.sum(axis=1, keepdims=True), 1e-12)
        return np.repeat(mean[:, None, :], T, axis=1)

    # -------------------------
    # Gain computation
    # -------------------------
    def _ig_channel_full_batch(
        self,
        *,
        r: np.ndarray,
        alpha: np.ndarray,
        beta: np.ndarray,
        gamma: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        r = np.asarray(r, dtype=float)
        alpha = np.asarray(alpha, dtype=float)
        beta = np.asarray(beta, dtype=float)
        gamma = np.asarray(gamma, dtype=float)

        if r.ndim != 3:
            raise ValueError(
                "r must have shape (n_samples, n_draws, n_classes) in batch channel."
            )
        S, T, K = r.shape
        thetas = self._sample_theta_batch(
            alpha=alpha,
            beta=beta,
            rng=rng,
            n_draws=T,
        )

        if self.channel_wrong_label_mode == "sample_dirichlet_wrong":
            g_cap = None
            if (
                (self.entropy_response_cap or self.response_entropy_cap)
                and self.gain_type == "entropy"
            ):
                g_cap = self._sample_label_distribution_batch(
                    gamma=gamma,
                    rng=rng,
                    n_draws=T,
                )
            Cs = self._channel_confusion_from_wrong_dirichlet_batch(
                gamma=gamma,
                theta=thetas,
                rng=rng,
                sample=self._use_mc_label_dirichlet(),
            )
            ig_draws = self._pair_gain(
                r,
                C=Cs,
                response_distribution=g_cap,
            )
            ig_draws = self._apply_response_entropy_regularizer(
                ig_draws,
                g=g_cap,
            )
            return self._aggregate_gain_draws(ig_draws)

        if self._use_mc_label_dirichlet():
            g_alpha = np.clip(gamma[:, None, :], 1e-12, None)
            if T != 1:
                g_alpha = np.repeat(g_alpha, T, axis=1)
            g = rng.gamma(shape=g_alpha, scale=1.0)
            g = g / np.maximum(g.sum(axis=-1, keepdims=True), 1e-12)
        else:
            g_mean = gamma / np.maximum(gamma.sum(axis=1, keepdims=True), 1e-12)
            g = np.repeat(g_mean[:, None, :], T, axis=1)

        ig_draws = self._pair_gain(
            r.reshape(-1, K),
            P_perf=thetas.reshape(-1, 1),
            P_annot=g.reshape(-1, 1, K),
            response_distribution=g.reshape(-1, 1, K),
        ).reshape(S, T)
        ig_draws = self._apply_response_entropy_regularizer(
            ig_draws,
            g=g,
        )
        return self._aggregate_gain_draws(ig_draws)

    def _ig_channel_topm_batch(
        self,
        *,
        r: np.ndarray,
        alpha: np.ndarray,
        beta: np.ndarray,
        gamma: np.ndarray,
        top_m: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        r = np.asarray(r, dtype=float)
        alpha = np.asarray(alpha, dtype=float)
        beta = np.asarray(beta, dtype=float)
        gamma = np.asarray(gamma, dtype=float)

        r_red, gamma_red = self._reduce_topm_vectors_batch(
            r=r, gamma=gamma, top_m=top_m
        )

        if r_red.ndim != 3:
            raise ValueError(
                "r_red must have shape (n_samples, n_draws, n_classes)."
            )
        S, T, K_red = r_red.shape
        thetas = self._sample_theta_batch(
            alpha=alpha,
            beta=beta,
            rng=rng,
            n_draws=T,
        )

        if self.channel_wrong_label_mode == "sample_dirichlet_wrong":
            g_cap = None
            if (
                (self.entropy_response_cap or self.response_entropy_cap)
                and self.gain_type == "entropy"
            ):
                g_cap = self._sample_label_distribution_batch(
                    gamma=gamma_red,
                    rng=rng,
                    n_draws=T,
                )
            Cs = self._channel_confusion_from_wrong_dirichlet_batch(
                gamma=gamma_red,
                theta=thetas,
                rng=rng,
                sample=self._use_mc_label_dirichlet(),
            )
            ig_draws = self._pair_gain(
                r_red,
                C=Cs,
                response_distribution=g_cap,
            )
            ig_draws = self._apply_response_entropy_regularizer(
                ig_draws,
                g=g_cap,
            )
            return self._aggregate_gain_draws(ig_draws)

        if self._use_mc_label_dirichlet():
            if gamma_red.ndim == 2:
                g_alpha = np.clip(gamma_red[:, None, :], 1e-12, None)
                if T != 1:
                    g_alpha = np.repeat(g_alpha, T, axis=1)
            else:
                g_alpha = np.clip(gamma_red, 1e-12, None)
            g_red = rng.gamma(shape=g_alpha, scale=1.0)
            g_red = g_red / np.maximum(g_red.sum(axis=-1, keepdims=True), 1e-12)
        else:
            g_mean_red = gamma_red / np.maximum(
                gamma_red.sum(axis=-1, keepdims=True), 1e-12
            )
            if gamma_red.ndim == 2:
                g_red = np.repeat(g_mean_red[:, None, :], T, axis=1)
            else:
                g_red = g_mean_red

        ig_draws = self._pair_gain(
            r_red.reshape(-1, K_red),
            P_perf=thetas.reshape(-1, 1),
            P_annot=g_red.reshape(-1, 1, K_red),
            response_distribution=g_red.reshape(-1, 1, K_red),
        ).reshape(S, T)
        ig_draws = self._apply_response_entropy_regularizer(
            ig_draws,
            g=g_red,
        )
        return self._aggregate_gain_draws(ig_draws)

    def _ig_pi_mixture_channel_batch(
        self,
        *,
        r: np.ndarray,
        alpha: np.ndarray,
        beta: np.ndarray,
        gamma: np.ndarray,
        K_pair: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        r = np.asarray(r, dtype=float)
        alpha = np.asarray(alpha, dtype=float)
        beta = np.asarray(beta, dtype=float)
        gamma = np.asarray(gamma, dtype=float)
        K_pair = np.asarray(K_pair, dtype=float)

        if r.ndim != 3:
            raise ValueError(
                "r must have shape (n_samples, n_draws, n_classes) "
                "in pi_mixture_channel."
            )
        if K_pair.ndim != 2:
            raise ValueError(
                "K_pair must have shape (n_observations, n_samples) "
                "in pi_mixture_channel."
            )
        if K_pair.shape[1] != r.shape[0]:
            raise ValueError("K_pair must agree with r on n_samples.")

        batch_size = self.gain_batch_size
        if batch_size is not None and r.shape[0] > batch_size:
            gains = np.empty(r.shape[0], dtype=float)
            for start in range(0, r.shape[0], batch_size):
                stop = min(start + batch_size, r.shape[0])
                gains[start:stop] = self._ig_pi_mixture_channel_batch_inner(
                    r=r[start:stop],
                    alpha=alpha[start:stop],
                    beta=beta[start:stop],
                    gamma=gamma[start:stop],
                    K_pair=K_pair[:, start:stop],
                    rng=rng,
                )
            return gains

        return self._ig_pi_mixture_channel_batch_inner(
            r=r,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            K_pair=K_pair,
            rng=rng,
        )

    def _ig_pi_mixture_channel_batch_inner(
        self,
        *,
        r: np.ndarray,
        alpha: np.ndarray,
        beta: np.ndarray,
        gamma: np.ndarray,
        K_pair: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        S, T, K = r.shape
        thetas = self._sample_theta_batch(
            alpha=alpha,
            beta=beta,
            rng=rng,
            n_draws=T,
        )
        g = self._sample_label_distribution_batch(
            gamma=gamma,
            rng=rng,
            n_draws=T,
        )
        C_orig = _channel_confusion_from_theta_g_batch(
            theta=thetas,
            g=g,
            normalize_g=True,
            check_input=False,
        )
        pi = self._pi_mixture_weight(
            g=g,
            K_pair=K_pair,
        )
        Cs = self._pi_mixture_confusion(
            C_orig=C_orig,
            g=g,
            pi=pi,
        )

        ig_draws = self._pair_gain(
            r,
            C=Cs,
            batch_size=self.gain_batch_size,
        )
        ig_draws = self._apply_response_entropy_regularizer(
            ig_draws,
            g=g,
        )
        return self._aggregate_gain_draws(ig_draws)

    def _ig_scalar_uniform_confusion_batch(
        self,
        *,
        r: np.ndarray,
        alpha: np.ndarray,
        beta: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        r = np.asarray(r, dtype=float)
        alpha = np.asarray(alpha, dtype=float)
        beta = np.asarray(beta, dtype=float)

        if r.ndim != 3:
            raise ValueError(
                "r must have shape (n_samples, n_draws, n_classes) in batch confusion."
            )
        _, T, K = r.shape
        thetas = self._sample_theta_batch(
            alpha=alpha,
            beta=beta,
            rng=rng,
            n_draws=T,
        )

        eye = np.eye(K, dtype=float)[None, None, :, :]
        off_base = (
            (np.ones((K, K), dtype=float) - np.eye(K, dtype=float)) / (K - 1)
        )[None, None, :, :]
        Cs = (1.0 - thetas)[..., None, None] * off_base + thetas[
            ..., None, None
        ] * eye

        ig_draws = self._pair_gain(
            r,
            C=Cs,
        )
        return self._aggregate_gain_draws(ig_draws)

    def _ig_diag_uniform_confusion_batch(
        self,
        *,
        r: np.ndarray,
        alpha: np.ndarray,
        beta: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        r = np.asarray(r, dtype=float)
        alpha = np.asarray(alpha, dtype=float)
        beta = np.asarray(beta, dtype=float)

        if r.ndim != 3:
            raise ValueError(
                "r must have shape (n_samples, n_draws, n_classes) in batch confusion."
            )
        S, T, K = r.shape
        if self.n_mc_samples <= 0:
            thetas = self._theta_point_estimate(
                alpha=alpha,
                beta=beta,
            )[:, None, :]
        else:
            thetas = rng.beta(
                alpha[:, None, :],
                beta[:, None, :],
                size=(S, T, K),
            ).astype(float)

        off = (1.0 - thetas) / (K - 1)
        Cs = np.repeat(off[..., None], K, axis=-1)
        idx = np.arange(K)
        Cs[..., idx, idx] = thetas

        ig_draws = self._pair_gain(
            r,
            C=Cs,
        )
        return self._aggregate_gain_draws(ig_draws)

    def _ig_full_confusion_batch(
        self,
        *,
        r: np.ndarray,
        delta: np.ndarray,
        rng: np.random.Generator,
        return_interval_bounds: bool = False,
    ) -> np.ndarray:
        r = np.asarray(r, dtype=float)
        delta = np.asarray(delta, dtype=float)

        if delta.ndim != 3 or delta.shape[1] != delta.shape[2]:
            raise ValueError(
                "delta must have shape (n_samples, K, K) in batch full_confusion."
            )

        if r.ndim != 3:
            raise ValueError(
                "r must have shape (n_samples, n_draws, n_classes) in batch confusion."
            )

        batch_size = self.gain_batch_size
        if batch_size is not None and delta.shape[0] > batch_size:
            gains = np.empty(delta.shape[0], dtype=float)
            lcbs = (
                np.empty(delta.shape[0], dtype=float)
                if return_interval_bounds
                else None
            )
            ucbs = (
                np.empty(delta.shape[0], dtype=float)
                if return_interval_bounds
                else None
            )
            for start in range(0, delta.shape[0], batch_size):
                stop = min(start + batch_size, delta.shape[0])
                out = self._ig_full_confusion_batch_inner(
                    r=r[start:stop],
                    delta=delta[start:stop],
                    rng=rng,
                    return_interval_bounds=return_interval_bounds,
                )
                if return_interval_bounds:
                    gains[start:stop], lcbs[start:stop], ucbs[start:stop] = out
                else:
                    gains[start:stop] = out
            if return_interval_bounds:
                return gains, lcbs, ucbs
            return gains

        return self._ig_full_confusion_batch_inner(
            r=r,
            delta=delta,
            rng=rng,
            return_interval_bounds=return_interval_bounds,
        )

    def _ig_full_confusion_batch_inner(
        self,
        *,
        r: np.ndarray,
        delta: np.ndarray,
        rng: np.random.Generator,
        return_interval_bounds: bool = False,
    ) -> np.ndarray:
        top_m = self.full_confusion_reduce_top_m
        if top_m is not None and top_m < r.shape[-1]:
            r, delta = self._reduce_full_confusion_topm(
                r=r,
                delta=delta,
                top_m=top_m,
            )

        T = r.shape[1]
        if not self._use_mc_label_dirichlet():
            if delta.ndim == 3:
                C_mean = delta / np.maximum(
                    delta.sum(axis=2, keepdims=True),
                    1e-12,
                )
                Cs = np.repeat(C_mean[:, None, :, :], T, axis=1)
            else:
                Cs = delta / np.maximum(
                    delta.sum(axis=3, keepdims=True),
                    1e-12,
                )
        else:
            if delta.ndim == 3:
                alpha = np.clip(delta[:, None, :, :], 1e-12, None)
                if T != 1:
                    alpha = np.repeat(alpha, T, axis=1)
            else:
                alpha = np.clip(delta, 1e-12, None)
            X = rng.gamma(shape=alpha, scale=1.0)
            Cs = X / np.maximum(X.sum(axis=3, keepdims=True), 1e-12)

        if self.gain_type == "entropy":
            ig_draws = self._entropy_gain_from_confusion_batch(
                r=r,
                C=Cs,
            )
            if self.entropy_response_cap:
                cap = self._entropy_gain_upper_bound(r, C=Cs)
                ig_draws = np.minimum(np.maximum(ig_draws, 0.0), cap)
        else:
            ig_draws = self._pair_gain(
                r,
                C=Cs,
                batch_size=self.gain_batch_size,
            )
        gain = self._aggregate_gain_draws(ig_draws)
        if not return_interval_bounds:
            return gain
        lcb, ucb = self._gain_draw_interval_bounds(ig_draws)
        return gain, lcb, ucb

    def _gain_draw_interval_bounds(
        self,
        gain_draws: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        gain_draws = np.asarray(gain_draws, dtype=float)
        if gain_draws.ndim != 2:
            raise ValueError(
                "gain_draws must have shape (n_samples, n_draws)."
            )
        q_low = self.utility_interval_lower
        q_high = self.utility_interval_upper
        if q_low <= 0.0:
            lcb = np.min(gain_draws, axis=1)
        else:
            lcb = np.quantile(gain_draws, q_low, axis=1)
        if q_high >= 1.0:
            ucb = np.max(gain_draws, axis=1)
        else:
            ucb = np.quantile(gain_draws, q_high, axis=1)
        return lcb, ucb

    @staticmethod
    def _reduce_full_confusion_topm(
        *,
        r: np.ndarray,
        delta: np.ndarray,
        top_m: int,
        eps: float = 1e-12,
    ) -> tuple[np.ndarray, np.ndarray]:
        r = np.asarray(r, dtype=float)
        delta = np.asarray(delta, dtype=float)

        if r.ndim != 3:
            raise ValueError(
                "r must have shape (n_samples, n_draws, n_classes)."
            )
        if delta.ndim != 3 or delta.shape[1] != delta.shape[2]:
            raise ValueError(
                "delta must have shape (n_samples, K, K) before reduction."
            )
        S, T, K = r.shape
        if delta.shape[0] != S or delta.shape[1] != K:
            raise ValueError("r and delta must agree on samples/classes.")
        if top_m <= 0:
            raise ValueError("top_m must be positive.")
        if top_m >= K:
            return r, delta

        M = top_m + 1
        r_red = np.empty((S, T, M), dtype=float)
        delta_red = np.empty((S, T, M, M), dtype=float)

        for i in range(S):
            delta_i = delta[i]
            for t in range(T):
                r_it = np.asarray(r[i, t], dtype=float)
                order = np.argsort(-r_it, kind="mergesort")
                keep = order[:top_m]
                other = order[top_m:]

                other_mass = float(r_it[other].sum())
                r_red[i, t, :top_m] = r_it[keep]
                r_red[i, t, top_m] = other_mass
                r_red[i, t] /= np.maximum(r_red[i, t].sum(), eps)

                delta_red[i, t, :top_m, :top_m] = delta_i[np.ix_(keep, keep)]
                delta_red[i, t, :top_m, top_m] = delta_i[
                    np.ix_(keep, other)
                ].sum(axis=1)

                if other_mass > eps:
                    weights = r_it[other] / other_mass
                else:
                    weights = np.full(other.size, 1.0 / other.size)
                other_row = weights @ delta_i[other]
                delta_red[i, t, top_m, :top_m] = other_row[keep]
                delta_red[i, t, top_m, top_m] = other_row[other].sum()

        return r_red, delta_red

    @staticmethod
    def _entropy_gain_from_confusion_batch(
        *,
        r: np.ndarray,
        C: np.ndarray,
        eps: float = 1e-12,
    ) -> np.ndarray:
        r = np.asarray(r, dtype=float)
        C = np.asarray(C, dtype=float)
        r = np.clip(r, eps, 1.0)
        C = np.clip(C, eps, 1.0)
        r = r / np.maximum(r.sum(axis=-1, keepdims=True), eps)
        C = C / np.maximum(C.sum(axis=-1, keepdims=True), eps)

        q = np.einsum("...k,...ky->...y", r, C, optimize=True)
        q = q / np.maximum(q.sum(axis=-1, keepdims=True), eps)
        h_response = KernelSmoothedBayesianAnnotatorGain._entropy_bits(
            q,
            eps=eps,
        )
        h_channel_rows = KernelSmoothedBayesianAnnotatorGain._entropy_bits(
            C,
            eps=eps,
        )
        h_response_given_class = np.sum(r * h_channel_rows, axis=-1)
        return np.maximum(h_response - h_response_given_class, 0.0)

    def _full_confusion_parameter_gain_batch(
        self,
        *,
        r: np.ndarray,
        delta: np.ndarray,
    ) -> np.ndarray:
        r = np.asarray(r, dtype=float)
        delta = np.asarray(delta, dtype=float)
        if r.ndim != 3:
            raise ValueError(
                "r must have shape (n_samples, n_draws, n_classes)."
            )
        if delta.ndim != 3 or delta.shape[1] != delta.shape[2]:
            raise ValueError(
                "delta must have shape (n_samples, K, K) for parameter gain."
            )
        if r.shape[0] != delta.shape[0] or r.shape[2] != delta.shape[1]:
            raise ValueError("r and delta must agree on samples/classes.")

        r = np.clip(r, 1e-12, 1.0)
        r = r / np.maximum(r.sum(axis=2, keepdims=True), 1e-12)
        row_gain = self._dirichlet_one_step_information_gain(delta)
        gain_draws = np.einsum("stk,sk->st", r, row_gain, optimize=True)
        return self._aggregate_gain_draws(np.maximum(gain_draws, 0.0))

    @staticmethod
    def _dirichlet_one_step_information_gain(
        alpha: np.ndarray,
        *,
        eps: float = 1e-12,
    ) -> np.ndarray:
        """Expected entropy reduction of a Dirichlet after one categorical draw.

        The result uses bits, matching the entropy gain used for predictive IG.
        For each row ``alpha`` it computes

            H[Dir(alpha)] - E_{y ~ alpha / alpha0} H[Dir(alpha + e_y)]

        without materializing all ``K`` possible posterior Dirichlets.
        """
        alpha = np.clip(np.asarray(alpha, dtype=float), eps, None)
        alpha0 = alpha.sum(axis=-1)
        K = alpha.shape[-1]
        pred = alpha / np.maximum(alpha0[..., None], eps)

        weighted_log_alpha = np.sum(pred * np.log(alpha), axis=-1)
        delta_sum = np.sum(
            pred
            * (
                alpha * digamma(alpha + 1.0)
                - (alpha - 1.0) * digamma(alpha)
            ),
            axis=-1,
        )
        gain_nats = (
            (alpha0 - K) * digamma(alpha0)
            - weighted_log_alpha
            + np.log(alpha0)
            - (alpha0 + 1.0 - K) * digamma(alpha0 + 1.0)
            + delta_sum
        )
        return np.maximum(gain_nats / np.log(2.0), 0.0)

    def _pair_gain(
        self,
        P: np.ndarray,
        *,
        P_perf: np.ndarray | None = None,
        P_annot: np.ndarray | None = None,
        C: np.ndarray | None = None,
        response_distribution: np.ndarray | None = None,
        batch_size: int | None = None,
    ) -> np.ndarray:
        gain = expected_score_gain(
            P,
            P_perf=P_perf,
            P_annot=P_annot,
            C=C,
            score=self.gain_type,
            normalize=True,
            check_input=False,
            batch_size=batch_size,
        )
        return self._apply_entropy_response_cap(
            gain,
            P,
            P_perf=P_perf,
            P_annot=P_annot,
            C=C,
            response_distribution=response_distribution,
        )

    def _apply_entropy_response_cap(
        self,
        gain: np.ndarray,
        P: np.ndarray,
        *,
        P_perf: np.ndarray | None = None,
        P_annot: np.ndarray | None = None,
        C: np.ndarray | None = None,
        response_distribution: np.ndarray | None = None,
    ) -> np.ndarray:
        if not self.entropy_response_cap or self.gain_type != "entropy":
            return gain
        cap = self._entropy_gain_upper_bound(
            P,
            P_perf=P_perf,
            P_annot=P_annot,
            C=C,
            response_distribution=response_distribution,
        )
        return np.minimum(np.maximum(gain, 0.0), cap)

    def _apply_response_entropy_regularizer(
        self,
        gain: np.ndarray,
        *,
        g: np.ndarray | None,
    ) -> np.ndarray:
        if not self.response_entropy_cap or self.gain_type != "entropy":
            return gain
        if g is None:
            return gain
        cap = self.response_entropy_cap_lambda * self._entropy_bits(g)
        return np.minimum(np.maximum(gain, 0.0), cap)

    def _pi_mixture_weight(
        self,
        *,
        g: np.ndarray,
        K_pair: np.ndarray,
        eps: float = 1e-12,
    ) -> np.ndarray:
        g = np.asarray(g, dtype=float)
        K_pair = np.asarray(K_pair, dtype=float)

        if g.ndim != 3:
            raise ValueError(
                "g must have shape (n_samples, n_draws, n_classes)."
            )
        if K_pair.ndim != 2 or K_pair.shape[1] != g.shape[0]:
            raise ValueError(
                "K_pair must have shape (n_observations, n_samples)."
            )

        K = g.shape[-1]
        h_max = np.log2(K)
        if h_max <= eps:
            return np.zeros(g.shape[:2], dtype=float)

        entropy = self._entropy_bits(g, eps=eps)
        collapse = np.clip(1.0 - entropy / h_max, 0.0, 1.0)

        weights = np.clip(K_pair, 0.0, None)
        weight_sum = weights.sum(axis=0)
        weight_sq_sum = np.sum(weights * weights, axis=0)
        n_eff = np.divide(
            weight_sum * weight_sum,
            np.maximum(weight_sq_sum, eps),
            out=np.zeros_like(weight_sum, dtype=float),
            where=weight_sq_sum > eps,
        )
        shrink = n_eff / np.maximum(n_eff + self.pi_mixture_kappa, eps)
        pi = shrink[:, None] * (collapse ** self.pi_mixture_gamma)
        return np.clip(pi, 0.0, self.pi_mixture_max)

    @staticmethod
    def _pi_mixture_confusion(
        *,
        C_orig: np.ndarray,
        g: np.ndarray,
        pi: np.ndarray,
        eps: float = 1e-12,
    ) -> np.ndarray:
        C_orig = np.asarray(C_orig, dtype=float)
        g = np.asarray(g, dtype=float)
        pi = np.asarray(pi, dtype=float)

        if C_orig.ndim < 4:
            raise ValueError(
                "C_orig must have shape (..., n_classes, n_classes)."
            )
        if g.shape != C_orig.shape[:-2] + (C_orig.shape[-1],):
            raise ValueError("g must match C_orig leading dimensions.")
        if pi.shape != C_orig.shape[:-2]:
            raise ValueError("pi must match C_orig leading dimensions.")

        g = np.clip(g, eps, 1.0)
        g = g / np.maximum(g.sum(axis=-1, keepdims=True), eps)
        pi = np.clip(pi, 0.0, 1.0)
        C_mix = (1.0 - pi[..., None, None]) * C_orig
        C_mix = C_mix + pi[..., None, None] * g[..., None, :]
        C_mix = np.maximum(C_mix, 0.0)
        return C_mix / np.maximum(C_mix.sum(axis=-1, keepdims=True), eps)

    @classmethod
    def _entropy_gain_upper_bound(
        cls,
        P: np.ndarray,
        *,
        P_perf: np.ndarray | None = None,
        P_annot: np.ndarray | None = None,
        C: np.ndarray | None = None,
        response_distribution: np.ndarray | None = None,
        eps: float = 1e-12,
    ) -> np.ndarray:
        r = np.asarray(P, dtype=float)
        if response_distribution is not None:
            response = np.asarray(response_distribution, dtype=float)
            response = np.clip(response, eps, 1.0)
            response = response / np.maximum(
                response.sum(axis=-1, keepdims=True),
                eps,
            )
            r = np.clip(r, eps, 1.0)
            r = r / np.maximum(r.sum(axis=-1, keepdims=True), eps)
            r = cls._broadcast_prior_to_distribution(r=r, q=response)
            return np.minimum(
                cls._entropy_bits(r, eps=eps),
                cls._entropy_bits(response, eps=eps),
            )

        if C is None:
            if P_annot is None:
                raise ValueError(
                    "Entropy response cap requires C, response_distribution, "
                    "or P_annot."
                )
            return cls._entropy_gain_upper_bound(
                r,
                response_distribution=P_annot,
                eps=eps,
            )

        C = np.asarray(C, dtype=float)

        r = np.clip(r, eps, 1.0)
        C = np.clip(C, eps, 1.0)
        r = r / np.maximum(r.sum(axis=-1, keepdims=True), eps)
        C = C / np.maximum(C.sum(axis=-1, keepdims=True), eps)
        r = cls._broadcast_prior_to_confusion(r=r, C=C)

        response = np.einsum("...k,...ky->...y", r, C)
        return np.minimum(
            cls._entropy_bits(r, eps=eps),
            cls._entropy_bits(response, eps=eps),
        )

    @staticmethod
    def _broadcast_prior_to_confusion(
        *,
        r: np.ndarray,
        C: np.ndarray,
    ) -> np.ndarray:
        leading = C.shape[:-2]
        if r.shape[:-1] == leading:
            return r
        n_missing = len(leading) - (r.ndim - 1)
        if n_missing < 0:
            raise ValueError(
                f"Cannot broadcast prior shape {r.shape} to confusion shape {C.shape}."
            )
        r_expanded = r.reshape(r.shape[:-1] + (1,) * n_missing + (r.shape[-1],))
        return np.broadcast_to(r_expanded, leading + (r.shape[-1],))

    @staticmethod
    def _broadcast_prior_to_distribution(
        *,
        r: np.ndarray,
        q: np.ndarray,
    ) -> np.ndarray:
        leading = q.shape[:-1]
        if r.shape[:-1] == leading:
            return r
        n_missing = len(leading) - (r.ndim - 1)
        if n_missing < 0:
            raise ValueError(
                f"Cannot broadcast prior shape {r.shape} to response shape {q.shape}."
            )
        r_expanded = r.reshape(r.shape[:-1] + (1,) * n_missing + (r.shape[-1],))
        return np.broadcast_to(r_expanded, leading + (r.shape[-1],))

    @staticmethod
    def _entropy_bits(P: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
        P = np.asarray(P, dtype=float)
        P = np.clip(P, eps, 1.0)
        P = P / np.maximum(P.sum(axis=-1, keepdims=True), eps)
        return -(P * (np.log(P) / np.log(2.0))).sum(axis=-1)

    def _sample_theta_batch(
        self,
        *,
        alpha: np.ndarray,
        beta: np.ndarray,
        rng: np.random.Generator,
        n_draws: int,
    ) -> np.ndarray:
        alpha = np.asarray(alpha, dtype=float)
        beta = np.asarray(beta, dtype=float)
        if n_draws <= 0:
            raise ValueError("n_draws must be positive.")
        if self.n_mc_samples <= 0:
            return self._theta_point_estimate(
                alpha=alpha,
                beta=beta,
            )[:, None]
        return rng.beta(
            alpha[:, None],
            beta[:, None],
            size=(alpha.shape[0], n_draws),
        ).astype(float)

    def _sample_label_distribution_batch(
        self,
        *,
        gamma: np.ndarray,
        rng: np.random.Generator,
        n_draws: int,
    ) -> np.ndarray:
        gamma = np.asarray(gamma, dtype=float)
        if gamma.ndim not in {2, 3}:
            raise ValueError(
                "gamma must have shape (n_samples, K) or "
                "(n_samples, n_draws, K)."
            )
        if n_draws <= 0:
            raise ValueError("n_draws must be positive.")

        if self._use_mc_label_dirichlet():
            alpha = np.clip(gamma, 1e-12, None)
            if alpha.ndim == 2:
                alpha = alpha[:, None, :]
                if n_draws != 1:
                    alpha = np.repeat(alpha, n_draws, axis=1)
            elif alpha.shape[1] != n_draws:
                raise ValueError("3D gamma must agree with n_draws.")
            x = rng.gamma(shape=alpha, scale=1.0)
            return x / np.maximum(x.sum(axis=-1, keepdims=True), 1e-12)

        mean = gamma / np.maximum(gamma.sum(axis=-1, keepdims=True), 1e-12)
        if mean.ndim == 2:
            return np.repeat(mean[:, None, :], n_draws, axis=1)
        if mean.shape[1] != n_draws:
            raise ValueError("3D gamma must agree with n_draws.")
        return mean

    def _theta_point_estimate(
        self,
        *,
        alpha: np.ndarray,
        beta: np.ndarray,
    ) -> np.ndarray:
        alpha = np.asarray(alpha, dtype=float)
        beta = np.asarray(beta, dtype=float)
        denom = np.maximum(alpha + beta, 1e-12)
        return np.clip(alpha / denom, 0.0, 1.0)

    def _aggregate_gain_draws(self, gain_draws: np.ndarray) -> np.ndarray:
        gain_draws = np.asarray(gain_draws, dtype=float)
        if gain_draws.ndim != 2:
            raise ValueError(
                "gain_draws must have shape (n_samples, n_draws)."
            )
        if self.gain_ucb_quantile is None:
            return gain_draws.mean(axis=1)
        return np.quantile(
            gain_draws,
            self.gain_ucb_quantile,
            axis=1,
        )

    @staticmethod
    def _reduce_topm_vectors_batch(
        *, r: np.ndarray, gamma: np.ndarray, top_m: int, eps: float = 1e-12
    ) -> tuple[np.ndarray, np.ndarray]:
        r = np.asarray(r, dtype=float)
        gamma = np.asarray(gamma, dtype=float)
        if r.ndim == 2:
            if gamma.ndim != 2 or r.shape != gamma.shape:
                raise ValueError(
                    "r and gamma must be 2D with identical shape (n_samples, n_classes)."
                )
            S, K = r.shape
        elif r.ndim == 3:
            if gamma.ndim != 2 or r.shape[0] != gamma.shape[0] or r.shape[2] != gamma.shape[1]:
                raise ValueError(
                    "For 3D r, gamma must have shape (n_samples, n_classes)."
                )
            S, _, K = r.shape
        else:
            raise ValueError(
                "r must be 2D or 3D with classes on the last axis."
            )
        if K < 2:
            raise ValueError("IG requires at least 2 classes.")
        if not (1 <= top_m < K):
            raise ValueError("top_m must satisfy 1 <= top_m < n_classes.")

        axis = r.ndim - 1
        idx_part = np.argpartition(-r, kth=top_m - 1, axis=axis)[..., :top_m]
        r_top_part = np.take_along_axis(r, idx_part, axis=axis)
        order = np.argsort(-r_top_part, axis=axis)
        idx = np.take_along_axis(idx_part, order, axis=axis)

        r_top = np.take_along_axis(r, idx, axis=axis)
        r_other = np.maximum(1.0 - r_top.sum(axis=axis), 0.0)
        r_red = np.concatenate([r_top, r_other[..., None]], axis=axis)
        r_red = np.clip(r_red, eps, 1.0)
        r_red = r_red / np.maximum(r_red.sum(axis=axis, keepdims=True), eps)

        if r.ndim == 2:
            gamma_top = np.take_along_axis(gamma, idx, axis=1)
            gamma_other = np.maximum(
                gamma.sum(axis=1) - gamma_top.sum(axis=1), eps
            )
            gamma_red = np.concatenate(
                [gamma_top, gamma_other[:, None]], axis=1
            )
        else:
            T = r.shape[1]
            gamma_bt = np.broadcast_to(gamma[:, None, :], (S, T, K))
            gamma_top = np.take_along_axis(gamma_bt, idx, axis=2)
            gamma_other = np.maximum(
                gamma_bt.sum(axis=2) - gamma_top.sum(axis=2), eps
            )
            gamma_red = np.concatenate(
                [gamma_top, gamma_other[..., None]], axis=2
            )
        gamma_red = np.clip(gamma_red, eps, None)

        return r_red, gamma_red

    @staticmethod
    def _channel_confusion_from_wrong_dirichlet_batch(
        *,
        gamma: np.ndarray,
        theta: np.ndarray,
        rng: np.random.Generator,
        sample: bool,
        eps: float = 1e-12,
    ) -> np.ndarray:
        gamma = np.asarray(gamma, dtype=float)
        theta = np.asarray(theta, dtype=float)

        if gamma.ndim not in {2, 3}:
            raise ValueError(
                f"gamma must have shape (n_samples, K) or (n_samples, n_draws, K), got {gamma.shape}."
            )
        if theta.ndim != 2:
            raise ValueError(
                f"theta must have shape (n_samples, n_draws), got {theta.shape}."
            )
        if gamma.shape[0] != theta.shape[0]:
            raise ValueError("gamma and theta must agree on n_samples.")

        T = theta.shape[1]
        if gamma.ndim == 2:
            gamma = np.repeat(gamma[:, None, :], T, axis=1)
        elif gamma.shape[1] != T:
            raise ValueError("gamma and theta must agree on n_draws.")

        S, _, K = gamma.shape
        gamma = np.clip(gamma, eps, None)
        theta = np.clip(theta, 0.0, 1.0)

        C = np.zeros((S, T, K, K), dtype=float)
        idx = np.arange(K)
        C[..., idx, idx] = theta[:, :, None]
        off_scale = (1.0 - theta)[:, :, None]

        for z in range(K):
            off_idx = idx != z
            alpha = gamma[:, :, off_idx]
            if sample:
                x = rng.gamma(shape=alpha, scale=1.0)
                off = x / np.maximum(x.sum(axis=-1, keepdims=True), eps)
            else:
                off = alpha / np.maximum(alpha.sum(axis=-1, keepdims=True), eps)
            C[:, :, z, off_idx] = off_scale * off

        return C

    def _mc_draw_count(self) -> int:
        return 1 if self.n_mc_samples <= 0 else self.n_mc_samples

    def _use_mc_label_dirichlet(self) -> bool:
        return self.sample_label_dirichlet and self.n_mc_samples > 0

    @classmethod
    def _full_confusion_dirichlet_prior(
        cls, *, K: int, accuracy_mean: float, row_strength: float
    ) -> np.ndarray:
        if K < 2:
            raise ValueError("K must be >= 2")
        if not (0.0 < accuracy_mean < 1.0):
            raise ValueError("accuracy_mean must be in (0, 1)")
        if row_strength <= 0:
            raise ValueError("row_strength must be > 0")
        off = (1.0 - accuracy_mean) / (K - 1)
        prior_mean = np.full((K, K), off, dtype=float)
        np.fill_diagonal(prior_mean, accuracy_mean)
        return row_strength * prior_mean

    @classmethod
    def _channel_prior_full_confusion_dirichlet_prior(
        cls,
        *,
        alpha: np.ndarray,
        beta: np.ndarray,
        gamma: np.ndarray,
        row_strength: float,
        eps: float = 1e-12,
    ) -> np.ndarray:
        alpha = np.asarray(alpha, dtype=float)
        beta = np.asarray(beta, dtype=float)
        gamma = np.asarray(gamma, dtype=float)

        if alpha.ndim != 1 or beta.ndim != 1:
            raise ValueError("alpha and beta must have shape (n_candidates,).")
        if gamma.ndim != 2:
            raise ValueError("gamma must have shape (n_candidates, K).")
        if alpha.shape != beta.shape or alpha.shape[0] != gamma.shape[0]:
            raise ValueError("alpha, beta, and gamma must agree on candidates.")
        if gamma.shape[1] < 2:
            raise ValueError("K must be >= 2")
        if row_strength <= 0:
            raise ValueError("row_strength must be > 0")

        theta = alpha / np.maximum(alpha + beta, eps)
        theta = np.clip(theta, eps, 1.0 - eps)
        g = gamma / np.maximum(gamma.sum(axis=1, keepdims=True), eps)
        g = np.clip(g, eps, 1.0)
        g = g / np.maximum(g.sum(axis=1, keepdims=True), eps)

        n_candidates, K = gamma.shape
        prior_mean = np.empty((n_candidates, K, K), dtype=float)
        idx = np.arange(K)
        for z in range(K):
            off = g.copy()
            #prior_mean[:, z, :] = g
            #off = np.full_like(g, fill_value=1/K, dtype=float)
            off[:, z] = 0.0
            off = off / np.maximum(off.sum(axis=1, keepdims=True), eps)
            prior_mean[:, z, :] = (1.0 - theta)[:, None] * off
            prior_mean[:, z, z] = theta

        prior_mean = prior_mean / np.maximum(
            prior_mean.sum(axis=2, keepdims=True),
            eps,
        )
        prior_mean = np.clip(prior_mean, eps, None)
        prior_mean = prior_mean / np.maximum(
            prior_mean.sum(axis=2, keepdims=True),
            eps,
        )
        return row_strength * prior_mean

    # -------------------------
    # Gamma resolution
    # -------------------------
    @staticmethod
    def _mix_with_global_sample_kernel(
        K_local: np.ndarray,
        *,
        lam: float,
    ) -> np.ndarray:
        K_local = np.asarray(K_local, dtype=float)
        if K_local.ndim != 2:
            raise ValueError(
                f"K_local must be 2D, got shape {K_local.shape}."
            )
        if not (0.0 <= lam <= 1.0):
            raise ValueError("lam must be in [0, 1].")
        if lam == 0.0:
            return K_local
        if lam == 1.0:
            return np.ones_like(K_local, dtype=float)
        return lam + (1.0 - lam) * K_local

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
    def parzen_beta_posterior(
        K: np.ndarray,
        p: np.ndarray,
        *,
        alpha0: float = 1.0,
        beta0: float = 1.0,
        use_ess: bool = False,
        tau: float = 1.0,
        eps: float = 1e-12,
    ):
        K = np.asarray(K, dtype=float)
        p = np.asarray(p, dtype=float)

        if K.ndim != 2:
            raise ValueError(f"K must be 2D, got shape {K.shape}")
        if p.ndim != 1:
            raise ValueError(f"p must be 1D, got shape {p.shape}")
        if K.shape[0] != p.shape[0]:
            raise ValueError(
                f"Shape mismatch: K rows {K.shape[0]} vs p {p.shape[0]}"
            )
        if alpha0 <= 0 or beta0 <= 0:
            raise ValueError("alpha0 and beta0 must be > 0")
        if tau <= 0:
            raise ValueError("tau must be > 0")

        p = np.clip(p, 0.0, 1.0)
        mass = K.sum(axis=0)
        s = p @ K
        f = (1.0 - p) @ K
        mu = np.where(mass > eps, s / np.maximum(mass, eps), 0.5)

        if not use_ess:
            alpha = alpha0 + s
            beta = beta0 + f
            info = {"mu": mu, "mass": mass, "s": s, "f": f}
            return alpha, beta, info

        m2 = (K**2).sum(axis=0)
        n_eff = (mass**2) / np.maximum(m2, eps)
        conc = tau * n_eff

        alpha = alpha0 + conc * mu
        beta = beta0 + conc * (1.0 - mu)

        info = {"mu": mu, "mass": mass, "n_eff": n_eff}
        return alpha, beta, info

    @staticmethod
    def parzen_dirichlet_posterior(
        K: np.ndarray,
        Y: np.ndarray,
        *,
        gamma0: np.ndarray,
        use_ess: bool = False,
        tau: float = 1.0,
        eps: float = 1e-12,
    ):
        K = np.asarray(K, dtype=float)
        Y = np.asarray(Y, dtype=float)
        gamma0 = np.asarray(gamma0, dtype=float)

        if K.ndim != 2:
            raise ValueError(f"K must be 2D, got shape {K.shape}")
        if Y.ndim != 2:
            raise ValueError(f"Y must be 2D, got shape {Y.shape}")
        if K.shape[0] != Y.shape[0]:
            raise ValueError(
                f"Shape mismatch: K rows {K.shape[0]} vs Y rows {Y.shape[0]}"
            )
        if Y.shape[1] != gamma0.shape[0]:
            raise ValueError(
                f"Y classes {Y.shape[1]} must equal gamma0 length {gamma0.shape[0]}"
            )
        if np.any(gamma0 <= 0):
            raise ValueError("gamma0 entries must be > 0")
        if tau <= 0:
            raise ValueError("tau must be > 0")

        counts = K.T @ Y
        mass = counts.sum(axis=1)
        mu = counts / np.maximum(mass[:, None], eps)

        if not use_ess:
            gamma = gamma0[None, :] + counts
            info = {"counts": counts, "mass": mass, "mu": mu}
            return gamma, info

        k_mass = K.sum(axis=0)
        k_m2 = (K**2).sum(axis=0)
        n_eff = (k_mass**2) / np.maximum(k_m2, eps)
        conc = tau * n_eff
        gamma = gamma0[None, :] + conc[:, None] * mu
        info = {"mass": mass, "mu": mu, "n_eff": n_eff}
        return gamma, info

    @staticmethod
    def full_confusion_dirichlet_posterior(
        K: np.ndarray,
        Y: np.ndarray,
        *,
        row_responsibility: np.ndarray,
        delta0: np.ndarray,
        use_ess: bool = False,
        tau: float = 1.0,
        eps: float = 1e-12,
    ) -> np.ndarray:
        """Vectorized row-wise Dirichlet posterior for full confusion matrices."""
        K = np.asarray(K, dtype=float)
        Y = np.asarray(Y, dtype=float)
        row_responsibility = np.asarray(row_responsibility, dtype=float)
        delta0 = np.asarray(delta0, dtype=float)

        if K.ndim != 2:
            raise ValueError(f"K must be 2D, got shape {K.shape}")
        if Y.ndim != 2:
            raise ValueError(f"Y must be 2D, got shape {Y.shape}")
        if row_responsibility.ndim != 2:
            raise ValueError(
                "row_responsibility must be 2D with shape (n_obs, K)."
            )
        if delta0.ndim == 2:
            if delta0.shape[0] != delta0.shape[1]:
                raise ValueError("delta0 must have shape (K, K).")
            delta0_by_candidate = delta0[None, :, :]
        elif delta0.ndim == 3:
            if (
                delta0.shape[0] != K.shape[1]
                or delta0.shape[1] != delta0.shape[2]
            ):
                raise ValueError(
                    "candidate-specific delta0 must have shape "
                    "(n_candidates, K, K)."
                )
            delta0_by_candidate = delta0
        else:
            raise ValueError(
                "delta0 must have shape (K, K) or (n_candidates, K, K)."
            )
        if K.shape[0] != Y.shape[0] or K.shape[0] != row_responsibility.shape[0]:
            raise ValueError(
                "K, Y, and row_responsibility must have the same number of rows."
            )
        if (
            Y.shape[1] != delta0_by_candidate.shape[1]
            or row_responsibility.shape[1] != delta0_by_candidate.shape[1]
        ):
            raise ValueError(
                "Y, row_responsibility, and delta0 must agree on n_classes."
            )
        if np.any(delta0 <= 0):
            raise ValueError("delta0 entries must be > 0")
        if tau <= 0:
            raise ValueError("tau must be > 0")

        n_candidates = K.shape[1]
        if K.shape[0] == 0:
            if delta0_by_candidate.shape[0] == 1:
                return np.broadcast_to(
                    delta0_by_candidate,
                    (
                        n_candidates,
                        delta0_by_candidate.shape[1],
                        delta0_by_candidate.shape[2],
                    ),
                ).copy()
            return delta0_by_candidate.copy()

        counts = np.einsum(
            "ns,nz,ny->szy",
            K,
            row_responsibility,
            Y,
            optimize=True,
        )
        if not use_ess:
            return delta0_by_candidate + counts

        mass = counts.sum(axis=2)
        mu = counts / np.maximum(mass[:, :, None], eps)
        k_m2 = np.einsum(
            "ns,nz->sz",
            K * K,
            row_responsibility * row_responsibility,
            optimize=True,
        )
        n_eff = (mass * mass) / np.maximum(k_m2, eps)
        conc = tau * n_eff
        return delta0_by_candidate + conc[:, :, None] * mu
