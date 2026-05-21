"""Interactive toy-bed visualizations for a local kNN annotator scorer.

The code in this module is intentionally notebook-friendly, but lives outside
the notebook so the toy bed can stay readable.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta as beta_dist
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import pairwise_distances
from sklearn.neural_network import MLPClassifier

from src.scorer import PairScorer


def make_oracle_probabilities(y: np.ndarray, n_classes: int) -> np.ndarray:
    """Return one-hot true-label probabilities."""
    y = np.asarray(y, dtype=int)
    P = np.zeros((len(y), n_classes), dtype=float)
    P[np.arange(len(y)), y] = 1.0
    return P


def make_noisy_oracle_probabilities(
    y: np.ndarray,
    n_classes: int,
    confidence: float = 0.8,
) -> np.ndarray:
    """Return smoothed true-label probabilities with uniform off-class mass."""
    y = np.asarray(y, dtype=int)
    confidence = float(np.clip(confidence, 1.0 / n_classes, 1.0))
    if n_classes <= 1:
        return np.ones((len(y), 1), dtype=float)
    P = np.full(
        (len(y), n_classes),
        (1.0 - confidence) / (n_classes - 1),
        dtype=float,
    )
    P[np.arange(len(y)), y] = confidence
    return P


@dataclass
class QueryPosterior:
    alpha: np.ndarray
    beta: np.ndarray
    bandwidth_sigma: np.ndarray
    effective_sample_size: np.ndarray
    evidence_mass: np.ndarray
    evidence_confidence: np.ndarray
    evidence_strength: np.ndarray

    @property
    def mean(self) -> np.ndarray:
        return self.alpha / (self.alpha + self.beta)

    def quantile(self, q: float) -> np.ndarray:
        return beta_dist.ppf(float(q), self.alpha, self.beta)


@dataclass
class ProbabilitySourceResult:
    train_probabilities: np.ndarray
    grid_probabilities: np.ndarray | None
    status: str


class BudgetAwareKnnSoftAccuracyScorer(PairScorer):
    """Budget-aware kernel estimator of local annotator correctness.

    The scorer estimates ``P(annotator m is correct near x)`` from RBF-weighted
    observed labels of annotator ``m``. Bandwidth can be scheduled directly or
    chosen per query point to target a local effective sample size.
    """

    def __init__(
        self,
        *,
        bandwidth_mode: str = "global_sigma",
        bandwidth_scope: str = "per_annotator",
        evidence_mode: str = "normalized_strength",
        correctness_mode: str = "hard_pseudo_entropy",
        sigma_min: float = 0.15,
        sigma_max: float = 1.5,
        ess_min: float = 3.0,
        ess_max: float = 30.0,
        strength_min: float = 2.0,
        strength_max: float = 30.0,
        alpha0: float = 1.0,
        beta0: float = 1.0,
        score_mode: str = "quantile",
        quantile: float = 0.9,
        missing_label=np.nan,
        exclude_self: bool = True,
        metric: str = "euclidean",
        random_state: int | None = None,
        ess_tolerance: float = 1e-3,
        ess_max_iter: int = 48,
    ):
        if bandwidth_mode not in {"global_sigma", "adaptive_ess"}:
            raise ValueError("bandwidth_mode must be one of {'global_sigma', 'adaptive_ess'}.")
        if bandwidth_scope not in {"per_annotator", "all_annotators"}:
            raise ValueError("bandwidth_scope must be one of {'per_annotator', 'all_annotators'}.")
        if evidence_mode not in {"normalized_strength", "raw_mass"}:
            raise ValueError("evidence_mode must be one of {'normalized_strength', 'raw_mass'}.")
        if correctness_mode not in {"soft_probability", "hard_pseudo_entropy"}:
            raise ValueError(
                "correctness_mode must be one of {'soft_probability', 'hard_pseudo_entropy'}."
            )
        if sigma_min <= 0 or sigma_max <= 0 or sigma_min > sigma_max:
            raise ValueError("Require 0 < sigma_min <= sigma_max.")
        if ess_min <= 0 or ess_max <= 0 or ess_min > ess_max:
            raise ValueError("Require 0 < ess_min <= ess_max.")
        if strength_min <= 0 or strength_max <= 0 or strength_min > strength_max:
            raise ValueError("Require 0 < strength_min <= strength_max.")
        if alpha0 <= 0 or beta0 <= 0:
            raise ValueError("alpha0 and beta0 must be positive.")
        if score_mode not in {"mean", "quantile", "thompson"}:
            raise ValueError("score_mode must be one of {'mean', 'quantile', 'thompson'}.")
        if not (0.0 < quantile < 1.0):
            raise ValueError("quantile must be in (0, 1).")
        if ess_tolerance <= 0:
            raise ValueError("ess_tolerance must be positive.")
        if ess_max_iter <= 0:
            raise ValueError("ess_max_iter must be positive.")

        self.bandwidth_mode = str(bandwidth_mode)
        self.bandwidth_scope = str(bandwidth_scope)
        self.evidence_mode = str(evidence_mode)
        self.correctness_mode = str(correctness_mode)
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.ess_min = float(ess_min)
        self.ess_max = float(ess_max)
        self.strength_min = float(strength_min)
        self.strength_max = float(strength_max)
        self.alpha0 = float(alpha0)
        self.beta0 = float(beta0)
        self.score_mode = str(score_mode)
        self.quantile = float(quantile)
        self.missing_label = missing_label
        self.exclude_self = bool(exclude_self)
        self.metric = str(metric)
        self.rng = np.random.default_rng(random_state)
        self.ess_tolerance = float(ess_tolerance)
        self.ess_max_iter = int(ess_max_iter)

        self.last_progress_targets_: np.ndarray | None = None
        self.last_observed_counts_: np.ndarray | None = None

    def _compute(
        self,
        X,
        y,
        sample_indices,
        annotator_indices,
        available_mask,
        *args,
        **kwargs,
    ):
        class_probabilities = np.asarray(kwargs["class_probabilities"], dtype=float)
        remaining_budget = kwargs.get("remaining_budget", None)
        X = np.asarray(X)
        y = np.asarray(y)

        sample_indices = np.asarray(sample_indices, dtype=int)
        annotator_indices = np.asarray(annotator_indices, dtype=int)
        observed_all = self._observed_mask(y)
        observed_counts = observed_all.sum(axis=0)
        progress_targets = np.asarray(
            [
                self._progress_for_annotator(
                    observed_all=observed_all,
                    annotator_index=int(annotator_index),
                    remaining_budget=remaining_budget,
                )
                for annotator_index in annotator_indices
            ],
            dtype=float,
        )
        self.last_progress_targets_ = progress_targets.copy()
        self.last_observed_counts_ = observed_counts.copy()

        X_flat = X.reshape(X.shape[0], -1)
        utilities = np.empty((len(sample_indices), len(annotator_indices)), dtype=float)
        for local_j, annotator_index in enumerate(annotator_indices):
            posterior = self.posterior_for_query_points(
                X_query=X_flat[sample_indices],
                X=X_flat,
                y=y,
                annotator_index=int(annotator_index),
                class_probabilities=class_probabilities,
                remaining_budget=remaining_budget,
                query_sample_indices=sample_indices,
            )
            utilities[:, local_j] = self.score_posterior(posterior)

        if available_mask is not None:
            utilities = utilities.copy()
            utilities[~available_mask] = -np.inf
        return utilities

    def posterior_for_query_points(
        self,
        *,
        X_query: np.ndarray,
        X: np.ndarray,
        y: np.ndarray,
        annotator_index: int,
        class_probabilities: np.ndarray,
        remaining_budget=None,
        query_sample_indices: np.ndarray | None = None,
    ) -> QueryPosterior:
        """Compute Beta posterior parameters for arbitrary query coordinates."""
        X_query = np.asarray(X_query, dtype=float)
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        class_probabilities = np.asarray(class_probabilities, dtype=float)

        if y.ndim != 2:
            raise ValueError("y must have shape (n_samples, n_annotators).")
        if class_probabilities.shape[0] != y.shape[0]:
            raise ValueError("class_probabilities must have one row per sample.")
        if not (0 <= annotator_index < y.shape[1]):
            raise IndexError("annotator_index is out of bounds.")

        observed_all = self._observed_mask(y)
        evidence_indices = np.flatnonzero(observed_all[:, annotator_index])
        progress = self._progress_for_annotator(
            observed_all=observed_all,
            annotator_index=annotator_index,
            remaining_budget=remaining_budget,
        )
        scheduled_sigma = self._geometric_schedule(
            self.sigma_max,
            self.sigma_min,
            progress,
        )
        scheduled_ess = self._geometric_schedule(
            self.ess_max,
            self.ess_min,
            progress,
        )
        scheduled_strength = self._geometric_schedule(
            self.strength_min,
            self.strength_max,
            progress,
        )

        n_query = X_query.shape[0]
        alpha = np.full(n_query, self.alpha0, dtype=float)
        beta = np.full(n_query, self.beta0, dtype=float)
        bandwidth_sigma = np.full(n_query, np.nan, dtype=float)
        effective_sample_size = np.zeros(n_query, dtype=float)
        evidence_mass = np.zeros(n_query, dtype=float)
        evidence_confidence = np.zeros(n_query, dtype=float)
        evidence_strength = np.full(n_query, scheduled_strength, dtype=float)

        reference_indices = self._bandwidth_reference_indices(
            observed_all,
            annotator_index=annotator_index,
        )
        if reference_indices.size == 0:
            return QueryPosterior(
                alpha=alpha,
                beta=beta,
                bandwidth_sigma=bandwidth_sigma,
                effective_sample_size=effective_sample_size,
                evidence_mass=evidence_mass,
                evidence_confidence=evidence_confidence,
                evidence_strength=evidence_strength,
            )

        if self.correctness_mode == "hard_pseudo_entropy":
            reference_confidence = self._normalized_entropy_confidence(
                class_probabilities[reference_indices]
            )
        else:
            reference_confidence = np.ones(reference_indices.size, dtype=float)

        if evidence_indices.size > 0:
            observed_labels = y[evidence_indices, annotator_index].astype(int)
            n_classes = class_probabilities.shape[1]
            if np.any(observed_labels < 0) or np.any(observed_labels >= n_classes):
                raise ValueError("Observed labels contain values outside the class range.")
            if self.correctness_mode == "soft_probability":
                correctness = class_probabilities[evidence_indices, observed_labels]
                evidence_item_confidence = np.ones(evidence_indices.size, dtype=float)
            else:
                pseudo_labels = np.argmax(class_probabilities[evidence_indices], axis=1)
                correctness = (observed_labels == pseudo_labels).astype(float)
                evidence_item_confidence = self._normalized_entropy_confidence(
                    class_probabilities[evidence_indices]
                )
        else:
            correctness = np.empty(0, dtype=float)
            evidence_item_confidence = np.empty(0, dtype=float)
        reference_distances = pairwise_distances(
            X_query,
            X[reference_indices],
            metric=self.metric,
        )
        if evidence_indices.size > 0:
            evidence_distances = pairwise_distances(
                X_query,
                X[evidence_indices],
                metric=self.metric,
            )
        else:
            evidence_distances = np.empty((n_query, 0), dtype=float)
        if self.exclude_self and query_sample_indices is not None:
            query_sample_indices = np.asarray(query_sample_indices, dtype=int)
            evidence_position = {idx: pos for pos, idx in enumerate(evidence_indices)}
            for row, sample_index in enumerate(query_sample_indices):
                pos = evidence_position.get(int(sample_index))
                if pos is not None:
                    evidence_distances[row, pos] = np.inf

        for row in range(n_query):
            if self.bandwidth_mode == "global_sigma":
                sigma = scheduled_sigma
            else:
                target_ess = min(float(scheduled_ess), float(reference_indices.size))
                sigma = self._sigma_for_target_ess(
                    reference_distances[row],
                    target_ess,
                    base_weights=reference_confidence,
                )
            bandwidth_sigma[row] = sigma
            if not np.isfinite(sigma) or sigma <= 0:
                continue

            ref_weights = self._rbf_weights(reference_distances[row], sigma) * reference_confidence
            effective_sample_size[row] = self._effective_sample_size(ref_weights)

            if evidence_indices.size == 0:
                continue
            finite_evidence = np.isfinite(evidence_distances[row])
            if not np.any(finite_evidence):
                continue
            spatial_weights = self._rbf_weights(evidence_distances[row, finite_evidence], sigma)
            confidence = evidence_item_confidence[finite_evidence]
            weights = spatial_weights * confidence
            s = correctness[finite_evidence]
            spatial_mass = float(np.sum(spatial_weights))
            mass = float(np.sum(weights))
            evidence_mass[row] = mass
            if spatial_mass > 0:
                evidence_confidence[row] = float(mass / spatial_mass)
            if mass <= 0:
                continue

            if self.evidence_mode == "raw_mass":
                alpha[row] += float(np.sum(weights * s))
                beta[row] += float(np.sum(weights * (1.0 - s)))
            else:
                mu = float(np.sum(weights * s) / mass)
                effective_strength = scheduled_strength * evidence_confidence[row]
                evidence_strength[row] = effective_strength
                alpha[row] += effective_strength * mu
                beta[row] += effective_strength * (1.0 - mu)

        return QueryPosterior(
            alpha=alpha,
            beta=beta,
            bandwidth_sigma=bandwidth_sigma,
            effective_sample_size=effective_sample_size,
            evidence_mass=evidence_mass,
            evidence_confidence=evidence_confidence,
            evidence_strength=evidence_strength,
        )

    def score_posterior(self, posterior: QueryPosterior) -> np.ndarray:
        if self.score_mode == "mean":
            return posterior.mean
        if self.score_mode == "quantile":
            return posterior.quantile(self.quantile)
        if self.score_mode == "thompson":
            return self.rng.beta(posterior.alpha, posterior.beta)
        raise RuntimeError("invalid score_mode")

    def _compute_progress_targets(
        self,
        *,
        observed_counts,
        annotator_indices,
        remaining_budget,
    ) -> np.ndarray:
        annotator_indices = np.asarray(annotator_indices, dtype=int)
        observed_counts = np.asarray(observed_counts, dtype=float)
        if remaining_budget is None:
            return np.zeros(len(annotator_indices), dtype=float)

        remaining_budget = np.asarray(remaining_budget, dtype=float)
        if remaining_budget.ndim == 0:
            progress = self._safe_progress(
                float(np.sum(observed_counts)),
                max(float(remaining_budget), 0.0),
            )
            return np.full(len(annotator_indices), progress, dtype=float)
        if remaining_budget.shape != observed_counts.shape:
            raise ValueError("remaining_budget must be scalar or per annotator.")

        out = np.empty(len(annotator_indices), dtype=float)
        for local_j, annotator_index in enumerate(annotator_indices):
            out[local_j] = self._safe_progress(
                float(observed_counts[annotator_index]),
                max(float(remaining_budget[annotator_index]), 0.0),
            )
        return out

    def _progress_for_annotator(
        self,
        *,
        observed_all: np.ndarray,
        annotator_index: int,
        remaining_budget,
    ) -> float:
        if self.bandwidth_scope == "all_annotators":
            n_observed = float(np.sum(observed_all.any(axis=1)))
        else:
            n_observed = float(np.sum(observed_all[:, annotator_index]))
        if remaining_budget is None:
            return 0.0
        remaining_budget = np.asarray(remaining_budget, dtype=float)
        if remaining_budget.ndim == 0:
            rem = max(float(remaining_budget), 0.0)
        elif remaining_budget.shape == (observed_all.shape[1],):
            if self.bandwidth_scope == "all_annotators":
                rem = max(float(np.sum(remaining_budget)), 0.0)
            else:
                rem = max(float(remaining_budget[annotator_index]), 0.0)
        else:
            raise ValueError(
                "remaining_budget must be scalar or an array with one value per annotator."
            )
        return self._safe_progress(n_observed, rem)

    @staticmethod
    def _safe_progress(n_observed: float, remaining_budget: float) -> float:
        denom = n_observed + remaining_budget
        if denom <= 0:
            return 1.0
        return float(n_observed / denom)

    @staticmethod
    def _geometric_schedule(start: float, stop: float, progress: float) -> float:
        progress = float(np.clip(progress, 0.0, 1.0))
        return float(start * (stop / start) ** progress)

    def _bandwidth_reference_indices(
        self,
        observed_all: np.ndarray,
        *,
        annotator_index: int,
    ) -> np.ndarray:
        if self.bandwidth_scope == "per_annotator":
            return np.flatnonzero(observed_all[:, annotator_index])
        return np.flatnonzero(observed_all.any(axis=1))

    @staticmethod
    def _rbf_weights(distances: np.ndarray, sigma: float) -> np.ndarray:
        distances = np.asarray(distances, dtype=float)
        out = np.zeros_like(distances, dtype=float)
        finite = np.isfinite(distances)
        if not finite.any():
            return out
        out[finite] = np.exp(-(distances[finite] ** 2) / (2.0 * float(sigma) ** 2))
        return out

    @staticmethod
    def _effective_sample_size(weights: np.ndarray) -> float:
        weights = np.asarray(weights, dtype=float)
        mass = float(np.sum(weights))
        sq_mass = float(np.sum(weights * weights))
        if mass <= 0 or sq_mass <= 0:
            return 0.0
        return float((mass * mass) / sq_mass)

    @staticmethod
    def _normalized_entropy_confidence(probabilities: np.ndarray) -> np.ndarray:
        probabilities = np.asarray(probabilities, dtype=float)
        if probabilities.ndim != 2:
            raise ValueError("probabilities must have shape (n_samples, n_classes).")
        n_classes = probabilities.shape[1]
        if n_classes <= 1:
            return np.ones(probabilities.shape[0], dtype=float)
        P = np.clip(probabilities, 0.0, None)
        row_sum = P.sum(axis=1, keepdims=True)
        valid = row_sum[:, 0] > 0
        P_norm = np.full_like(P, 1.0 / n_classes, dtype=float)
        P_norm[valid] = P[valid] / row_sum[valid]
        safe_P = np.clip(P_norm, 1e-12, 1.0)
        entropy = -np.sum(P_norm * np.log(safe_P), axis=1)
        confidence = 1.0 - entropy / np.log(float(n_classes))
        return np.clip(confidence, 0.0, 1.0)

    def _sigma_for_target_ess(
        self,
        distances: np.ndarray,
        target_ess: float,
        base_weights: np.ndarray | None = None,
    ) -> float:
        distances = np.asarray(distances, dtype=float)
        finite = np.isfinite(distances)
        if base_weights is None:
            base_weights = np.ones_like(distances, dtype=float)
        else:
            base_weights = np.asarray(base_weights, dtype=float)
            if base_weights.shape != distances.shape:
                raise ValueError("base_weights must have the same shape as distances.")
        distances = distances[finite]
        base_weights = np.clip(base_weights[finite], 0.0, None)
        positive = base_weights > 0
        distances = distances[positive]
        base_weights = base_weights[positive]
        n_ref = distances.size
        if n_ref == 0:
            return np.nan
        if n_ref == 1:
            return self.sigma_min

        target_ess = float(np.clip(target_ess, 1.0, n_ref))
        if target_ess <= 1.0 + self.ess_tolerance:
            return self.sigma_min

        lo = self.sigma_min
        hi = self.sigma_max
        hi_ess = self._effective_sample_size(self._rbf_weights(distances, hi) * base_weights)
        if hi_ess < target_ess:
            for _ in range(32):
                hi *= 2.0
                hi_ess = self._effective_sample_size(
                    self._rbf_weights(distances, hi) * base_weights
                )
                if hi_ess >= target_ess or hi > 1e6:
                    break
        if hi_ess < target_ess:
            return hi

        for _ in range(self.ess_max_iter):
            mid = 0.5 * (lo + hi)
            mid_ess = self._effective_sample_size(
                self._rbf_weights(distances, mid) * base_weights
            )
            if abs(mid_ess - target_ess) <= self.ess_tolerance:
                return mid
            if mid_ess < target_ess:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    def _observed_mask(self, labels: np.ndarray) -> np.ndarray:
        labels = np.asarray(labels)
        try:
            if bool(np.isnan(self.missing_label)):
                return ~np.isnan(labels)
        except TypeError:
            pass
        return labels != self.missing_label


class ToyKnnScorerWidget:
    """Interactive ipywidgets/ipympl inspector for the toy kNN scorer."""

    def __init__(
        self,
        *,
        X: np.ndarray | None = None,
        y_true: np.ndarray | None = None,
        sim: Any,
        data: Any | None = None,
        n_cols: int = 3,
        missing_label=np.nan,
    ):
        if data is not None:
            X = data.X_train
            y_true = data.y_train
        if X is None or y_true is None:
            raise ValueError("Pass either data=... or both X=... and y_true=....")

        self.X = np.asarray(X, dtype=float)
        self.y_true = np.asarray(y_true, dtype=int)
        self.sim = sim
        self.n_samples, self.n_features = self.X.shape
        if self.n_features != 2:
            raise ValueError("ToyKnnScorerWidget expects 2D coordinates.")
        self.n_annotators = int(sim.z.shape[1])
        self.n_classes = int(np.max(self.y_true)) + 1
        self.n_cols = int(n_cols)
        self.missing_label = missing_label
        self.Y_obs = np.full(sim.z.shape, np.nan)
        self._thompson_seed = 0
        self.cycle_history: list[dict[str, Any]] = []
        self._cycle_rng = None
        self._cycle_rng_seed = None
        self._cycle_message = ""
        self._last_cycle_pair: tuple[int, int] | None = None

        self._class_colors = plt.get_cmap("tab10")(
            np.arange(max(self.n_classes, 10))
        )
        self._axes_to_annotator: dict[Any, int] = {}
        self._click_cid = None

        self._build_controls()
        self.fig = None
        self.axes = []
        self.status = widgets.HTML()
        self.history = widgets.HTML()
        self.out = widgets.Output()
        self.layout = widgets.VBox([self.controls, self.out, self.status, self.history])

    def display(self):
        """Return the widget layout for notebook display."""
        self.redraw()
        return self.layout

    def _build_controls(self):
        self.reward_source = widgets.Dropdown(
            options=[
                ("true labels", "oracle"),
                ("true labels + uniform noise", "noisy_oracle"),
                ("MLP classifier", "classifier"),
            ],
            value="oracle",
            description="reward",
        )
        self.surface_mode = widgets.Dropdown(
            options=[
                ("utility", "utility"),
                ("posterior mean", "posterior_mean"),
                ("bandwidth sigma", "bandwidth_sigma"),
                ("effective sample size", "effective_sample_size"),
                ("evidence mass", "evidence_mass"),
                ("evidence confidence", "evidence_confidence"),
                ("true annotator accuracy", "true_accuracy"),
            ],
            value="utility",
            description="surface",
        )
        self.score_mode = widgets.Dropdown(
            options=["mean", "quantile", "thompson"],
            value="quantile",
            description="score",
        )
        self.bandwidth_mode = widgets.Dropdown(
            options=[
                ("global sigma", "global_sigma"),
                ("adaptive ESS", "adaptive_ess"),
            ],
            value="global_sigma",
            description="bw mode",
        )
        self.bandwidth_scope = widgets.Dropdown(
            options=[
                ("per annotator", "per_annotator"),
                ("all annotators", "all_annotators"),
            ],
            value="per_annotator",
            description="bw scope",
        )
        self.evidence_mode = widgets.Dropdown(
            options=[
                ("normalized strength", "normalized_strength"),
                ("raw mass", "raw_mass"),
            ],
            value="normalized_strength",
            description="evidence",
        )
        self.correctness_mode = widgets.Dropdown(
            options=[
                ("hard pseudo + entropy", "hard_pseudo_entropy"),
                ("soft probability", "soft_probability"),
            ],
            value="hard_pseudo_entropy",
            description="correct",
        )
        self.sigma_min = widgets.FloatSlider(
            value=0.15,
            min=0.01,
            max=3.0,
            step=0.01,
            readout_format=".2f",
            description="sigma min",
        )
        self.sigma_max = widgets.FloatSlider(
            value=1.5,
            min=0.01,
            max=5.0,
            step=0.01,
            readout_format=".2f",
            description="sigma max",
        )
        self.ess_min = widgets.FloatSlider(
            value=3.0,
            min=1.0,
            max=max(float(self.n_samples), 1.0),
            step=1.0,
            readout_format=".0f",
            description="ESS min",
        )
        self.ess_max = widgets.FloatSlider(
            value=min(30.0, float(self.n_samples)),
            min=1.0,
            max=max(float(self.n_samples), 1.0),
            step=1.0,
            readout_format=".0f",
            description="ESS max",
        )
        self.strength_min = widgets.FloatSlider(
            value=2.0,
            min=0.1,
            max=100.0,
            step=0.5,
            readout_format=".1f",
            description="str min",
        )
        self.strength_max = widgets.FloatSlider(
            value=30.0,
            min=0.1,
            max=200.0,
            step=0.5,
            readout_format=".1f",
            description="str max",
        )
        self.alpha0 = widgets.FloatSlider(
            value=1.0,
            min=0.1,
            max=20.0,
            step=0.1,
            readout_format=".1f",
            description="alpha0",
        )
        self.beta0 = widgets.FloatSlider(
            value=1.0,
            min=0.1,
            max=20.0,
            step=0.1,
            readout_format=".1f",
            description="beta0",
        )
        self.quantile = widgets.FloatSlider(
            value=0.9,
            min=0.05,
            max=0.99,
            step=0.01,
            readout_format=".2f",
            description="quantile",
        )
        self.exclude_self = widgets.Checkbox(value=True, description="exclude self")
        self.remaining_budget = widgets.IntSlider(
            value=200,
            min=0,
            max=int(self.n_samples * self.n_annotators),
            step=1,
            description="budget",
        )
        self.grid_resolution = widgets.IntSlider(
            value=80,
            min=30,
            max=160,
            step=10,
            description="grid",
        )
        self.oracle_confidence = widgets.FloatSlider(
            value=0.8,
            min=1.0 / self.n_classes,
            max=1.0,
            step=0.01,
            readout_format=".2f",
            description="oracle conf",
        )
        self.min_votes_for_classifier = widgets.IntSlider(
            value=1,
            min=1,
            max=self.n_annotators,
            step=1,
            description="min votes",
        )
        self.hidden_layer_size = widgets.IntSlider(
            value=32,
            min=4,
            max=128,
            step=4,
            description="MLP width",
        )
        self.mlp_alpha = widgets.FloatLogSlider(
            value=1e-3,
            base=10,
            min=-6,
            max=0,
            step=0.5,
            readout_format=".1e",
            description="MLP alpha",
        )
        self.mlp_max_iter = widgets.IntSlider(
            value=500,
            min=50,
            max=2000,
            step=50,
            description="MLP iter",
        )
        self.classifier_seed = widgets.IntText(value=0, description="MLP seed")
        self.classifier_view_class = widgets.Dropdown(
            options=[(f"class {c}", c) for c in range(self.n_classes)],
            value=0,
            description="P class",
        )
        self.initial_per_annotator = widgets.IntSlider(
            value=3,
            min=0,
            max=self.n_samples,
            step=1,
            description="init/ann",
        )
        self.init_seed = widgets.IntText(value=0, description="init seed")
        self.show_other_observed = widgets.Checkbox(value=True, description="other marks")
        self.show_latent = widgets.Checkbox(value=False, description="show latent")

        self.clear_button = widgets.Button(description="clear")
        self.random_init_button = widgets.Button(description="random init")
        self.resample_button = widgets.Button(description="resample Thompson")
        self.cycle_step_button = widgets.Button(description="select next")
        self.auto_run_button = widgets.Button(description="auto run")
        self.clear_history_button = widgets.Button(description="clear history")
        self.cycle_steps = widgets.IntSlider(
            value=10,
            min=1,
            max=int(self.n_samples * self.n_annotators),
            step=1,
            description="cycle steps",
        )
        self.cycle_seed = widgets.IntText(value=0, description="cycle seed")
        self.clear_button.on_click(self._clear_clicked)
        self.random_init_button.on_click(self._random_init_clicked)
        self.resample_button.on_click(self._resample_clicked)
        self.cycle_step_button.on_click(self._cycle_step_clicked)
        self.auto_run_button.on_click(self._auto_run_clicked)
        self.clear_history_button.on_click(self._clear_history_clicked)

        watched = [
            self.reward_source,
            self.surface_mode,
            self.score_mode,
            self.bandwidth_mode,
            self.bandwidth_scope,
            self.evidence_mode,
            self.correctness_mode,
            self.sigma_min,
            self.sigma_max,
            self.ess_min,
            self.ess_max,
            self.strength_min,
            self.strength_max,
            self.alpha0,
            self.beta0,
            self.quantile,
            self.exclude_self,
            self.remaining_budget,
            self.grid_resolution,
            self.oracle_confidence,
            self.min_votes_for_classifier,
            self.hidden_layer_size,
            self.mlp_alpha,
            self.mlp_max_iter,
            self.classifier_seed,
            self.classifier_view_class,
            self.show_other_observed,
            self.show_latent,
        ]
        for control in watched:
            control.observe(self._control_changed, names="value")

        self.controls = widgets.VBox(
            [
                widgets.HBox(
                    [
                        self.reward_source,
                        self.surface_mode,
                        self.score_mode,
                        self.evidence_mode,
                        self.correctness_mode,
                    ]
                ),
                widgets.HBox([self.bandwidth_mode, self.bandwidth_scope, self.remaining_budget]),
                widgets.HBox([self.sigma_min, self.sigma_max, self.ess_min, self.ess_max]),
                widgets.HBox([self.strength_min, self.strength_max]),
                widgets.HBox([self.alpha0, self.beta0, self.quantile, self.exclude_self]),
                widgets.HBox([self.grid_resolution, self.oracle_confidence, self.classifier_view_class]),
                widgets.HBox(
                    [
                        self.min_votes_for_classifier,
                        self.hidden_layer_size,
                        self.mlp_alpha,
                        self.mlp_max_iter,
                        self.classifier_seed,
                    ]
                ),
                widgets.HBox(
                    [
                        self.initial_per_annotator,
                        self.init_seed,
                        self.random_init_button,
                        self.clear_button,
                        self.resample_button,
                        self.show_other_observed,
                        self.show_latent,
                    ]
                ),
                widgets.HBox(
                    [
                        self.cycle_step_button,
                        self.auto_run_button,
                        self.cycle_steps,
                        self.cycle_seed,
                        self.clear_history_button,
                    ]
                ),
            ]
        )

    def _control_changed(self, _change):
        self.redraw()

    def _clear_clicked(self, _button):
        self.Y_obs[:, :] = np.nan
        self.cycle_history.clear()
        self._last_cycle_pair = None
        self._cycle_rng = None
        self._cycle_rng_seed = None
        self._cycle_message = "cleared observations"
        self.redraw()

    def _random_init_clicked(self, _button):
        self.Y_obs[:, :] = np.nan
        self.cycle_history.clear()
        self._last_cycle_pair = None
        self._cycle_rng = None
        self._cycle_rng_seed = None
        rng = np.random.default_rng(int(self.init_seed.value))
        n_init = min(int(self.initial_per_annotator.value), self.n_samples)
        for annotator_index in range(self.n_annotators):
            if n_init == 0:
                continue
            chosen = rng.choice(self.n_samples, size=n_init, replace=False)
            self.Y_obs[chosen, annotator_index] = self.sim.z[chosen, annotator_index]
        self._cycle_message = f"random init: {n_init} labels per annotator"
        self.redraw()

    def _resample_clicked(self, _button):
        self._thompson_seed += 1
        self.redraw()

    def _cycle_step_clicked(self, _button):
        self._run_cycle_steps(1)

    def _auto_run_clicked(self, _button):
        self._run_cycle_steps(int(self.cycle_steps.value))

    def _clear_history_clicked(self, _button):
        self.cycle_history.clear()
        self._last_cycle_pair = None
        self._cycle_rng = None
        self._cycle_rng_seed = None
        self._cycle_message = "cleared cycle history"
        self.redraw()

    def _make_scorer(self) -> BudgetAwareKnnSoftAccuracyScorer:
        sigma_min = float(self.sigma_min.value)
        sigma_max = max(sigma_min, float(self.sigma_max.value))
        if sigma_max != self.sigma_max.value:
            self.sigma_max.value = sigma_max
        ess_min = float(self.ess_min.value)
        ess_max = max(ess_min, float(self.ess_max.value))
        if ess_max != self.ess_max.value:
            self.ess_max.value = ess_max
        strength_min = float(self.strength_min.value)
        strength_max = max(strength_min, float(self.strength_max.value))
        if strength_max != self.strength_max.value:
            self.strength_max.value = strength_max
        return BudgetAwareKnnSoftAccuracyScorer(
            bandwidth_mode=str(self.bandwidth_mode.value),
            bandwidth_scope=str(self.bandwidth_scope.value),
            evidence_mode=str(self.evidence_mode.value),
            correctness_mode=str(self.correctness_mode.value),
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            ess_min=ess_min,
            ess_max=ess_max,
            strength_min=strength_min,
            strength_max=strength_max,
            alpha0=float(self.alpha0.value),
            beta0=float(self.beta0.value),
            score_mode=str(self.score_mode.value),
            quantile=float(self.quantile.value),
            missing_label=self.missing_label,
            exclude_self=bool(self.exclude_self.value),
            random_state=int(self.classifier_seed.value) + self._thompson_seed,
        )

    def _make_grid(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        pad_x = 0.08 * (self.X[:, 0].max() - self.X[:, 0].min())
        pad_y = 0.08 * (self.X[:, 1].max() - self.X[:, 1].min())
        res = int(self.grid_resolution.value)
        x = np.linspace(self.X[:, 0].min() - pad_x, self.X[:, 0].max() + pad_x, res)
        y = np.linspace(self.X[:, 1].min() - pad_y, self.X[:, 1].max() + pad_y, res)
        xx, yy = np.meshgrid(x, y)
        return xx, yy, np.column_stack([xx.ravel(), yy.ravel()])

    def _probability_source(self, grid: np.ndarray) -> ProbabilitySourceResult:
        source = str(self.reward_source.value)
        if source == "oracle":
            return ProbabilitySourceResult(
                train_probabilities=make_oracle_probabilities(self.y_true, self.n_classes),
                grid_probabilities=None,
                status="reward source: true labels",
            )
        fallback = make_noisy_oracle_probabilities(
            self.y_true,
            self.n_classes,
            confidence=float(self.oracle_confidence.value),
        )
        if source == "noisy_oracle":
            return ProbabilitySourceResult(
                train_probabilities=fallback,
                grid_probabilities=None,
                status="reward source: true labels + uniform noise",
            )

        train_idx, labels, weights = self._majority_vote_training_data()
        if train_idx.size == 0 or np.unique(labels).size < 2:
            return ProbabilitySourceResult(
                train_probabilities=fallback,
                grid_probabilities=None,
                status="MLP fallback: waiting for pseudo-labels from at least 2 classes",
            )

        clf = MLPClassifier(
            hidden_layer_sizes=(int(self.hidden_layer_size.value),),
            alpha=float(self.mlp_alpha.value),
            max_iter=int(self.mlp_max_iter.value),
            random_state=int(self.classifier_seed.value),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            clf.fit(self.X[train_idx], labels, sample_weight=weights)

        P_train = self._predict_all_classes(clf, self.X)
        P_grid = self._predict_all_classes(clf, grid)
        train_pred = np.argmax(P_train[train_idx], axis=1)
        train_acc = float(np.mean(train_pred == labels))
        return ProbabilitySourceResult(
            train_probabilities=P_train,
            grid_probabilities=P_grid,
            status=(
                f"MLP reward: trained on {train_idx.size} pseudo-labeled samples, "
                f"classes={list(map(int, clf.classes_))}, pseudo-label acc={train_acc:.3f}"
            ),
        )

    def _majority_vote_training_data(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        observed = ~np.isnan(self.Y_obs)
        min_votes = int(self.min_votes_for_classifier.value)
        rows = []
        labels = []
        weights = []
        for sample_index in range(self.n_samples):
            votes = self.Y_obs[sample_index, observed[sample_index]].astype(int)
            if votes.size < min_votes:
                continue
            counts = np.bincount(votes, minlength=self.n_classes)
            label = int(np.flatnonzero(counts == counts.max())[0])
            rows.append(sample_index)
            labels.append(label)
            weights.append(float(counts.max() / votes.size))
        return (
            np.asarray(rows, dtype=int),
            np.asarray(labels, dtype=int),
            np.asarray(weights, dtype=float),
        )

    def _predict_all_classes(self, clf: MLPClassifier, X: np.ndarray) -> np.ndarray:
        P_fit = clf.predict_proba(X)
        P = np.full((X.shape[0], self.n_classes), 1e-12, dtype=float)
        P[:, clf.classes_.astype(int)] = P_fit
        return P / P.sum(axis=1, keepdims=True)

    def _cycle_random_state(self):
        seed = int(self.cycle_seed.value)
        if self._cycle_rng is None or self._cycle_rng_seed != seed:
            self._cycle_rng = np.random.default_rng(seed)
            self._cycle_rng_seed = seed
        return self._cycle_rng

    def _available_cycle_samples(self) -> np.ndarray:
        observed = ~np.isnan(self.Y_obs)
        return np.flatnonzero(~observed.all(axis=1))

    def _score_sample_annotators(
        self,
        sample_index: int,
        scorer: BudgetAwareKnnSoftAccuracyScorer,
        class_probabilities: np.ndarray,
    ) -> np.ndarray:
        annotator_indices = np.arange(self.n_annotators)
        available_mask = np.isnan(self.Y_obs[[sample_index], :])
        return scorer(
            X=self.X,
            y=self.Y_obs,
            sample_indices=np.asarray([sample_index], dtype=int),
            annotator_indices=annotator_indices,
            available_mask=available_mask,
            class_probabilities=class_probabilities,
            remaining_budget=int(self.remaining_budget.value),
        )[0]

    def _select_next_cycle_pair(self) -> tuple[int, int, float, str] | None:
        eligible_samples = self._available_cycle_samples()
        if eligible_samples.size == 0:
            self._cycle_message = "cycle stopped: no eligible samples remain"
            return None

        rng = self._cycle_random_state()
        scorer = self._make_scorer()
        _, _, grid = self._make_grid()
        prob = self._probability_source(grid)
        shuffled = rng.permutation(eligible_samples)
        for sample_index in shuffled:
            utilities = self._score_sample_annotators(
                int(sample_index),
                scorer,
                prob.train_probabilities,
            )
            if not np.any(np.isfinite(utilities)):
                continue
            annotator_index = int(np.nanargmax(utilities))
            score = float(utilities[annotator_index])
            if np.isfinite(score):
                return int(sample_index), annotator_index, score, prob.status

        self._cycle_message = "cycle stopped: no finite utilities for eligible samples"
        return None

    def _apply_cycle_pair(
        self,
        sample_index: int,
        annotator_index: int,
        score: float,
        probability_status: str,
    ):
        self.Y_obs[sample_index, annotator_index] = self.sim.z[sample_index, annotator_index]
        self._last_cycle_pair = (sample_index, annotator_index)

        rows, cols = np.where(~np.isnan(self.Y_obs))
        queried_accuracy = np.nan
        if rows.size > 0:
            queried_accuracy = float(
                np.mean(self.Y_obs[rows, cols].astype(int) == self.y_true[rows])
            )
        queried_label = int(self.sim.z[sample_index, annotator_index])
        correct = bool(queried_label == self.y_true[sample_index])
        step = len(self.cycle_history) + 1
        self.cycle_history.append(
            {
                "step": step,
                "sample": int(sample_index),
                "annotator": int(annotator_index),
                "score": float(score),
                "queried_label": queried_label,
                "true_label": int(self.y_true[sample_index]),
                "correct": correct,
                "queried_accuracy": queried_accuracy,
                "probability_status": probability_status,
            }
        )
        self._cycle_message = (
            f"cycle step {step}: sample={sample_index}, annotator={annotator_index}, "
            f"score={score:.3f}, correct={correct}"
        )

    def _run_cycle_steps(self, n_steps: int):
        completed = 0
        for _ in range(max(int(n_steps), 0)):
            selected = self._select_next_cycle_pair()
            if selected is None:
                break
            sample_index, annotator_index, score, probability_status = selected
            self._apply_cycle_pair(
                sample_index,
                annotator_index,
                score,
                probability_status,
            )
            completed += 1
        if completed > 1:
            self._cycle_message += f" | auto-run added {completed} labels"
        self.redraw()

    def _render_history_table(self, max_rows: int = 12) -> str:
        if not self.cycle_history:
            return "<b>Cycle history:</b> empty"
        rows = self.cycle_history[-max_rows:]
        header = (
            "<tr>"
            "<th>step</th><th>sample</th><th>ann</th><th>score</th>"
            "<th>label</th><th>true</th><th>correct</th><th>cum acc</th>"
            "</tr>"
        )
        body = []
        for item in rows:
            body.append(
                "<tr>"
                f"<td>{item['step']}</td>"
                f"<td>{item['sample']}</td>"
                f"<td>{item['annotator']}</td>"
                f"<td>{item['score']:.3f}</td>"
                f"<td>{item['queried_label']}</td>"
                f"<td>{item['true_label']}</td>"
                f"<td>{item['correct']}</td>"
                f"<td>{item['queried_accuracy']:.3f}</td>"
                "</tr>"
            )
        note = ""
        if len(self.cycle_history) > max_rows:
            note = f"<div>showing last {max_rows} of {len(self.cycle_history)} steps</div>"
        table = (
            "<table style='border-collapse: collapse;'>"
            "<style>"
            "td, th { border: 1px solid #ddd; padding: 2px 6px; text-align: right; }"
            "th { background: #f4f4f4; }"
            "</style>"
            f"{header}{''.join(body)}</table>"
        )
        return f"<b>Cycle history:</b>{note}{table}"

    def redraw(self):
        xx, yy, grid = self._make_grid()
        scorer = self._make_scorer()
        prob = self._probability_source(grid)

        with self.out:
            self.out.clear_output(wait=True)
            if self.fig is not None:
                plt.close(self.fig)
            n_extra = 1 if self.reward_source.value == "classifier" else 0
            n_panels = self.n_annotators + n_extra
            n_rows = int(np.ceil(n_panels / self.n_cols))
            self.fig, axes = plt.subplots(
                n_rows,
                self.n_cols,
                figsize=(4.7 * self.n_cols, 4.2 * n_rows),
                squeeze=False,
                constrained_layout=True,
            )
            self.axes = list(axes.ravel())
            self._axes_to_annotator = {}

            mappable = None
            for annotator_index in range(self.n_annotators):
                ax = self.axes[annotator_index]
                self._axes_to_annotator[ax] = annotator_index
                surface, diagnostic = self._surface_for_annotator(
                    scorer=scorer,
                    grid=grid,
                    annotator_index=annotator_index,
                    class_probabilities=prob.train_probabilities,
                )
                vmin, vmax, color_label = self._surface_limits(str(self.surface_mode.value))
                surface_for_plot = np.nan_to_num(surface, nan=vmin, posinf=vmax, neginf=vmin)
                surface_for_plot = np.clip(surface_for_plot, vmin, vmax)
                mappable = ax.contourf(
                    xx,
                    yy,
                    surface_for_plot.reshape(xx.shape),
                    levels=np.linspace(vmin, vmax, 21),
                    vmin=vmin,
                    vmax=vmax,
                    cmap="viridis",
                    alpha=0.82,
                )
                self._draw_samples(ax, annotator_index)
                n_obs = int(np.sum(~np.isnan(self.Y_obs[:, annotator_index])))
                emp = self._observed_accuracy(annotator_index)
                ax.set_title(
                    f"annotator {annotator_index} | obs={n_obs} | acc={emp} \n {diagnostic} "
                )
                ax.set_xlabel("x1")
                ax.set_ylabel("x2")
                ax.grid(alpha=0.18)

            if n_extra:
                ax = self.axes[self.n_annotators]
                self._draw_classifier_panel(ax, xx, yy, grid, prob)

            for ax in self.axes[n_panels:]:
                ax.axis("off")

            if mappable is not None:
                self.fig.colorbar(
                    mappable,
                    ax=self.axes[:n_panels],
                    shrink=0.82,
                    label=color_label,
                )
            if self._click_cid is not None:
                self.fig.canvas.mpl_disconnect(self._click_cid)
            self._click_cid = self.fig.canvas.mpl_connect("button_press_event", self._on_click)
            plt.show()

        observed_total = int(np.sum(~np.isnan(self.Y_obs)))
        cycle_message = f"<br>{self._cycle_message}" if self._cycle_message else ""
        self.status.value = (
            f"{prob.status}<br>observed labels: {observed_total}{cycle_message}"
        )
        self.history.value = self._render_history_table()

    def _surface_for_annotator(
        self,
        *,
        scorer: BudgetAwareKnnSoftAccuracyScorer,
        grid: np.ndarray,
        annotator_index: int,
        class_probabilities: np.ndarray,
    ) -> tuple[np.ndarray, str]:
        if self.surface_mode.value == "true_accuracy":
            return self._true_accuracy_surface(grid, annotator_index), "truth"
        posterior = scorer.posterior_for_query_points(
            X_query=grid,
            X=self.X,
            y=self.Y_obs,
            annotator_index=annotator_index,
            class_probabilities=class_probabilities,
            remaining_budget=int(self.remaining_budget.value),
        )
        diagnostic = self._posterior_diagnostic(posterior)
        if self.surface_mode.value == "posterior_mean":
            return np.clip(posterior.mean, 0.0, 1.0), diagnostic
        if self.surface_mode.value == "bandwidth_sigma":
            return posterior.bandwidth_sigma, diagnostic
        if self.surface_mode.value == "effective_sample_size":
            return posterior.effective_sample_size, diagnostic
        if self.surface_mode.value == "evidence_mass":
            return posterior.evidence_mass, diagnostic
        if self.surface_mode.value == "evidence_confidence":
            return posterior.evidence_confidence, diagnostic
        return np.clip(scorer.score_posterior(posterior), 0.0, 1.0), diagnostic

    def _posterior_diagnostic(self, posterior: QueryPosterior) -> str:
        sigma = self._finite_mean(posterior.bandwidth_sigma)
        ess = self._finite_mean(posterior.effective_sample_size)
        mass = self._finite_mean(posterior.evidence_mass)
        confidence = self._finite_mean(posterior.evidence_confidence)
        return f"sigma={sigma} | ESS={ess} | mass={mass} | conf={confidence}"

    @staticmethod
    def _finite_mean(values: np.ndarray) -> str:
        values = np.asarray(values, dtype=float)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return "nan"
        return f"{float(np.mean(finite)):.2f}"

    def _surface_limits(self, mode: str) -> tuple[float, float, str]:
        if mode in {"utility", "posterior_mean", "true_accuracy", "evidence_confidence"}:
            return 0.0, 1.0, "score / probability"
        if mode == "bandwidth_sigma":
            vmin = float(self.sigma_min.value)
            vmax = max(float(self.sigma_max.value), vmin + 1e-9)
            return vmin, vmax, "sigma"
        if mode == "effective_sample_size":
            return 0.0, max(float(self.ess_max.value), 1.0), "effective sample size"
        if mode == "evidence_mass":
            return 0.0, max(float(self._max_evidence_count()), 1.0), "evidence mass"
        return 0.0, 1.0, "value"

    def _max_evidence_count(self) -> int:
        observed = ~np.isnan(self.Y_obs)
        if self.bandwidth_scope.value == "all_annotators":
            return int(np.sum(observed.any(axis=1)))
        if observed.shape[1] == 0:
            return 0
        return int(np.max(observed.sum(axis=0)))

    def _true_accuracy_surface(self, grid: np.ndarray, annotator_index: int) -> np.ndarray:
        mode = getattr(self.sim, "mode", "")
        params = getattr(self.sim, "params", {})
        if mode == "global" and "accuracies" in params:
            return np.full(grid.shape[0], float(params["accuracies"][annotator_index]))
        if mode == "local" and {"expertise_centers", "min_acc", "max_acc", "bandwidth"} <= set(params):
            center = params["expertise_centers"][annotator_index]
            dist2 = np.sum((grid - center) ** 2, axis=1)
            locality = np.exp(-dist2 / (2.0 * params["bandwidth"][annotator_index] ** 2))
            return (
                params["min_acc"][annotator_index]
                + (params["max_acc"][annotator_index] - params["min_acc"][annotator_index])
                * locality
            )
        nearest = np.argmin(pairwise_distances(grid, self.X), axis=1)
        return np.asarray(self.sim.p_correct[nearest, annotator_index], dtype=float)

    def _draw_samples(self, ax, annotator_index: int):
        for class_index in range(self.n_classes):
            mask = self.y_true == class_index
            ax.scatter(
                self.X[mask, 0],
                self.X[mask, 1],
                s=18,
                marker="o",
                color=self._class_colors[class_index],
                alpha=0.25,
                linewidths=0,
            )

        observed = ~np.isnan(self.Y_obs)
        own = observed[:, annotator_index]
        other = observed.any(axis=1) & ~own
        if self.show_other_observed.value and np.any(other):
            ax.scatter(
                self.X[other, 0],
                self.X[other, 1],
                s=42,
                marker="o",
                facecolors="none",
                edgecolors="0.45",
                linewidths=0.8,
                alpha=0.55,
            )

        if self.show_latent.value:
            latent = ~own
            latent_correct = self.sim.z[:, annotator_index] == self.y_true
            self._scatter_by_class(
                ax,
                latent & latent_correct,
                marker=".",
                size=20,
                alpha=0.35,
                linewidths=0,
            )
            self._scatter_by_class(
                ax,
                latent & ~latent_correct,
                marker="x",
                size=34,
                alpha=0.35,
                linewidths=1.0,
            )

        correct = own & (self.sim.z[:, annotator_index] == self.y_true)
        wrong = own & ~correct
        self._scatter_by_class(
            ax,
            correct,
            marker="o",
            size=68,
            alpha=0.95,
            edgecolors="black",
            linewidths=0.7,
        )
        self._scatter_by_class(
            ax,
            wrong,
            marker="x",
            size=72,
            alpha=0.98,
            linewidths=2.0,
        )
        self._draw_cycle_highlight(ax, annotator_index)

    def _draw_cycle_highlight(self, ax, annotator_index: int):
        if self._last_cycle_pair is None:
            return
        sample_index, selected_annotator = self._last_cycle_pair
        if not (0 <= sample_index < self.n_samples):
            return
        x = self.X[sample_index, 0]
        y = self.X[sample_index, 1]
        if annotator_index == selected_annotator:
            ax.scatter(
                [x],
                [y],
                s=190,
                marker="o",
                facecolors="none",
                edgecolors="white",
                linewidths=3.0,
                alpha=0.95,
                zorder=20,
            )
            ax.scatter(
                [x],
                [y],
                s=230,
                marker="o",
                facecolors="none",
                edgecolors="black",
                linewidths=1.5,
                alpha=0.95,
                zorder=21,
            )
        else:
            ax.scatter(
                [x],
                [y],
                s=135,
                marker="o",
                facecolors="none",
                edgecolors="white",
                linewidths=2.0,
                alpha=0.65,
                zorder=20,
            )

    def _scatter_by_class(
        self,
        ax,
        mask: np.ndarray,
        *,
        marker: str,
        size: float,
        alpha: float,
        edgecolors=None,
        linewidths=1.0,
    ):
        for class_index in range(self.n_classes):
            cls_mask = mask & (self.y_true == class_index)
            if not np.any(cls_mask):
                continue
            kwargs = {
                "s": size,
                "marker": marker,
                "color": self._class_colors[class_index],
                "alpha": alpha,
                "linewidths": linewidths,
            }
            if edgecolors is not None and marker != "x":
                kwargs["edgecolors"] = edgecolors
            ax.scatter(self.X[cls_mask, 0], self.X[cls_mask, 1], **kwargs)

    def _draw_classifier_panel(self, ax, xx, yy, grid, prob: ProbabilitySourceResult):
        class_index = int(self.classifier_view_class.value)
        if prob.grid_probabilities is None:
            surface = np.full(grid.shape[0], np.nan)
        else:
            surface = prob.grid_probabilities[:, class_index]
        ax.contourf(
            xx,
            yy,
            surface.reshape(xx.shape),
            levels=np.linspace(0.0, 1.0, 21),
            vmin=0.0,
            vmax=1.0,
            cmap="viridis",
            alpha=0.82,
        )
        self._draw_training_samples(ax)
        ax.set_title(f"MLP P(class {class_index})")
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.grid(alpha=0.18)

    def _draw_training_samples(self, ax):
        train_idx, labels, weights = self._majority_vote_training_data()
        if train_idx.size == 0:
            return
        for class_index in range(self.n_classes):
            mask = labels == class_index
            if not np.any(mask):
                continue
            idx = train_idx[mask]
            ax.scatter(
                self.X[idx, 0],
                self.X[idx, 1],
                s=40 + 50 * weights[mask],
                color=self._class_colors[class_index],
                edgecolors="black",
                linewidths=0.6,
                alpha=0.9,
            )

    def _observed_accuracy(self, annotator_index: int) -> str:
        own = ~np.isnan(self.Y_obs[:, annotator_index])
        if not np.any(own):
            return "nan"
        acc = np.mean(self.sim.z[own, annotator_index] == self.y_true[own])
        return f"{acc:.2f}"

    def _on_click(self, event):
        annotator_index = self._axes_to_annotator.get(event.inaxes)
        if annotator_index is None or event.xdata is None or event.ydata is None:
            return
        point = np.asarray([[event.xdata, event.ydata]], dtype=float)
        distances = pairwise_distances(point, self.X)[0]
        sample_index = int(np.argmin(distances))
        x_range = max(float(np.ptp(self.X[:, 0])), 1e-12)
        y_range = max(float(np.ptp(self.X[:, 1])), 1e-12)
        threshold = 0.045 * float(np.hypot(x_range, y_range))
        if distances[sample_index] > threshold:
            return
        if np.isnan(self.Y_obs[sample_index, annotator_index]):
            self.Y_obs[sample_index, annotator_index] = self.sim.z[
                sample_index,
                annotator_index,
            ]
        else:
            self.Y_obs[sample_index, annotator_index] = np.nan
        self.redraw()
