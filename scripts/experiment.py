import sys
import json
from pathlib import Path

import warnings
import hydra
import numpy as np

from skorch.callbacks import LRScheduler
from torch.optim import RAdam
from torch.optim.lr_scheduler import CosineAnnealingLR
from hydra.utils import instantiate, get_class, to_absolute_path

from skactiveml.utils import majority_vote, is_labeled, call_func, is_unlabeled
from skactiveml.pool import SubSamplingWrapper

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _resolve_annotator_total_capacities(al_cfg, n_annotators):
    annotator_capacity_cfg = getattr(al_cfg, "annotator_capacity", None)
    if annotator_capacity_cfg is None:
        return None

    mode = str(getattr(annotator_capacity_cfg, "mode", "none")).lower()
    if mode == "none":
        return None

    if mode != "relative_equal_share":
        raise ValueError(
            "annotator_capacity.mode must be one of "
            "{'none', 'relative_equal_share'}."
        )

    multiplier = float(getattr(annotator_capacity_cfg, "multiplier", 1.0))
    if multiplier <= 0:
        raise ValueError("annotator_capacity.multiplier must be > 0.")

    total_pair_budget = int(al_cfg.init_pair_budget) + max(
        int(al_cfg.n_cycles) - 1, 0
    ) * int(al_cfg.actual_pair_budget)
    equal_share = (
        float(total_pair_budget) / float(n_annotators) if n_annotators > 0 else 0.0
    )
    cap = int(np.ceil(multiplier * equal_share))
    return np.full(n_annotators, cap, dtype=np.int64)


def _utility_tsne_notice(message):
    warnings.warn(message)
    print(f"[utility_tsne] {message}")


def _annotator_behavior_tsne_notice(message):
    warnings.warn(message)
    print(f"[annotator_behavior_tsne] {message}")


def _finite_diagnostic_stat(values, stat="mean"):
    if values is None:
        return None

    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return None
    if stat == "mean":
        return float(np.mean(finite))
    if stat == "p90":
        return float(np.quantile(finite, 0.9))
    raise ValueError("stat must be one of {'mean', 'p90'}.")


def _format_diagnostic_value(value):
    if value is None:
        return "nan"
    return f"{float(value):.4g}"


def _append_diagnostic_stat(parts, name, values, stat="mean"):
    parts.append(
        f"{name}={_format_diagnostic_value(_finite_diagnostic_stat(values, stat))}"
    )


def _valid_rho_values(scorer):
    rho = getattr(scorer, "last_rho_", None)
    h_actual = getattr(scorer, "last_h_actual_", None)
    h_ref = getattr(scorer, "last_h_ref_", None)
    if rho is None or h_actual is None or h_ref is None:
        return None, 0, 0

    rho = np.asarray(rho, dtype=float)
    h_actual = np.asarray(h_actual, dtype=float)
    h_ref = np.asarray(h_ref, dtype=float)
    valid = np.isfinite(rho) & np.isfinite(h_actual) & np.isfinite(h_ref)
    return rho[valid], int(np.count_nonzero(valid)), int(rho.size)


def _maybe_print_budget_aware_local_agreement_diagnostics(scorer, cycle_idx):
    responsive_combination = getattr(scorer, "last_responsive_combination_", None)
    if responsive_combination is None:
        return

    final_score = getattr(scorer, "last_final_score_", None)
    if final_score is None:
        n_pairs = 0
        n_finite_pairs = 0
    else:
        final_score = np.asarray(final_score, dtype=float)
        n_pairs = final_score.size
        n_finite_pairs = int(np.count_nonzero(np.isfinite(final_score)))

    local_success = getattr(scorer, "last_local_success_", None)
    local_failure = getattr(scorer, "last_local_failure_", None)
    local_mass = None
    if local_success is not None and local_failure is not None:
        local_mass = np.asarray(local_success, dtype=float) + np.asarray(
            local_failure, dtype=float
        )

    parts = [
        "[budget_aware_local_agreement]",
        f"cycle={cycle_idx}",
        f"locality={getattr(scorer, 'locality_mode', 'unknown')}",
        f"local_evidence={getattr(scorer, 'last_local_evidence_mode_', 'unknown')}",
        f"combination={responsive_combination}",
        f"gated_ts={getattr(scorer, 'last_gated_thompson_mode_', 'unknown')}",
        f"evidence={getattr(scorer, 'last_evidence_weighting_', getattr(scorer, 'evidence_weighting', 'unknown'))}",
        f"agreement={getattr(scorer, 'last_agreement_mode_', getattr(scorer, 'agreement_mode', 'unknown'))}",
        f"constraint_pressure={getattr(scorer, 'last_constraint_pressure_', 'unknown')}",
        f"rho_correction={getattr(scorer, 'last_use_rho_correction_', getattr(scorer, 'use_rho_correction', 'unknown'))}",
        f"score_mode={getattr(scorer, 'score_mode', 'unknown')}",
        f"ucb={getattr(scorer, 'last_ucb_mode_', getattr(scorer, 'ucb_mode', 'unknown'))}",
        f"bias={getattr(scorer, 'bias_model_correction', 'unknown')}",
        f"finite_pairs={n_finite_pairs}/{n_pairs}",
    ]
    _append_diagnostic_stat(
        parts, "q_mean", getattr(scorer, "last_evidence_weight_", None)
    )
    _append_diagnostic_stat(parts, "mu_pool", getattr(scorer, "last_mu_pool_", None))
    _append_diagnostic_stat(
        parts, "mu_global_mean", getattr(scorer, "last_mu_global_", None)
    )
    _append_diagnostic_stat(
        parts, "tau_pool_mean", getattr(scorer, "last_tau_pool_", None)
    )
    _append_diagnostic_stat(
        parts, "alpha_G_mean", getattr(scorer, "last_alpha_global_", None)
    )
    _append_diagnostic_stat(
        parts, "beta_G_mean", getattr(scorer, "last_beta_global_", None)
    )
    rho_values, rho_valid, rho_total = _valid_rho_values(scorer)
    _append_diagnostic_stat(parts, "local_mass_mean", local_mass)
    parts.append(f"rho_valid={rho_valid}/{rho_total}")
    _append_diagnostic_stat(parts, "rho_mean", rho_values)
    _append_diagnostic_stat(parts, "rho_p90", rho_values, stat="p90")
    _append_diagnostic_stat(
        parts, "rho_eff_mean", getattr(scorer, "last_rho_effective_", None)
    )
    kernel_bandwidth = getattr(scorer, "last_local_kernel_bandwidth_", None)
    if kernel_bandwidth is not None:
        _append_diagnostic_stat(parts, "kernel_bw_mean", kernel_bandwidth)
        _append_diagnostic_stat(parts, "kernel_bw_p90", kernel_bandwidth, stat="p90")
        _append_diagnostic_stat(
            parts,
            "kernel_weight_mean",
            getattr(scorer, "last_local_kernel_weight_sum_", None),
        )
        _append_diagnostic_stat(
            parts,
            "kernel_weight_p90",
            getattr(scorer, "last_local_kernel_weight_sum_", None),
            stat="p90",
        )
    _append_diagnostic_stat(parts, "raw_mean", getattr(scorer, "last_raw_score_", None))
    _append_diagnostic_stat(
        parts, "final_mean", getattr(scorer, "last_final_score_", None)
    )

    lambda_local = getattr(scorer, "last_lambda_local_", None)
    if lambda_local is None:
        _append_diagnostic_stat(parts, "nu_mean", getattr(scorer, "last_nu_", None))
        _append_diagnostic_stat(
            parts, "nu_p90", getattr(scorer, "last_nu_", None), stat="p90"
        )
    else:
        _append_diagnostic_stat(parts, "lambda_mean", lambda_local)
        _append_diagnostic_stat(parts, "lambda_p90", lambda_local, stat="p90")
        _append_diagnostic_stat(
            parts, "alpha_L_mean", getattr(scorer, "last_alpha_local_", None)
        )
        _append_diagnostic_stat(
            parts, "beta_L_mean", getattr(scorer, "last_beta_local_", None)
        )

    p_bias = getattr(scorer, "last_p_bias_", None)
    if p_bias is not None:
        _append_diagnostic_stat(parts, "p_bias_mean", p_bias)
        _append_diagnostic_stat(
            parts, "bias_score_mean", getattr(scorer, "last_bias_score_", None)
        )

    print(" ".join(parts))


def _preprocess_tsne_features(X, mode):
    X = np.asarray(X, dtype=np.float32).reshape(X.shape[0], -1)
    mode = str(mode).lower()

    if mode == "none":
        return X

    if mode == "l2_normalize":
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        return X / np.maximum(norms, np.float32(1e-12))

    if mode == "standardize":
        mean = X.mean(axis=0, dtype=np.float64)
        std = X.std(axis=0, dtype=np.float64)
        return ((X - mean) / np.maximum(std, 1e-12)).astype(np.float32, copy=False)

    raise ValueError(
        "annotator_behavior_tsne.feature_preprocess must be one of "
        "{'auto', 'none', 'l2_normalize', 'standardize'}."
    )


def _resolve_annotator_behavior_preprocess(tsne_cfg, simulation_cfg):
    mode = str(getattr(tsne_cfg, "feature_preprocess", "auto")).lower()
    if mode != "auto":
        return mode

    sim_mode = (
        None
        if simulation_cfg is None
        else getattr(simulation_cfg, "feature_preprocess", None)
    )
    if str(sim_mode).lower() == "l2_normalize":
        return "l2_normalize"
    return "none"


def _sample_tsne_plot_indices(
    *,
    candidate_indices,
    y_true,
    max_points,
    stratify_by_true_class,
    random_state,
):
    candidate_indices = np.asarray(candidate_indices, dtype=int)
    if max_points <= 0 or candidate_indices.size <= max_points:
        return np.sort(candidate_indices)

    rng = np.random.default_rng(random_state)
    if not stratify_by_true_class:
        return np.sort(rng.choice(candidate_indices, size=max_points, replace=False))

    by_class = []
    for cls in np.unique(y_true[candidate_indices]):
        cls_indices = candidate_indices[y_true[candidate_indices] == cls]
        rng.shuffle(cls_indices)
        by_class.append(cls_indices)

    selected = []
    offsets = np.zeros(len(by_class), dtype=int)
    while len(selected) < max_points:
        made_progress = False
        for class_pos, cls_indices in enumerate(by_class):
            if len(selected) >= max_points:
                break
            if offsets[class_pos] >= cls_indices.size:
                continue
            selected.append(int(cls_indices[offsets[class_pos]]))
            offsets[class_pos] += 1
            made_progress = True
        if not made_progress:
            break

    return np.sort(np.asarray(selected, dtype=int))


def _maybe_plot_annotator_behavior_tsne(
    *,
    tsne_cfg,
    X_train,
    y_train,
    z_train,
    classes,
    missing_label,
    simulation_cfg=None,
):
    if tsne_cfg is None or not bool(getattr(tsne_cfg, "enabled", False)):
        return []

    if z_train is None:
        _annotator_behavior_tsne_notice(
            "enabled=True but z_train is unavailable; skipping."
        )
        return []

    z_train = np.asarray(z_train)
    if z_train.ndim != 2:
        _annotator_behavior_tsne_notice(
            "expected z_train with shape (n_samples, n_annotators), got "
            f"{z_train.shape}; skipping."
        )
        return []

    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)
    if X_train.shape[0] != y_train.shape[0] or y_train.shape[0] != z_train.shape[0]:
        raise ValueError(
            "X_train, y_train, and z_train must agree on n_samples; got "
            f"{X_train.shape[0]}, {y_train.shape[0]}, {z_train.shape[0]}."
        )

    n_annotators = z_train.shape[1]
    configured_annotators = [int(a) for a in getattr(tsne_cfg, "annotator_indices", [])]
    if configured_annotators:
        selected_annotators = [
            annotator_id
            for annotator_id in configured_annotators
            if 0 <= annotator_id < n_annotators
        ]
        skipped_annotators = sorted(
            set(configured_annotators) - set(selected_annotators)
        )
        if skipped_annotators:
            _annotator_behavior_tsne_notice(
                "skipping annotator ids outside "
                f"[0, {n_annotators - 1}]: {skipped_annotators}."
            )
    else:
        selected_annotators = list(range(n_annotators))

    if not selected_annotators:
        _annotator_behavior_tsne_notice("no valid annotators selected; skipping.")
        return []

    observed_mask = is_labeled(
        z_train[:, selected_annotators], missing_label=missing_label
    )
    candidate_indices = np.flatnonzero(observed_mask.any(axis=1))
    if candidate_indices.size < 3:
        _annotator_behavior_tsne_notice(
            "requires at least 3 samples with selected annotator labels; "
            f"got {candidate_indices.size}. Skipping."
        )
        return []

    max_points = int(getattr(tsne_cfg, "max_points", 5000))
    plot_indices = _sample_tsne_plot_indices(
        candidate_indices=candidate_indices,
        y_true=y_train,
        max_points=max_points,
        stratify_by_true_class=bool(getattr(tsne_cfg, "stratify_by_true_class", True)),
        random_state=int(getattr(tsne_cfg, "random_state", 0)),
    )
    if plot_indices.size < 3:
        _annotator_behavior_tsne_notice(
            "requires at least 3 sampled points for t-SNE; "
            f"got {plot_indices.size}. Skipping."
        )
        return []

    preprocess_mode = _resolve_annotator_behavior_preprocess(tsne_cfg, simulation_cfg)
    X_plot = _preprocess_tsne_features(X_train[plot_indices], preprocess_mode)

    configured_perplexity = max(1.0, float(getattr(tsne_cfg, "perplexity", 30)))
    perplexity = min(
        configured_perplexity,
        max(1, (plot_indices.size - 1) // 3),
    )
    seed = int(getattr(tsne_cfg, "random_state", 0))

    from sklearn.manifold import TSNE
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt

    _annotator_behavior_tsne_notice(
        "creating shared t-SNE: "
        f"annotators={selected_annotators}, samples={len(plot_indices)}, "
        f"preprocess={preprocess_mode}, perplexity={perplexity}."
    )

    coords = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=seed,
    ).fit_transform(X_plot)

    output_dir_cfg = getattr(tsne_cfg, "output_dir", "figures/annotator_behavior_tsne")
    if output_dir_cfg in (None, ""):
        output_dir_cfg = "figures/annotator_behavior_tsne"
    output_dir = Path(str(output_dir_cfg))
    output_dir.mkdir(parents=True, exist_ok=True)
    file_format = str(getattr(tsne_cfg, "file_format", "png")).lstrip(".")
    if not file_format:
        file_format = "png"
    dpi = int(getattr(tsne_cfg, "dpi", 160))

    classes = np.asarray(classes)
    class_to_color = {int(cls): pos for pos, cls in enumerate(classes)}
    class_positions = np.asarray(
        [class_to_color[int(cls)] for cls in y_train[plot_indices]],
        dtype=int,
    )
    n_classes = max(len(classes), 1)
    cmap_name = "tab20" if n_classes <= 20 else "nipy_spectral"
    cmap = plt.get_cmap(cmap_name, n_classes)
    norm = mcolors.BoundaryNorm(np.arange(n_classes + 1) - 0.5, ncolors=n_classes)

    plot_info = []
    for annotator_id in selected_annotators:
        annotations = z_train[plot_indices, annotator_id]
        labeled_mask = is_labeled(annotations, missing_label=missing_label)
        if not np.any(labeled_mask):
            _annotator_behavior_tsne_notice(
                f"annotator {annotator_id} has no labels in sampled points; "
                "skipping plot."
            )
            continue

        correct_mask = labeled_mask & (annotations == y_train[plot_indices])
        incorrect_mask = labeled_mask & ~correct_mask
        n_labeled = int(labeled_mask.sum())
        n_correct = int(correct_mask.sum())
        n_incorrect = int(incorrect_mask.sum())
        accuracy = n_correct / n_labeled if n_labeled else np.nan

        fig, ax = plt.subplots(figsize=(8, 6))
        scatter = None
        if np.any(correct_mask):
            scatter = ax.scatter(
                coords[correct_mask, 0],
                coords[correct_mask, 1],
                c=class_positions[correct_mask],
                cmap=cmap,
                norm=norm,
                marker="o",
                s=24,
                alpha=0.85,
                linewidths=0,
                label="correct",
            )
        if np.any(incorrect_mask):
            scatter = ax.scatter(
                coords[incorrect_mask, 0],
                coords[incorrect_mask, 1],
                c=class_positions[incorrect_mask],
                cmap=cmap,
                norm=norm,
                marker="x",
                s=52,
                alpha=0.95,
                linewidths=1.2,
                label="incorrect",
            )

        colorbar = fig.colorbar(scatter, ax=ax)
        colorbar.set_label("True class")
        if n_classes <= 30:
            colorbar.set_ticks(np.arange(n_classes))
            colorbar.set_ticklabels([str(cls) for cls in classes])

        ax.set_title(
            "Annotator behavior t-SNE "
            f"annotator={annotator_id}, "
            f"acc={accuracy:.3f}, "
            f"labeled={n_labeled}, correct={n_correct}, "
            f"incorrect={n_incorrect}"
        )
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        ax.legend(loc="best")
        fig.tight_layout()

        output_path = output_dir / (
            f"annotator_behavior_tsne_annotator_"
            f"{int(annotator_id):03d}.{file_format}"
        )
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        _annotator_behavior_tsne_notice(f"saved {output_path}")

        plot_info.append(
            {
                "annotator_id": int(annotator_id),
                "plot_indices": plot_indices.copy(),
                "labeled_indices": plot_indices[labeled_mask].copy(),
                "correct_indices": plot_indices[correct_mask].copy(),
                "incorrect_indices": plot_indices[incorrect_mask].copy(),
                "accuracy": float(accuracy),
                "perplexity": float(perplexity),
                "feature_preprocess": preprocess_mode,
                "output_path": output_path,
            }
        )

    return plot_info


def _maybe_plot_utility_tsne(
    *,
    tsne_cfg,
    cycle_idx,
    clf,
    X_pool,
    y_pool,
    sample_indices,
    annotator_indices,
    utilities,
    pair_indices,
    missing_label,
):
    if tsne_cfg is None or not bool(getattr(tsne_cfg, "enabled", False)):
        return []

    cycles = [int(c) for c in getattr(tsne_cfg, "cycles", [])]
    if cycles and int(cycle_idx) not in set(cycles):
        _utility_tsne_notice(
            f"cycle {cycle_idx} skipped; configured cycles={cycles}. "
            "Cycle indices are zero-based, and [] means all cycles."
        )
        return []

    plot_annotators = [int(a) for a in getattr(tsne_cfg, "annotator_indices", [])]
    if not plot_annotators:
        _utility_tsne_notice(
            "utility_tsne.enabled=True but no annotator_indices are "
            "configured; skipping utility TSNE diagnostic."
        )
        return []

    sample_indices = np.asarray(sample_indices, dtype=int)
    annotator_indices = np.asarray(annotator_indices, dtype=int)
    utilities = np.asarray(utilities, dtype=float)
    pair_indices = np.asarray(pair_indices, dtype=int)

    if sample_indices.size == 0 or annotator_indices.size == 0:
        _utility_tsne_notice(
            "skipping because the current scorer grid is empty: "
            f"n_samples={sample_indices.size}, "
            f"n_annotators={annotator_indices.size}."
        )
        return []

    annotator_pos = {
        int(annotator_id): pos for pos, annotator_id in enumerate(annotator_indices)
    }
    selected_annotators = [
        annotator_id
        for annotator_id in plot_annotators
        if annotator_id in annotator_pos
    ]
    skipped_annotators = sorted(set(plot_annotators) - set(selected_annotators))
    if skipped_annotators:
        _utility_tsne_notice(
            "utility_tsne skipping annotators not present in current scorer "
            f"grid: {skipped_annotators}."
        )
    if not selected_annotators:
        _utility_tsne_notice(
            "no requested annotators remain after filtering by current "
            f"annotator_indices={annotator_indices.tolist()}."
        )
        return []

    labeled_mask = is_labeled(y_pool, missing_label=missing_label)
    labeled_sample_mask = (
        labeled_mask.any(axis=1) if labeled_mask.ndim == 2 else labeled_mask
    )
    labeled_indices = np.flatnonzero(labeled_sample_mask)
    annotator_labeled_by_id = {}
    for annotator_id in selected_annotators:
        if labeled_mask.ndim == 2:
            annotator_labeled_by_id[annotator_id] = np.flatnonzero(
                labeled_mask[:, annotator_id]
            )
        else:
            annotator_labeled_by_id[annotator_id] = labeled_indices.copy()

    seed = int(getattr(tsne_cfg, "random_state", 0)) + int(cycle_idx)
    rng = np.random.default_rng(seed)
    max_background_points = int(getattr(tsne_cfg, "max_background_points", 2000))
    if max_background_points <= 0:
        labeled_indices = np.empty(0, dtype=int)
    elif labeled_indices.size > max_background_points:
        labeled_indices = np.sort(
            rng.choice(
                labeled_indices,
                size=max_background_points,
                replace=False,
            )
        )

    annotator_labeled_union = (
        np.unique(
            np.concatenate(
                [values for values in annotator_labeled_by_id.values() if values.size]
            )
        )
        if any(values.size for values in annotator_labeled_by_id.values())
        else np.empty(0, dtype=int)
    )
    plot_indices = np.unique(
        np.concatenate([labeled_indices, sample_indices, annotator_labeled_union])
    )
    if plot_indices.size < 3:
        _utility_tsne_notice(
            "utility_tsne requires at least 3 total labeled/scored points; "
            f"got {plot_indices.size}. Skipping."
        )
        return []

    X_plot = X_pool[plot_indices]
    try:
        pred_out = clf.predict_proba(X_plot, extra_outputs=["embeddings"])
        if not isinstance(pred_out, tuple) or len(pred_out) < 2:
            raise ValueError("clf.predict_proba did not return embeddings as a tuple.")
        embedding = np.asarray(pred_out[1], dtype=float)
        used_embedding_fallback = False
    except Exception as exc:
        _utility_tsne_notice(
            "utility_tsne falling back to raw X because classifier embeddings "
            f"are unavailable: {exc}"
        )
        embedding = np.asarray(X_plot, dtype=float)
        used_embedding_fallback = True

    embedding = embedding.reshape(embedding.shape[0], -1)
    configured_perplexity = max(1.0, float(getattr(tsne_cfg, "perplexity", 30)))
    perplexity = min(
        configured_perplexity,
        max(1, (plot_indices.size - 1) // 3),
    )

    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt

    _utility_tsne_notice(
        "creating plots for cycle "
        f"{cycle_idx}: annotators={selected_annotators}, "
        f"labeled_background={len(labeled_indices)}, "
        f"annotator_labeled_union={len(annotator_labeled_union)}, "
        f"scored={len(sample_indices)}, total_tsne_points={len(plot_indices)}, "
        f"perplexity={perplexity}."
    )

    coords = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=seed,
    ).fit_transform(embedding)

    plot_pos = {int(idx): pos for pos, idx in enumerate(plot_indices)}
    background_pos = np.asarray(
        [plot_pos[int(idx)] for idx in labeled_indices if int(idx) in plot_pos],
        dtype=int,
    )
    scored_pos = np.asarray(
        [plot_pos[int(idx)] for idx in sample_indices],
        dtype=int,
    )
    output_dir_cfg = getattr(tsne_cfg, "output_dir", "figures/utility_tsne")
    if output_dir_cfg in (None, ""):
        output_dir_cfg = "figures/utility_tsne"
    output_dir = Path(str(output_dir_cfg))
    output_dir.mkdir(parents=True, exist_ok=True)
    file_format = str(getattr(tsne_cfg, "file_format", "png")).lstrip(".")
    if not file_format:
        file_format = "png"
    dpi = int(getattr(tsne_cfg, "dpi", 160))
    plot_info = []

    for annotator_id in selected_annotators:
        local_pos = annotator_pos[annotator_id]
        utility_values = utilities[:, local_pos]
        valid = np.isfinite(utility_values)
        annotator_labeled_indices = annotator_labeled_by_id[annotator_id]
        annotator_labeled_pos = np.asarray(
            [
                plot_pos[int(idx)]
                for idx in annotator_labeled_indices
                if int(idx) in plot_pos
            ],
            dtype=int,
        )

        if pair_indices.size == 0:
            assigned_samples = np.empty(0, dtype=int)
        else:
            assigned_samples = pair_indices[pair_indices[:, 1] == annotator_id, 0]
        assigned_pos = np.asarray(
            [plot_pos[int(idx)] for idx in assigned_samples if int(idx) in plot_pos],
            dtype=int,
        )

        fig, ax = plt.subplots(figsize=(8, 6))
        if background_pos.size:
            ax.scatter(
                coords[background_pos, 0],
                coords[background_pos, 1],
                c="0.75",
                s=14,
                alpha=0.35,
                linewidths=0,
                label="labeled",
            )

        if np.any(~valid):
            invalid_pos = scored_pos[~valid]
            ax.scatter(
                coords[invalid_pos, 0],
                coords[invalid_pos, 1],
                facecolors="none",
                edgecolors="0.55",
                s=44,
                linewidths=1.0,
                label="scored unavailable",
            )

        scatter = None
        if np.any(valid):
            valid_pos = scored_pos[valid]
            scatter = ax.scatter(
                coords[valid_pos, 0],
                coords[valid_pos, 1],
                c=utility_values[valid],
                cmap="viridis",
                s=42,
                alpha=0.95,
                edgecolors="black",
                linewidths=0.25,
                label="scored",
            )
            colorbar = fig.colorbar(scatter, ax=ax)
            colorbar.set_label(f"Utility for annotator {annotator_id}")

        if annotator_labeled_pos.size:
            ax.scatter(
                coords[annotator_labeled_pos, 0],
                coords[annotator_labeled_pos, 1],
                marker="s",
                s=72,
                facecolors="none",
                edgecolors="#1f77b4",
                linewidths=1.4,
                label="labeled by annotator",
            )

        if assigned_pos.size:
            ax.scatter(
                coords[assigned_pos, 0],
                coords[assigned_pos, 1],
                marker="*",
                s=180,
                facecolors="none",
                edgecolors="red",
                linewidths=1.7,
                label="assigned",
            )

        ax.set_title(
            "Utility TSNE "
            f"cycle={cycle_idx}, annotator={annotator_id}, "
            f"labeled={len(labeled_indices)}, scored={len(sample_indices)}, "
            f"by_annotator={len(annotator_labeled_indices)}, "
            f"assigned={len(assigned_samples)}"
        )
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        ax.legend(loc="best")
        fig.tight_layout()
        output_path = output_dir / (
            f"utility_tsne_cycle_{int(cycle_idx):03d}_"
            f"annotator_{int(annotator_id)}.{file_format}"
        )
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        _utility_tsne_notice(f"saved {output_path}")

        plot_info.append(
            {
                "cycle_idx": int(cycle_idx),
                "annotator_id": int(annotator_id),
                "plot_indices": plot_indices.copy(),
                "labeled_indices": labeled_indices.copy(),
                "annotator_labeled_indices": annotator_labeled_indices.copy(),
                "sample_indices": sample_indices.copy(),
                "assigned_samples": np.asarray(assigned_samples, dtype=int),
                "perplexity": float(perplexity),
                "used_embedding_fallback": used_embedding_fallback,
                "output_path": output_path,
            }
        )

    return plot_info


@hydra.main(config_path="../configs", config_name="experiment", version_base=None)
def experiment(cfg):
    warnings.filterwarnings("ignore")
    from src.dataset import HFNumpyFeaturePipeline, ensure_z_train_cached
    from src.utils import (
        seed_everything,
        log_results_to_mlflow,
        compute_cycle_metrics,
        pretty_dataset_report,
        pretty_cycle_metrics,
    )

    # Load dataset. -----------------------------------------------------------
    spec = instantiate(cfg.dataset)
    cls_embedder_cfg = getattr(cfg, "classification_embedder", None)

    # Backward-compatible fallback for older configs that only define
    # `embedder`.
    if cls_embedder_cfg is None:
        cls_embedder_cfg = cfg.embedder

    clf_embedder = instantiate(cls_embedder_cfg)
    clf_embedder_fingerprint = getattr(clf_embedder, "fingerprint", lambda: None)()

    pipe_cfg = instantiate(cfg.pipeline)
    pipe = HFNumpyFeaturePipeline(spec=spec, embedder=clf_embedder, cfg=pipe_cfg)
    np_arrays = pipe.get_arrays()
    X_train = np_arrays["X_train"]
    y_train = np_arrays["y_train"]
    z_train = np_arrays.get("z_train", None)
    X_test = np_arrays["X_test"]
    y_test = np_arrays["y_test"]
    n_samples = X_train.shape[0]
    n_features = X_train.shape[1]
    classes = np.unique(y_train)

    # Optional: Simulate noisy labels. ----------------------------------------
    if z_train is None and getattr(cfg, "simulation", None) is not None:
        sim_cfg = instantiate(cfg.simulation)
        sim_embedder_cfg = getattr(cfg, "simulation_embedder", None)
        if sim_embedder_cfg is None:
            sim_embedder_cfg = cls_embedder_cfg

        sim_embedder = instantiate(sim_embedder_cfg)
        sim_embedder_fingerprint = getattr(sim_embedder, "fingerprint", lambda: None)()
        use_same_embedder = sim_embedder_fingerprint == clf_embedder_fingerprint

        # Bind cache to dataset + simulation embedder so classification and
        # simulation can intentionally use different models without reusing
        # stale z_train from another simulation backbone.
        dataset_id = f"{spec.source}|train={list(spec.train_splits)}|y={spec.y_key}"
        dataset_id = (
            f"{dataset_id}|sim_embedder="
            f"{json.dumps(sim_embedder_fingerprint, sort_keys=True)}"
        )

        sim_X_train = np_arrays["X_train"] if use_same_embedder else None

        try:
            z_train, _ = ensure_z_train_cached(
                dataset_id=dataset_id,
                X_train_features=sim_X_train,  # only used on cache miss
                y_train=np_arrays["y_train"],
                cfg=sim_cfg,
                embedder_fingerprint=sim_embedder_fingerprint,
            )
        except ValueError as exc:
            needs_sim_features = "cache miss and X_train_features is None" in str(exc)
            if not needs_sim_features:
                raise

            sim_pipe = HFNumpyFeaturePipeline(
                spec=spec, embedder=sim_embedder, cfg=pipe_cfg
            )
            sim_arrays = sim_pipe.get_arrays()
            z_train, _ = ensure_z_train_cached(
                dataset_id=dataset_id,
                X_train_features=sim_arrays["X_train"],
                y_train=np_arrays["y_train"],
                cfg=sim_cfg,
                embedder_fingerprint=sim_embedder_fingerprint,
            )
        # z_train = np.column_stack([y_train] * z_train.shape[1])
        np_arrays["z_train"] = z_train

    # Print dataset summary. --------------------------------------------------
    pretty_dataset_report(
        classes=classes,
        n_features=n_features,
        n_samples=n_samples,
        np_arrays=np_arrays,
    )

    _maybe_plot_annotator_behavior_tsne(
        tsne_cfg=getattr(cfg, "annotator_behavior_tsne", None),
        X_train=X_train,
        y_train=y_train,
        z_train=z_train,
        classes=classes,
        missing_label=cfg.missing_label,
        simulation_cfg=getattr(cfg, "simulation", None),
    )

    if cfg.exit_after_simulation:
        print("Exiting after simulation as per config.")
        return

    # Seed everything. --------------------------------------------------------
    seed_everything(seed=cfg.seed, deterministic=False)

    # Build module. -----------------------------------------------------------
    module_dict = dict(cfg.module)
    clf_module = get_class(module_dict.pop("clf_module"))
    module_dict[f"module__in_features"] = n_features
    module_dict[f"module__out_features"] = len(classes)

    # Build learning rate scheduler. ------------------------------------------
    cosine_scheduler = LRScheduler(
        policy=CosineAnnealingLR,
        step_every="epoch",
        T_max=cfg.training.max_epochs,
    )

    # Build dictionary for neural network. ------------------------------------
    neural_net_param_dict = {
        # Module-related parameters.
        **module_dict,
        # Optimizer-related parameters.
        "max_epochs": cfg.training.max_epochs,
        "optimizer": RAdam,
        "optimizer__weight_decay": cfg.training.weight_decay,
        "optimizer__lr": cfg.training.learning_rate,
        "optimizer__decoupled_weight_decay": True,
        "callbacks": [("lr_scheduler", cosine_scheduler)],
        # Data loading parameters.
        "iterator_train__shuffle": True,
        "iterator_train__num_workers": cfg.training.num_workers,
        "iterator_train__batch_size": cfg.training.train_batch_size,
        "iterator_valid__batch_size": cfg.training.eval_batch_size,
        "iterator_train__drop_last": True,
        "train_split": None,
        # Misc.
        "verbose": 0,
        "device": cfg.device,
    }

    # Build classifier. -------------------------------------------------------
    clf = instantiate(
        cfg.classifier,
        clf_module=clf_module,
        neural_net_param_dict=neural_net_param_dict,
        classes=classes,
        missing_label=cfg.missing_label,
    )

    # Build sample query strategies. ------------------------------------------
    init_qs = instantiate(cfg.sample.init, missing_label=cfg.missing_label)
    init_qs = SubSamplingWrapper(
        query_strategy=init_qs,
        max_candidates=cfg.al.max_candidate_samples,
        exclude_non_subsample=True,
        missing_label=cfg.missing_label,
        random_state=cfg.seed,
    )
    actual_qs = instantiate(cfg.sample.actual, missing_label=cfg.missing_label)
    actual_qs = SubSamplingWrapper(
        query_strategy=actual_qs,
        max_candidates=cfg.al.max_candidate_samples,
        exclude_non_subsample=True,
        missing_label=cfg.missing_label,
        random_state=cfg.seed,
    )

    # Build sample-annotator pair utility model. ------------------------------
    init_scorer = instantiate(cfg.scorer.init)
    actual_scorer = instantiate(cfg.scorer.actual)

    # Build sample-annotator pair assigners. ----------------------------------
    init_assigner = instantiate(cfg.assigner.init)
    actual_assigner = instantiate(cfg.assigner.actual)

    # Build ratio scheduler. --------------------------------------------------
    ratio_scheduler = instantiate(cfg.scheduler)

    # Initialize data pool.
    X_pool = X_train
    y_pool = np.full_like(z_train, fill_value=cfg.missing_label)
    annotator_total_caps = _resolve_annotator_total_capacities(
        cfg.al,
        n_annotators=y_pool.shape[1],
    )

    # Setup logging helpers. --------------------------------------------------
    steps = []
    cycle_log = []
    prev_present = None
    total_pair_budget = int(cfg.al.init_pair_budget) + max(
        int(cfg.al.n_cycles) - 1, 0
    ) * int(cfg.al.actual_pair_budget)

    # Perform active learning cycle. ------------------------------------------
    for cycle_idx in range(cfg.al.n_cycles):

        # Set sampler, scorer, and assigner. ----------------------------------
        current_qs = init_qs if cycle_idx == 0 else actual_qs
        current_scorer = init_scorer if cycle_idx == 0 else actual_scorer
        current_assigner = init_assigner if cycle_idx == 0 else actual_assigner

        # Get current assignment per sample ratio.
        assignment_per_sample_ratio = ratio_scheduler(cycle_idx)
        current_pair_budget = (
            cfg.al.init_pair_budget if cycle_idx == 0 else cfg.al.actual_pair_budget
        )
        remaining_pair_budget = current_pair_budget + max(
            int(cfg.al.n_cycles) - cycle_idx - 1,
            0,
        ) * int(cfg.al.actual_pair_budget)
        current_sample_budget = int(
            -(-current_pair_budget // assignment_per_sample_ratio)
        )

        # Update availability of annotators.
        available_mask = np.logical_and(
            is_unlabeled(y_pool, missing_label=cfg.missing_label),
            is_labeled(z_train, missing_label=cfg.missing_label),
        )
        annotator_label_counts = np.sum(
            is_labeled(y_pool, missing_label=cfg.missing_label), axis=0
        ).astype(int, copy=False)
        annotator_remaining_counts = np.sum(available_mask, axis=0).astype(
            int, copy=False
        )
        if annotator_total_caps is not None:
            annotator_remaining_counts = np.minimum(
                annotator_remaining_counts,
                np.maximum(annotator_total_caps - annotator_label_counts, 0),
            )
        annotator_indices = np.flatnonzero(annotator_remaining_counts > 0)
        constraint_pressure = current_assigner.constraint_pressure(
            budget=current_pair_budget,
            annotator_indices=annotator_indices,
            annotator_remaining_counts=annotator_remaining_counts,
        )

        # Select candidate samples.
        is_cand = is_unlabeled(y_pool, missing_label=cfg.missing_label)
        is_cand = (
            is_cand.all(axis=-1)
            if cfg.al.fully_unlabeled_cand
            else is_cand.any(axis=-1)
        )
        candidates = np.flatnonzero(is_cand)
        if len(candidates) == 0:
            print(
                "[active_learning] "
                f"cycle={cycle_idx} has no candidate samples left "
                f"(fully_unlabeled_cand={cfg.al.fully_unlabeled_cand}); "
                "stopping early."
            )
            break

        # Select samples. -----------------------------------------------------
        y_agg = majority_vote(y_pool, classes=classes, missing_label=cfg.missing_label)
        sample_indices = call_func(
            f_callable=current_qs.query,
            X=X_pool,
            y=y_agg,
            candidates=candidates,
            batch_size=current_sample_budget,
            clf=clf,
            fit_clf=False,
        )

        # Compute utilities for selected samples. -----------------------------
        if len(sample_indices) == 0 or len(annotator_indices) == 0:
            utilities = np.empty(
                (len(sample_indices), len(annotator_indices)), dtype=float
            )
        else:
            utilities = call_func(
                f_callable=current_scorer,
                X=X_pool,
                y=y_pool,
                sample_indices=sample_indices,
                annotator_indices=annotator_indices,
                clf=clf,
                available_mask=available_mask[
                    np.ix_(sample_indices, annotator_indices)
                ],
                budget_total=total_pair_budget,
                remaining_budget=remaining_pair_budget,
                constraint_pressure=constraint_pressure,
            )
            budget_locality = getattr(
                current_scorer,
                "last_budget_aware_locality_",
                None,
            )
            if budget_locality is not None:
                diag = budget_locality.diagnostics
                print(
                    "[budget_aware_locality] "
                    f"cycle={cycle_idx} "
                    f"k_t={budget_locality.k_t} "
                    f"k_final={budget_locality.k_final} "
                    f"s_local={budget_locality.s_local:.4g} "
                    f"T_expected={diag['T_expected_t']:.4g} "
                    f"k_t/N={diag['k_t_over_N']:.4g} "
                    f"feasible={diag['local_modeling_feasible']}"
                )
            _maybe_print_budget_aware_local_agreement_diagnostics(
                current_scorer,
                cycle_idx,
            )

        # Assign annotators to samples given utilities. -----------------------
        pair_indices = current_assigner(
            utilities=utilities,
            sample_indices=sample_indices,
            annotator_indices=annotator_indices,
            budget=current_pair_budget,
            annotator_label_counts=annotator_label_counts,
            annotator_remaining_counts=annotator_remaining_counts,
        )

        _maybe_plot_utility_tsne(
            tsne_cfg=getattr(cfg, "utility_tsne", None),
            cycle_idx=cycle_idx,
            clf=clf,
            X_pool=X_pool,
            y_pool=y_pool,
            sample_indices=sample_indices,
            annotator_indices=annotator_indices,
            utilities=utilities,
            pair_indices=pair_indices,
            missing_label=cfg.missing_label,
        )

        # Query labels according to assignment. -------------------------------
        y_pool[(pair_indices[:, 0], pair_indices[:, 1])] = z_train[
            (pair_indices[:, 0], pair_indices[:, 1])
        ]
        # y_pool[sample_indices, 0] = y_train[sample_indices]

        # Retrain classifier and infer predictions for test samples. ----------
        clf.fit(X_pool, y_pool)
        p_pred_test = clf.predict_proba(X_test)

        # Log results of current cycle.
        steps.append((steps[-1] if len(steps) > 0 else 0) + current_pair_budget)
        entry = compute_cycle_metrics(
            y_acquired=y_pool,
            y_true=y_train,
            missing_label=cfg.missing_label,
            prev_present=prev_present,
            classes=classes,
            p_pred_test=p_pred_test,
            y_test=y_test,
        )
        prev_present = is_labeled(y_pool, missing_label=cfg.missing_label)
        cycle_log.append(entry)

        # Print active learning cycle summary. --------------------------------
        pretty_cycle_metrics(m=entry, cycle=cycle_idx)

    # Log results via mlflow. -------------------------------------------------
    log_results_to_mlflow(
        cfg=cfg,
        cycle_metrics=cycle_log,
        experiment_name=cfg.experiment_name,
        db_path=to_absolute_path(cfg.results_path),
        artifact_root=to_absolute_path(cfg.results_path),
        steps=steps,
        pre_write_jitter_max_seconds=cfg.mlflow_write_jitter_max_seconds,
    )


if __name__ == "__main__":
    experiment()
