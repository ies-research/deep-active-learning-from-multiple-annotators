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

from src.calibration import (
    build_soft_vote_targets,
    select_calibration_indices,
    set_classifier_temperature,
    tune_temperature_from_logits,
)

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
        float(total_pair_budget) / float(n_annotators)
        if n_annotators > 0
        else 0.0
    )
    cap = int(np.ceil(multiplier * equal_share))
    return np.full(n_annotators, cap, dtype=np.int64)


@hydra.main(
    config_path="../configs", config_name="experiment", version_base=None
)
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
    clf_embedder_fingerprint = getattr(
        clf_embedder, "fingerprint", lambda: None
    )()

    pipe_cfg = instantiate(cfg.pipeline)
    pipe = HFNumpyFeaturePipeline(
        spec=spec, embedder=clf_embedder, cfg=pipe_cfg
    )
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
        sim_embedder_fingerprint = getattr(
            sim_embedder, "fingerprint", lambda: None
        )()
        use_same_embedder = (
            sim_embedder_fingerprint == clf_embedder_fingerprint
        )

        # Bind cache to dataset + simulation embedder so classification and
        # simulation can intentionally use different models without reusing
        # stale z_train from another simulation backbone.
        dataset_id = (
            f"{spec.source}|train={list(spec.train_splits)}|y={spec.y_key}"
        )
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
            needs_sim_features = (
                "cache miss and X_train_features is None" in str(exc)
            )
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
        #z_train = np.column_stack([y_train] * z_train.shape[1])
        np_arrays["z_train"] = z_train

    # Print dataset summary. --------------------------------------------------
    pretty_dataset_report(
        classes=classes,
        n_features=n_features,
        n_samples=n_samples,
        np_arrays=np_arrays,
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

    def _build_classifier():
        cosine_scheduler = LRScheduler(
            policy=CosineAnnealingLR,
            step_every="epoch",
            T_max=cfg.training.max_epochs,
        )
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
        clf = instantiate(
            cfg.classifier,
            clf_module=clf_module,
            neural_net_param_dict=neural_net_param_dict,
            classes=classes,
            missing_label=cfg.missing_label,
        )
        return set_classifier_temperature(clf, 1.0)

    # The split classifier tunes T. The full classifier owns standard AL metrics.
    split_clf = _build_classifier()
    full_clf = _build_classifier()

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
    calib_cfg = getattr(cfg, "calibration", None)
    calibration_enabled = bool(
        getattr(calib_cfg, "enabled", False)
    ) if calib_cfg is not None else False
    policy_classifier = str(
        getattr(calib_cfg, "policy_classifier", "split")
    ).lower() if calib_cfg is not None else "full"
    if policy_classifier not in {"split", "full"}:
        raise ValueError("calibration.policy_classifier must be 'split' or 'full'.")

    # Perform active learning cycle. ------------------------------------------
    for cycle_idx in range(cfg.al.n_cycles):
        policy_clf = (
            split_clf
            if calibration_enabled and policy_classifier == "split"
            else full_clf
        )

        # Set sampler, scorer, and assigner. ----------------------------------
        current_qs = init_qs if cycle_idx == 0 else actual_qs
        current_scorer = init_scorer if cycle_idx == 0 else actual_scorer
        current_assigner = init_assigner if cycle_idx == 0 else actual_assigner

        # Get current assignment per sample ratio.
        assignment_per_sample_ratio = ratio_scheduler(cycle_idx)
        current_pair_budget = (
            cfg.al.init_pair_budget
            if cycle_idx == 0
            else cfg.al.actual_pair_budget
        )
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
        annotator_remaining_counts = np.sum(
            available_mask, axis=0
        ).astype(int, copy=False)
        if annotator_total_caps is not None:
            annotator_remaining_counts = np.minimum(
                annotator_remaining_counts,
                np.maximum(annotator_total_caps - annotator_label_counts, 0),
            )
        annotator_indices = np.flatnonzero(annotator_remaining_counts > 0)

        # Select candidate samples.
        is_cand = is_unlabeled(y_pool, missing_label=cfg.missing_label)
        is_cand = (
            is_cand.all(axis=-1)
            if cfg.al.fully_unlabeled_cand
            else is_cand.any(axis=-1)
        )
        candidates = np.flatnonzero(is_cand)

        # Select samples. -----------------------------------------------------
        y_agg = majority_vote(
            y_pool, classes=classes, missing_label=cfg.missing_label
        )
        sample_indices = call_func(
            f_callable=current_qs.query,
            X=X_pool,
            y=y_agg,
            candidates=candidates,
            batch_size=current_sample_budget,
            clf=policy_clf,
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
                clf=policy_clf,
                available_mask=available_mask[np.ix_(sample_indices, annotator_indices)],
            )
            print(np.nanmean(utilities, axis=0))

        # Assign annotators to samples given utilities. -----------------------
        remaining_budget = cfg.al.n_cycles * cfg.al.actual_pair_budget - cycle_idx * cfg.al.actual_pair_budget
        pair_indices = call_func(
            f_callable=current_assigner,
            utilities=utilities,
            utility_draws=getattr(current_scorer, "last_utility_draws_", None),
            sample_indices=sample_indices,
            annotator_indices=annotator_indices,
            budget=current_pair_budget,
            X=X_pool,
            clf=policy_clf,
            y=y_pool,
            missing_label=cfg.missing_label,
            annotator_label_counts=annotator_label_counts,
            annotator_remaining_counts=annotator_remaining_counts,
            remaining_budget=remaining_budget,
        )

        # Query labels according to assignment. -------------------------------
        y_pool[(pair_indices[:, 0], pair_indices[:, 1])] = z_train[
            (pair_indices[:, 0], pair_indices[:, 1])
        ]
        print(np.unique(pair_indices[:, 1], return_counts=True))
        # y_pool[sample_indices, 0] = y_train[sample_indices]

        # Retrain classifiers and infer predictions for test samples. ---------
        calibration_metrics = {}
        if calibration_enabled:
            calib_indices, split_stats = select_calibration_indices(
                y_pool,
                missing_label=cfg.missing_label,
                validation_fraction=float(calib_cfg.validation_fraction),
                min_labeled_samples=int(calib_cfg.min_labeled_samples),
                min_validation_samples=int(calib_cfg.min_validation_samples),
                min_votes_per_sample=int(calib_cfg.min_votes_per_sample),
                random_state=int(cfg.seed) + int(cycle_idx),
            )
            y_split = y_pool.copy()
            if calib_indices.size > 0:
                y_split[calib_indices] = cfg.missing_label

            set_classifier_temperature(split_clf, 1.0)
            split_clf.fit(X_pool, y_split)

            if calib_indices.size > 0:
                _, calib_logits = split_clf.predict_proba(
                    X_pool[calib_indices],
                    extra_outputs=["logits"],
                )
                calib_targets, calib_votes, calib_n_votes = (
                    build_soft_vote_targets(
                        y_pool[calib_indices],
                        classes=classes,
                        missing_label=cfg.missing_label,
                        smoothing_total=float(calib_cfg.soft_vote_smoothing_total),
                    )
                )
                calib_result = tune_temperature_from_logits(
                    calib_logits,
                    calib_targets,
                    vote_counts=calib_votes,
                    objective=str(calib_cfg.objective),
                    bounds=(
                        float(calib_cfg.temperature_min),
                        float(calib_cfg.temperature_max),
                    ),
                    ece_n_bins=int(getattr(calib_cfg, "ece_n_bins", 15)),
                )
                temperature = calib_result.temperature
                calibration_metrics.update(calib_result.metrics)
                calibration_metrics["calib_votes_mean"] = float(
                    np.mean(calib_n_votes)
                )
            else:
                temperature = 1.0
                calibration_metrics.update(
                    {
                        "calib_temperature": 1.0,
                        "calib_nll_before": np.nan,
                        "calib_nll_after": np.nan,
                        "calib_brier_before": np.nan,
                        "calib_brier_after": np.nan,
                        "calib_ece_before": np.nan,
                        "calib_ece_after": np.nan,
                        "calib_confidence_before": np.nan,
                        "calib_confidence_after": np.nan,
                        "calib_majority_acc_before": np.nan,
                        "calib_majority_acc_after": np.nan,
                        "calib_majority_balanced_acc_before": np.nan,
                        "calib_majority_balanced_acc_after": np.nan,
                        "calib_votes_mean": np.nan,
                    }
                )
            calibration_metrics.update(split_stats)
            set_classifier_temperature(split_clf, temperature)

            set_classifier_temperature(full_clf, 1.0)
            full_clf.fit(X_pool, y_pool)
            set_classifier_temperature(full_clf, temperature)
        else:
            set_classifier_temperature(full_clf, 1.0)
            full_clf.fit(X_pool, y_pool)
            set_classifier_temperature(full_clf, 1.0)
            calibration_metrics.update(
                {
                    "calib_enabled": 0.0,
                    "calib_temperature": 1.0,
                    "calib_selected_samples": 0.0,
                }
            )

        p_pred_test = full_clf.predict_proba(X_test)

        # Log results of current cycle.
        steps.append(
            (steps[-1] if len(steps) > 0 else 0) + current_pair_budget
        )
        entry = compute_cycle_metrics(
            y_acquired=y_pool,
            y_true=y_train,
            missing_label=cfg.missing_label,
            prev_present=prev_present,
            classes=classes,
            p_pred_test=p_pred_test,
            y_test=y_test,
        )
        active_policy_clf = (
            split_clf
            if calibration_enabled and policy_classifier == "split"
            else full_clf
        )
        policy_p_pred_test = active_policy_clf.predict_proba(X_test)
        policy_entry = compute_cycle_metrics(
            y_acquired=y_pool,
            y_true=y_train,
            missing_label=cfg.missing_label,
            prev_present=prev_present,
            classes=classes,
            p_pred_test=policy_p_pred_test,
            y_test=y_test,
        )
        entry.update(
            {
                f"policy_{k}": v
                for k, v in policy_entry.items()
                if k.startswith("test_")
            }
        )
        if calibration_enabled and policy_classifier == "full":
            split_p_pred_test = split_clf.predict_proba(X_test)
            split_entry = compute_cycle_metrics(
                y_acquired=y_pool,
                y_true=y_train,
                missing_label=cfg.missing_label,
                prev_present=prev_present,
                classes=classes,
                p_pred_test=split_p_pred_test,
                y_test=y_test,
            )
            entry.update(
                {
                    f"split_{k}": v
                    for k, v in split_entry.items()
                    if k.startswith("test_")
                }
            )
        entry.update(calibration_metrics)
        entry["calib_policy_classifier_split"] = float(
            calibration_enabled and policy_classifier == "split"
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
