import sys

import warnings
import hydra
import numpy as np

from skorch.callbacks import LRScheduler
from torch.optim import RAdam
from torch.optim.lr_scheduler import CosineAnnealingLR
from hydra.utils import instantiate, get_class, to_absolute_path

from skactiveml.utils import majority_vote, is_labeled, call_func, is_unlabeled
from skactiveml.pool import SubSamplingWrapper


@hydra.main(
    config_path="../configs", config_name="experiment", version_base=None
)
def experiment(cfg):
    warnings.filterwarnings("ignore")
    sys.path.append("../")
    from src.dataset import HFNumpyFeaturePipeline, ensure_z_train_cached
    from src.utils import (
        seed_everything,
        log_results_to_mlflow,
        compute_cycle_metrics,
        pretty_dataset_report,
        pretty_cycle_metrics
    )

    # Load dataset. -----------------------------------------------------------
    spec = instantiate(cfg.dataset)
    embedder = instantiate(cfg.embedder)
    pipe_cfg = instantiate(cfg.pipeline)
    pipe = HFNumpyFeaturePipeline(spec=spec, embedder=embedder, cfg=pipe_cfg)
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

        # A stable dataset identifier for caching. Keep it tied to the dataset
        # spec, not embedder.
        dataset_id = (
            f"{spec.source}|train={list(spec.train_splits)}|y={spec.y_key}"
        )

        z_train, info = ensure_z_train_cached(
            dataset_id=dataset_id,
            X_train_features=np_arrays["X_train"],  # only used on cache miss
            y_train=np_arrays["y_train"],
            cfg=sim_cfg,
            embedder_fingerprint=getattr(
                embedder, "fingerprint", lambda: None
            )(),
        )
        z_train[:, :-1] = 0
        np_arrays["z_train"] = z_train

    # Print dataset summary. ---------------------------------------------------
    pretty_dataset_report(
        classes=classes,
        n_features=n_features,
        n_samples=n_samples,
        np_arrays=np_arrays,
    )

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
    )
    actual_qs = instantiate(cfg.sample.actual, missing_label=cfg.missing_label)
    actual_qs = SubSamplingWrapper(
        query_strategy=actual_qs,
        max_candidates=cfg.al.max_candidate_samples,
        exclude_non_subsample=True,
        missing_label=cfg.missing_label,
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

    # Setup logging helpers. --------------------------------------------------
    steps = []
    cycle_log = []
    prev_present = None

    # Perform active learning cycle. ------------------------------------------
    for cycle_idx in range(cfg.al.n_cycles):

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
        current_sample_budget = int(-(-current_pair_budget // assignment_per_sample_ratio))

        # Update availability of annotators.
        available_mask = np.logical_and(
            is_unlabeled(y_pool, missing_label=cfg.missing_label),
            is_labeled(z_train, missing_label=cfg.missing_label),
        )

        # Select candidate samples.
        is_cand = is_unlabeled(y_pool, missing_label=cfg.missing_label)
        is_cand = (
            is_cand.any(axis=-1)
            if cfg.al.fully_unlabeled_cand
            else is_cand.all(axis=-1)
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
            clf=clf,
            fit_clf=False,
        )

        # Compute utilities for selected samples. -----------------------------
        utilities = call_func(
            f_callable=current_scorer,
            X=X_pool,
            y=y_pool,
            sample_indices=sample_indices,
            clf=clf,
            available_mask=available_mask[sample_indices],
        )

        # Assign annotators to samples given utilities. -----------------------
        pair_indices = call_func(
            f_callable=current_assigner,
            utilities=utilities,
            sample_indices=sample_indices,
            budget=current_pair_budget,
        )

        # Query labels according to assignment. -------------------------------
        y_pool[(pair_indices[:, 0], pair_indices[:, 1])] = z_train[
            (pair_indices[:, 0], pair_indices[:, 1])
        ]
        #y_pool[sample_indices, 0] = y_train[sample_indices]

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
    )


if __name__ == "__main__":
    experiment()
