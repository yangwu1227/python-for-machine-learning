import logging
import os
from collections.abc import Callable
from typing import Any, Dict, Optional, Tuple, cast

import joblib
import numpy as np
import optuna
import polars as pl
from hydra import compose, core, initialize
from lightgbm import LGBMClassifier, register_logger
from lightgbm.callback import _EarlyStoppingCallback, early_stopping
from omegaconf import OmegaConf
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils.class_weight import compute_sample_weight

from model_utils import (
    add_additional_args,
    compute_class_distribution,
    create_study,
    get_db_url,
    get_logger,
    parser,
    study_report,
)


def create_lgb_estimator(
    hyperparameters: Dict[str, Any],
    test_mode: int,
) -> Tuple[LGBMClassifier, _EarlyStoppingCallback]:
    """
    Function that creates the LightGBM estimator.

    Parameters
    ----------
    hyperparameters : Dict[str, Any]
        Dictionary with hyperparameters.
    test_mode: int
        Whether to run in test mode, using cpu rather than gpu.

    Returns
    -------
    Tuple[LGBMClassifier, _EarlyStoppingCallback]
        The untrianed Lightgbm estimator and the early stopping callback.
    """
    lgb_early_stopping = early_stopping(
        stopping_rounds=10 if test_mode else 100,
        first_metric_only=True,
        verbose=True,
    )

    lgb_clf = LGBMClassifier(
        objective="binary",
        device_type="cpu" if test_mode else "cuda",
        # Every iteration, randomly select samples
        bagging_freq=1,
        # No subsampling for the positive class
        pos_bagging_fraction=1.0,
        num_iterations=9999,
        learning_rate=1e-3,
        **hyperparameters,
    )

    return lgb_clf, lgb_early_stopping


def lightgbm_objective(
    trial: optuna.Trial,
    config: Dict[str, Any],
    aws_params: Dict[str, str],
    estimator_func: Callable,
    train_data: Tuple[pl.DataFrame, np.ndarray],
    test_mode: int,
    logger: logging.Logger,
) -> float:
    group_variable: np.ndarray = train_data[0][config["features"]["id"]].to_numpy()
    X_train: pl.DataFrame = train_data[0].drop(config["features"]["id"])
    y_train: np.ndarray = train_data[1]

    # Hyperparameters space
    model_hyperparameters = {
        "max_depth": trial.suggest_int("max_depth", 6, 15),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 50, 2000, step=50),
        "min_sum_hessian_in_leaf": trial.suggest_float(
            "min_sum_hessian_in_leaf", 1e-2, 1000
        ),
        # Under-sample the negative class
        "neg_bagging_fraction": trial.suggest_float("neg_bagging_fraction", 0.3, 0.9),
        "scale_pos_weight": trial.suggest_float(
            "scale_pos_weight", 1, np.sum(y_train == 0) / np.sum(y_train == 1)
        ),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.3, 0.9),
        "feature_fraction_bynode": trial.suggest_float(
            "feature_fraction_bynode", 0.3, 0.9
        ),
    }
    model_hyperparameters["num_leaves"] = trial.suggest_int(
        "num_leaves", 10, max(int(2 ** model_hyperparameters["max_depth"]), 2**9)
    )
    use_sample_weight = trial.suggest_categorical("use_sample_weight", [True, False])

    # -------------------------- Nested cross-validation ------------------------- #

    n_splits_outer = 2 if test_mode else 5
    n_splits_inner = 2 if test_mode else 3

    # Outer CV splitter
    sgkf_outer = StratifiedGroupKFold(n_splits=n_splits_outer)

    ap_scores = {}
    for outer_fold, (train_indices_outer, val_indices_outer) in enumerate(
        sgkf_outer.split(X_train, y_train, group_variable), 1
    ):
        # Split data for generalization evaluation
        X_train_outer, X_val_outer = (
            X_train[train_indices_outer],
            X_train[val_indices_outer],
        )
        y_train_outer, y_val_outer = (
            y_train[train_indices_outer],
            y_train[val_indices_outer],
        )
        group_variable_outer = group_variable[train_indices_outer]

        y_train_class_dist_outer = compute_class_distribution(y_train_outer)
        y_val_class_dist_outer = compute_class_distribution(y_val_outer)
        log_prefix_outer = f"Trial {trial.number} | Outer fold {outer_fold} |"
        logger.info(f"{log_prefix_outer} {'-' * 50}")
        logger.info(f"{log_prefix_outer} Train outer shape: {X_train_outer.shape}")
        logger.info(f"{log_prefix_outer} Val outer shape: {X_val_outer.shape}")
        logger.info(
            f"{log_prefix_outer} Train outer target distribution: {y_train_class_dist_outer}"
        )
        logger.info(
            f"{log_prefix_outer} Val outer target distribution: {y_val_class_dist_outer}"
        )

        # Inner CV splitter
        sgkf_inner = StratifiedGroupKFold(n_splits=n_splits_inner)
        best_ap_score_inner = -np.inf
        best_model: LGBMClassifier = None  # type: ignore[assignment]
        for inner_fold, (train_indices_inner, val_indices_inner) in enumerate(
            sgkf_inner.split(X_train_outer, y_train_outer, group_variable_outer), 1
        ):
            # Split data for hyperparameter tuning
            X_train_inner, X_val_inner = (
                X_train_outer[train_indices_inner],
                X_train_outer[val_indices_inner],
            )
            y_train_inner, y_val_inner = (
                y_train_outer[train_indices_inner],
                y_train_outer[val_indices_inner],
            )
            y_train_class_dist_inner = compute_class_distribution(y_train_inner)
            y_val_class_dist_inner = compute_class_distribution(y_val_inner)
            log_prefix_inner = f"{log_prefix_outer} Inner fold {inner_fold} |"
            logger.info(f"{log_prefix_inner} {'-' * 50}")
            logger.info(f"{log_prefix_inner} Train inner shape: {X_train_inner.shape}")
            logger.info(f"{log_prefix_inner} Val inner shape: {X_val_inner.shape}")
            logger.info(
                f"{log_prefix_inner} Train inner target distribution: {y_train_class_dist_inner}"
            )
            logger.info(
                f"{log_prefix_inner} Val inner target distribution: {y_val_class_dist_inner}"
            )

            # Compute sample weights
            sample_weights_inner: Optional[np.ndarray] = None
            if use_sample_weight:
                sample_weights_inner = compute_sample_weight(
                    class_weight="balanced", y=y_train_inner
                )

            # Train estimator
            estimator, lgb_early_stopping = estimator_func(
                hyperparameters=model_hyperparameters,
                test_mode=test_mode,
            )
            estimator.fit(
                X=X_train_inner,
                y=y_train_inner,
                sample_weight=sample_weights_inner,
                eval_set=[(X_train_inner, y_train_inner), (X_val_inner, y_val_inner)],
                eval_names=["train", "val"],
                eval_metric="average_precision",
                feature_name=X_train_inner.columns,
                callbacks=[lgb_early_stopping],
            )

            # Predict on validation set
            pred_prob_pos_inner = estimator.predict_proba(X_val_inner)[:, 1]
            ap_score_inner = average_precision_score(
                y_true=y_val_inner,
                y_score=pred_prob_pos_inner,
                pos_label=1,
            )
            # Update the best model and score if the current model is better
            if ap_score_inner > best_ap_score_inner:
                logger.info(
                    f" Previous: {best_ap_score_inner:.4f}, New: {ap_score_inner:.4f}, Improvement: {ap_score_inner - best_ap_score_inner:.4f}"
                )
                best_ap_score_inner = ap_score_inner
                best_model = estimator

        # Predict on outer validation set
        pred_prob_pos_outer = best_model.predict_proba(X_val_outer)[:, 1]
        ap_score_outer = average_precision_score(
            y_true=y_val_outer,
            y_score=pred_prob_pos_outer,
            pos_label=1,
        )
        ap_scores[outer_fold] = ap_score_outer

    # Compute average AP score
    avg_ap_score = np.mean(list(ap_scores.values()))
    logger.info(f"Trial {trial.number} | Average AP score: {avg_ap_score:.4f}")

    # ---------------------- Retrain on entire training set ---------------------- #

    logger.info(f"Trial {trial.number} | Retraining on entire training set")

    sample_weights: Optional[np.ndarray] = None
    if use_sample_weight:
        sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)

    estimator, lgb_early_stopping = estimator_func(
        hyperparameters=model_hyperparameters,
        test_mode=test_mode,
    )
    estimator.fit(
        X=X_train,
        y=y_train,
        sample_weight=sample_weights,
        eval_set=[(X_train, y_train)],
        eval_names=["val"],
        eval_metric="average_precision",
        feature_name=X_train.columns,
        callbacks=[lgb_early_stopping],
    )

    # -------------------------------- Save model -------------------------------- #

    logger.info(f"Trial {trial.number} | Saving model")
    local_model_dir = os.path.join("/tmp", f"model_trial_{trial.number}.joblib")
    joblib.dump(estimator, local_model_dir)
    # Save the sagemaker job name as an attribute for easy retrieval
    trial.set_user_attr("job_name", aws_params["job_name"])

    return avg_ap_score


def main() -> int:
    # ---------------------------------- Set up ---------------------------------- #

    core.global_hydra.GlobalHydra.instance().clear()
    initialize(version_base="1.2", config_path="config", job_name="lightgbm_training")
    config: Dict[str, Any] = cast(
        Dict[str, Any],
        OmegaConf.to_container(compose(config_name="main"), resolve=True),
    )

    additional_args = {"study_name": str}
    args = add_additional_args(parser, additional_args)()
    logger = get_logger(name="training_lightgbm")
    # Register the logger to LightGBM
    register_logger(logger)
    job_name = args.training_env["job_name"]

    # ---------------------------- Load data ---------------------------- #

    logger.info("Loading training data...")
    data = pl.read_parquet(os.path.join(args.train, "train.parquet"))
    if args.test_mode:
        # In test mode, sample some data to reduce computation time
        data = data.sample(n=500, seed=config["random_seed"])
    X = data.drop(config["features"]["target"])
    y = data[config["features"]["target"]].to_numpy()
    logger.info(f"Training data shape: {X.shape}")
    logger.info(f"Training target distribution: {compute_class_distribution(y)}")

    # ------------------------------ Set up database ----------------------------- #

    logger.info("Setting up optuna database...")
    db_url = get_db_url(
        host=args.host,
        db_name=args.db_name,
        db_secret=args.db_secret,
        region_name=args.region_name,
    )

    # ------------------------------- Optimization ------------------------------- #

    logger.info("Optimizing objective function...")

    def objective_wrapper(trial: optuna.Trial) -> float:
        return lightgbm_objective(
            trial=trial,
            config=config,
            aws_params={"job_name": job_name},
            estimator_func=create_lgb_estimator,
            train_data=(X, y),
            test_mode=args.test_mode,
            logger=logger,
        )

    study = create_study(
        study_name=args.study_name, storage=db_url, direction="maximize"
    )
    study.optimize(objective_wrapper, n_trials=args.n_trials)
    study_report(study=study, logger=logger)

    # ----------------------- Retrieve and save best model ----------------------- #

    # Handle cases where the best model may not exist in the current training job
    # During parallel training jobs, the best model for a given trial may be saved in another training job
    try:
        logger.info("Retrieving best model and saving to model directory...")
        best_model: LGBMClassifier = joblib.load(
            os.path.join("/tmp", f"model_trial_{study.best_trial.number}.joblib")
        )
        # Save to model_dir for persistent storage, which will be uploaded to S3
        best_model_dir = os.path.join(args.model_dir, "best_model.joblib")
        joblib.dump(best_model, best_model_dir)
        logger.info(f"Best model saved to {best_model_dir}")
    except FileNotFoundError as error:
        logger.info(
            f"The best model does not exist in the current training job: {error}"
        )
    except Exception as error:
        logger.error(f"An error occurred while saving the best model: {error}")

    return 0


if __name__ == "__main__":
    main()
