import logging
import os
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from custom_utils import (add_additional_args, create_preprocessor,
                          create_study, custom_log_loss, get_db_url,
                          get_logger, parser, study_report)
from hydra import compose, core, initialize
from inference import input_fn, model_fn, predict_fn
from omegaconf import OmegaConf
from sklearn.compose import ColumnTransformer
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight

# ------------------ Function for creating xgboost estimator ----------------- #


def create_xgb_estimator(
    hyperparameters: Dict[str, Any], validation: bool, scale_pos_weight: float
) -> xgb.XGBClassifier:
    """
    Function that creates the XGBoost estimator.

    Parameters
    ----------
    hyperparameters : Dict[str, Any]
        Dictionary with hyperparameters.
    validation : bool
        Whether to use early stopping for validation set (True) or the entire training set (False).
    scale_pos_weight : float
        The ratio of the number of negative class to the positive class.

    Returns
    -------
    xgb.XGBClassifier
        The untrianed XGBoost estimator.
    """
    xgb_early_stopping = xgb.callback.EarlyStopping(
        rounds=50,
        metric_name="logloss",
        data_name=(
            "validation_1" if validation else "validation_0"
        ),  # When training on the entire training set, use the training set as the single validation set with index 0
        maximize=False,
        save_best=True,  # Save the best model
    )

    xgb_clf = xgb.XGBClassifier(
        objective="binary:logistic",
        booster="gbtree",
        tree_method="hist",
        importance_type="gain",
        predictor="cpu_predictor",
        eval_metric="logloss",
        n_jobs=-1,
        sampling_method="uniform",
        scale_pos_weight=scale_pos_weight,
        callbacks=[xgb_early_stopping],
        **hyperparameters,
    )

    return xgb_clf


# ----------------------------- Xgboost objective ---------------------------- #


def xgboost_objective(
    trial: optuna.Trial,
    aws_params: Dict[str, str],
    config: Dict[str, Any],
    preprocessor_func: Callable,
    estimator_func: Callable,
    train_data: Tuple[np.ndarray],
    test_mode: int,
    logger: logging.Logger,
) -> float:
    """
    Surrogate function for optuna to optimize.

    Parameters
    ----------
    trial : optuna.Trial
        Optuna trial object.
    aws_params : Dict[str, str]
        Dictionary containing the AWS parameters: the S3 bucket and key and training job name.
    config : Dict[str, Any]
        Dictionary containing the configuration parameters.
    preprocessor_func : Callable
        Function that creates the preprocessor pipeline.
    estimator_func : Callable
        Function that creates the estimator.
    train_data : Tuple[np.ndarray]
        Tuple containing train data and labels.
    test_mode: int
        Whether to run in test mode or not.
    logger : logging.Logger
        Logger object.

    Returns
    -------
    float
        The mean balanced log loss from cross-validation.
    """
    # Hyperparameters space
    model_hyperparameter = {
        "n_estimators": trial.suggest_int("n_estimators", 500, 1700),
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.8, log=True),
        "gamma": trial.suggest_float("gamma", 1, 20),
        "max_delta_step": trial.suggest_int("max_delta_step", 1, 20),
        "min_child_weight": trial.suggest_float(
            "min_child_weight", 0, 20
        ),  # Smaller values since this is a imbalanced dataset and small number of samples from a minority class can be in a leaf
        "max_leaves": trial.suggest_int("max_leaves", 0, 200),
        "subsample": trial.suggest_float("subsample", 0.5, 0.9),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 0.9),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.3, 0.9),
        "colsample_bynode": trial.suggest_float("colsample_bynode", 0.3, 0.9),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.01, 200),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.01, 200),
        "max_bin": trial.suggest_categorical("max_bin", [2**i for i in range(8, 11)]),
        "grow_policy": trial.suggest_categorical(
            "grow_policy", ["depthwise", "lossguide"]
        ),
    }

    # ----------------------------- Cross-validation ----------------------------- #

    skf = StratifiedKFold(n_splits=2 if test_mode else 20, shuffle=True)

    # Container for the cross-validation scores
    log_loss_scores = {}

    for fold, (train_index, val_index) in enumerate(
        skf.split(train_data[0], train_data[1])
    ):
        X_train, X_val = train_data[0].iloc[train_index], train_data[0].iloc[val_index]
        y_train, y_val = train_data[1].iloc[train_index], train_data[1].iloc[val_index]

        logger.info(
            f"Training set trial {trial.number} target distribution: {{0: {np.round(np.mean(y_train == 0), 2)}, 1: {np.round(np.mean(y_train == 1), 2)}}}"
        )
        logger.info(
            f"Validation set trial {trial.number} target distribution: {{0: {np.round(np.mean(y_val == 0), 2)}, 1: {np.round(np.mean(y_val == 1), 2)}}}"
        )

        # Compute sample weights
        sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)
        # Compute the ratio of the number of negative class to the positive class
        scale_pos_weight = np.sum(y_train == 0) / np.sum(y_train == 1)

        fold_preprocessor = preprocessor_func(
            top_5_feat=list(config["xgboost.top_5_feat"]),
            top_15_feat=list(config["xgboost.top_15_feat"]),
            num_feat=list(config["num_feat"]),
            cat_feat=list(config["cat_feat"]),
        )

        logger.info(
            f"Preprocessing training and validation data for trial {trial.number} fold {fold + 1}..."
        )

        # Fit and transform training data
        X_train = fold_preprocessor.fit_transform(X_train, y_train)
        # Transform validation data
        X_val = fold_preprocessor.transform(X_val)

        # Create estimator
        fold_estimator = estimator_func(
            hyperparameters=model_hyperparameter,
            validation=True,
            scale_pos_weight=scale_pos_weight,
        )

        logger.info(f"Training estimator for trial {trial.number} fold {fold + 1}...")

        # Train estimator
        fold_estimator.fit(
            X=X_train,
            y=y_train,
            sample_weight=sample_weights,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            verbose=200,
        )

        # Compute log loss on validation set
        logger.info(f"Computing log loss for trial {trial.number} fold {fold + 1}...")
        # Obtain the 1-D positive class probabilities
        y_pred = fold_estimator.predict_proba(X_val)[:, 1]
        log_loss_scores[f"fold_{fold + 1}"] = custom_log_loss(y_val, y_pred)

    # Compute mean log loss
    mean_log_loss = np.mean(list(log_loss_scores.values()))
    logger.info(f"Mean log loss for trial {trial.number}: {mean_log_loss}")

    # ---------------------- Retrain on entire training data --------------------- #

    logger.info(f"Retraining on entire training data for trial {trial.number}...")
    X_train, y_train = train_data[0], train_data[1]

    # Sample weights
    sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)
    # Compute the ratio of the number of negative class to the positive class
    scale_pos_weight = np.sum(y_train == 0) / np.sum(y_train == 1)

    # Create pipeline and insert estimator
    model_pipeline = preprocessor_func(
        top_5_feat=list(config["xgboost.top_5_feat"]),
        top_15_feat=list(config["xgboost.top_15_feat"]),
        num_feat=list(config["num_feat"]),
        cat_feat=list(config["cat_feat"]),
    )
    # Apply preprocessing to training data
    X_train = model_pipeline.fit_transform(X_train, y_train)

    # Train model and append trained estimator to the pipeline as a final step
    xgb_estimator = estimator_func(
        hyperparameters=model_hyperparameter,
        validation=False,
        scale_pos_weight=scale_pos_weight,
    )
    xgb_estimator.fit(
        X=X_train,
        y=y_train,
        sample_weight=sample_weights,
        verbose=200,
        eval_set=[(X_train, y_train)],
    )
    model_pipeline.steps.append(["xgb", xgb_estimator])

    # --------------------------------- Save mode -------------------------------- #

    logger.info(f"Saving model to tmp directory for trial {trial.number}...")
    local_model_dir = os.path.join("/tmp", f"model_trial_{trial.number}.joblib")
    joblib.dump(model_pipeline, local_model_dir)

    trial.set_user_attr("job_name", aws_params["job_name"])

    return mean_log_loss


if __name__ == "__main__":
    # ---------------------------------- Set up ---------------------------------- #

    additional_args = {"study_name": str}

    args = add_additional_args(parser, additional_args)()

    logger = get_logger(name=__name__)

    job_name = args.training_env["job_name"]

    # Hydra
    core.global_hydra.GlobalHydra.instance().clear()
    initialize(version_base="1.2", config_path="config", job_name="xgboost_training")
    config = OmegaConf.to_container(compose(config_name="main"), resolve=True)

    # --------------------------------- Load data -------------------------------- #

    logger.info("Loading data...")

    data = pd.read_csv(os.path.join(args.train, "train.csv"))
    if args.test_mode:
        data = data.sample(300)
    X, y = data.reset_index(drop=True).drop(["Class", "Id"], axis=1), data["Class"]

    logger.info(f"Training data shape: {X.shape}")
    logger.info(f"Class distribution: {{0: {np.sum(y == 0)}, 1: {np.sum(y == 1)}}}")

    # ------------------------------ Set up database ----------------------------- #

    logger.info("Setting up optuna database...")

    db_url = get_db_url(
        host=args.host,
        db_name=args.db_name,
        db_secret=args.db_secret,
        region_name=args.region_name,
    )

    logger.info(f"Database URL: {db_url}")

    # ------------------------------- Optimization ------------------------------- #

    logger.info("Optimizing objective function...")

    def objective_wrapper(trial: optuna.Trial) -> Callable:
        return xgboost_objective(
            trial=trial,
            aws_params={"job_name": job_name},
            config=config,
            preprocessor_func=create_preprocessor,
            estimator_func=create_xgb_estimator,
            train_data=(X, y),
            test_mode=args.test_mode,
            logger=logger,
        )

    study = create_study(
        study_name=args.study_name, storage=db_url, direction="minimize"
    )
    study.optimize(objective_wrapper, n_trials=args.n_trials, n_jobs=-1)

    study_report(study=study, logger=logger)

    # ----------------------- Retrieve and save best model ----------------------- #

    logger.info("Retrieving best model and saving to model directory...")

    best_model = joblib.load(
        os.path.join("/tmp", f"model_trial_{study.best_trial.number}.joblib")
    )

    # Save to model_dir for persistent storage
    model_dir = os.path.join(args.model_dir, f"best-model.joblib")
    joblib.dump(best_model, model_dir)
