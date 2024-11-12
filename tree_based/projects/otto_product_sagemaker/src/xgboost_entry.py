import logging
import os
from typing import Any, Callable, Dict, List, Union

import boto3
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from tree_based.projects.otto_product_sagemaker.src.model_utils import (
    add_additional_args,
    create_pipeline,
    get_logger,
    parser,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight

# ------------------ Function for creating xgboost estimator ----------------- #


def create_xgb_estimator(
    hyperparameters: Dict[str, Any], validation: bool
) -> xgb.XGBClassifier:
    """
    Function that creates the XGBoost estimator.

    Parameters
    ----------
    hyperparameters : Dict[str, Any]
        Dictionary with hyperparameters.
    validation : bool
        Whether to use early stopping for validation set (True) or the entire training set (False).

    Returns
    -------
    xgb.XGBClassifier
        The untrianed XGBoost estimator.
    """
    xgb_early_stopping = xgb.callback.EarlyStopping(
        rounds=50,
        metric_name="mlogloss",
        data_name=(
            "validation_1" if validation else "validation_0"
        ),  # When training on the entire training set, use the training set as the single validation set with index 0
        maximize=True,
        save_best=True,  # Save the best model
    )

    xgb_clf = xgb.XGBClassifier(
        objective="multi:softprob",
        booster="gbtree",
        tree_method="gpu_hist",
        importance_type="gain",
        predictor="gpu_predictor",
        eval_metric="mlogloss",
        n_jobs=-1,
        n_estimators=hyperparameters["n_estimators"],
        max_depth=hyperparameters["max_depth"],
        learning_rate=hyperparameters["learning_rate"],
        min_child_weight=hyperparameters["min_child_weight"],
        gamma=hyperparameters["gamma"],
        max_delta_step=hyperparameters["max_delta_step"],
        subsample=hyperparameters["subsample"],
        max_leaves=hyperparameters["max_leaves"],
        sampling_method="uniform",
        colsample_bytree=hyperparameters["colsample_bytree"],
        colsample_bylevel=hyperparameters["colsample_bylevel"],
        colsample_bynode=hyperparameters["colsample_bynode"],
        reg_alpha=hyperparameters["reg_alpha"],
        reg_lambda=hyperparameters["reg_lambda"],
        max_bin=hyperparameters["max_bin"],
        callbacks=[xgb_early_stopping],
    )

    return xgb_clf


# ---------------------------- Train XGBoost model --------------------------- #


def train_xgboost(
    pipeline_func: Callable,
    estimator_func: Callable,
    inter_feat: List[str],
    poly_feat: List[str],
    hyperparameters: Dict[str, Any],
    train_data: List[Union[pd.DataFrame, pd.Series]],
    logger: logging.Logger,
) -> Pipeline:
    """
    Train the XGBoost model pipeline.

    Parameters
    ----------
    pipeline_func : Callable
        Function that creates the pipeline.
    estimator_func : Callable
        Function that creates the estimator.
    inter_feat : List[str]
        List of interaction features.
    poly_feat : List[str]
        List of polynomial features.
    hyperparameters : Dict[str, Any]
        Dictionary with hyperparameters.
    train_data : List[Union[pd.DataFrame, pd.Series]]
        The training data.
    logger : logging.Logger
        The Logger object.

    Returns
    -------
    Pipeline
        The trained pipeline that uses the entire training set.
    """
    # ----------------------------- Cross-validation ----------------------------- #

    skf = StratifiedKFold(n_splits=5, shuffle=True)

    # Container for the cross-validation scores
    logloss_scores = {}

    for fold, (train_index, val_index) in enumerate(
        skf.split(train_data[0], train_data[1])
    ):
        # Get train and validation data (do not move to gpu memory since we need stacking)
        # Lightgbm currently does not support gpu inputs so to match that we need xgboost to be cpu inputs
        # We can use gpu for xgboost by using cudf.DataFrame.from_pandas and cudf.Series.from_pandas
        X_train, X_val = train_data[0].iloc[train_index], train_data[0].iloc[val_index]
        y_train, y_val = train_data[1].iloc[train_index], train_data[1].iloc[val_index]

        # Compute sample weights (using numpy)
        fold_sample_weights = compute_sample_weight(
            class_weight="balanced", y=y_train.values
        )

        # Fit model
        fold_pipeline = pipeline_func(
            inter_feat,
            poly_feat,
            hyperparameters["svd"],
            hyperparameters["n_components"],
        )

        logger.info(
            f"Preprocessing training and validation data for fold {fold + 1}..."
        )

        # Fit and transform training data
        X_train = fold_pipeline.fit_transform(X_train, y_train)
        # Transform validation data
        X_val = fold_pipeline.transform(X_val)

        # Create estimator
        fold_estimator = estimator_func(
            hyperparameters=hyperparameters, validation=True
        )

        logger.info(f"Fitting model for fold {fold + 1}...")

        # Train model
        fold_estimator.fit(
            X=X_train,
            y=y_train,
            sample_weight=fold_sample_weights,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            verbose=200,
        )

        logger.info(f"Extracting best score for fold {fold + 1}...")
        logloss_scores[f"fold_{fold + 1}"] = np.min(
            fold_estimator.evals_result()["validation_1"]["mlogloss"]
        )

    # Compute mean cross-validation score
    mean_logloss_score = np.mean([logloss_scores[key] for key in logloss_scores])
    logger.info(f"Mean logloss score: {mean_logloss_score}")

    # ---------------------- Retrain on entire training data --------------------- #

    logger.info("Retraining on entire training data...")
    X_train, y_train = train_data[0], train_data[1]

    # Compute sample weights (using numpy)
    sample_weights = compute_sample_weight(class_weight="balanced", y=y_train.values)

    # Create pipeline and insert estimator
    model_pipeline = pipeline_func(
        inter_feat, poly_feat, hyperparameters["svd"], hyperparameters["n_components"]
    )
    # Apply pipeline to training data
    X_train = model_pipeline.fit_transform(X_train, y_train)

    # Train model and append the trained estimator to the pipeline as the final step (need to break up the pipeline into two parts since early stopping needs X_train to already be transformed)
    xgb_estimator = estimator_func(hyperparameters=hyperparameters, validation=False)
    xgb_estimator.fit(
        X=X_train,
        y=y_train,
        sample_weight=sample_weights,
        verbose=200,
        eval_set=[(X_train, y_train)],
    )
    model_pipeline.steps.append(["xgb_clf", xgb_estimator])

    return model_pipeline


def main() -> int:
    # ---------------------------------- Set up ---------------------------------- #

    additional_args = {
        "n_estimators": int,
        "max_depth": int,
        "learning_rate": float,
        "min_child_weight": float,
        "gamma": float,
        "max_leaves": int,
        "max_delta_step": int,
        "subsample": float,
        "colsample_bytree": float,
        "colsample_bylevel": float,
        "colsample_bynode": float,
        "reg_alpha": float,
        "reg_lambda": float,
        "max_bin": int,
        "svd": int,
        "n_components": int,
    }

    args = add_additional_args(parser, additional_args)()

    logger = get_logger(__name__)

    job_name = args.training_env["job_name"]

    # --------------------------------- Load data -------------------------------- #

    logger.info("Loading data...")

    data = pd.read_parquet(os.path.join(args.train, "train_base.parquet"))
    X, y = data.drop("target", axis=1), data.target

    logger.info(f"Input data matrix has shape: {X.shape}")

    # -------------------------------- Train model ------------------------------- #

    inter_feat = ["feat_11", "feat_25", "feat_34", "feat_60", "feat_67"]
    poly_feat = inter_feat + [
        "feat_14",
        "feat_15",
        "feat_36",
        "feat_39",
        "feat_40",
        "feat_50",
        "feat_62",
        "feat_75",
        "feat_86",
        "feat_90",
    ]

    model_pipeline = train_xgboost(
        pipeline_func=create_pipeline,
        estimator_func=create_xgb_estimator,
        inter_feat=inter_feat,
        poly_feat=poly_feat,
        hyperparameters={
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
            "learning_rate": args.learning_rate,
            "gamma": args.gamma,
            "max_delta_step": args.max_delta_step,
            "min_child_weight": args.min_child_weight,
            "max_leaves": args.max_leaves,
            "subsample": args.subsample,
            "colsample_bytree": args.colsample_bytree,
            "colsample_bylevel": args.colsample_bylevel,
            "colsample_bynode": args.colsample_bynode,
            "reg_alpha": args.reg_alpha,
            "reg_lambda": args.reg_lambda,
            "max_bin": args.max_bin,
            "svd": args.svd,
            "n_components": args.n_components,
        },
        train_data=[X, y],
        logger=logger,
    )

    # ------------------------------- Persist model ------------------------------ #

    # Save sklearn pipeline (including the estimator) to model directory for inference
    logger.info("Saving model to model directory...")
    local_model_dir = os.path.join(args.model_dir, f"{job_name}-model.joblib")
    joblib.dump(model_pipeline, local_model_dir)

    # Save model pipline to S3 for stacking
    logger.info("Saving model to S3...")
    s3_client = boto3.client("s3")
    s3_client.upload_file(
        Filename=local_model_dir,
        Bucket=args.s3_bucket,
        Key=os.path.join(args.s3_key, f"model/{job_name}-model.joblib"),
    )
    s3_client.close()

    return 0


if __name__ == "__main__":
    main()
