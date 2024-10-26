import argparse
import json
import logging
import os
import pickle
from typing import Any, Callable, Dict, List, Tuple, Union

import boto3
import cudf
import cupy as cp
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from custom_utils import (add_additional_args, create_pipeline, get_logger,
                          parser)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import (compute_class_weight,
                                        compute_sample_weight)

# ----------------- Function for creating lightgbm estimator ----------------- #


def create_lgb_estimator(
    hyperparameters: Dict[str, Any],
) -> Tuple[lgb.LGBMClassifier, Union[None, Callable]]:
    """
    Function that creates a lightgbm estimator.

    Parameters
    ----------
    hyperparameters : Dict[str, Any]
        Dictionary with hyperparameters.

    Returns
    -------
    Tuple[lgb.LGBMClassifier, Union[None, Callable]]
        Tuple with lightgbm estimator and either None or the early stopping callback.
    """
    # The best_iteration (index of iteration with best performance) attribute is only available when early stopping is used, which is also used at prediction time
    lgb_early_stopping = lgb.early_stopping(
        stopping_rounds=50,
        first_metric_only=False,  # Use all metrics for early stopping
        verbose=True,  # A custom logger will be used to print the messages outside of the function
        # min_delta=0.001 # Not available until release 4.0
    )

    lgb_clf = lgb.LGBMClassifier(
        device_type="cpu",
        n_jobs=-1,
        importance_type="gain",
        boosting="gbdt",
        objective="multiclass",
        num_class=9,
        metric="multi_logloss",
        auc_mu_weights=None,
        # Only used for multi-class classification (may require calibration afterwards)
        class_weight="balanced",
        # As a general rule, if we reduce num_iterations, we should increase learning_rate
        learning_rate=hyperparameters["learning_rate"],
        num_iterations=hyperparameters["num_iterations"],
        # Maximum number of nodes per tree and lower values can reduce training time
        # Should be smaller than 2^max_depth
        num_leaves=hyperparameters["num_leaves"],
        # Tree depth
        max_depth=hyperparameters["max_depth"],
        # The max number of bins that feature values will be bucketed in
        max_bin=hyperparameters["max_bin"],
        # The final max output of tree leaves is learning_rate * max_delta_step
        max_delta_step=hyperparameters["max_delta_step"],
        # Minimum loss reduction required to make a further partition on a leaf node of the tree, also known as 'min_gain_to_split' in the native API
        min_gain_to_split=hyperparameters["min_gain_to_split"],
        # Minimum sum of instance weight (Hessian) needed in a child (leaf)
        min_sum_hessian_in_leaf=hyperparameters["min_sum_hessian_in_leaf"],
        # Minimum number of data needed in a child (leaf)
        min_data_in_leaf=hyperparameters["min_data_in_leaf"],
        lambda_l1=hyperparameters["lambda_l1"],
        lambda_l2=hyperparameters["lambda_l2"],
        # Larger values give stronger regularization
        path_smooth=hyperparameters["path_smooth"],
        bagging_fraction=hyperparameters["bagging_fraction"],
        bagging_freq=hyperparameters["bagging_freq"],
        feature_fraction=hyperparameters["feature_fraction"],
        feature_fraction_bynode=hyperparameters["feature_fraction_bynode"],
        extra_trees=hyperparameters["extra_trees"],
    )

    return lgb_clf, lgb_early_stopping


# ----------------------------- Training function ---------------------------- #


def train_lgb(
    pipeline_func: Callable,
    estimator_func: Callable,
    inter_feat: List[str],
    poly_feat: List[str],
    hyperparameters: Dict[str, Any],
    train_data: List[Union[pd.DataFrame, pd.Series]],
    logger: logging.Logger,
) -> Pipeline:
    """
    Train the lightgbm model pipeline.

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
        # Get train and validation data (stay in cpu as lightgbm does not support gpu inputs yet)
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

        # Create estimator and early stopping callback
        fold_estimator, fold_lgb_early_stopping = estimator_func(
            hyperparameters=hyperparameters
        )

        logger.info(f"Training lightgbm model for fold {fold + 1}...")

        # Train model
        fold_estimator.fit(
            X=X_train,
            y=y_train,
            sample_weight=fold_sample_weights,
            # Training data is ignored by early_stopping callback
            eval_set=[(X_train, y_train), (X_val, y_val)],
            eval_names=["train", "val"],
            callbacks=[fold_lgb_early_stopping],
        )

        logger.info(f"Extracting best score for fold {fold + 1}...")
        logloss_scores[f"fold_{fold + 1}"] = fold_estimator.best_score_["val"][
            "multi_logloss"
        ]

    mean_logloss_score = np.mean([logloss_scores[key] for key in logloss_scores])
    logger.info(f"Mean logloss score: {mean_logloss_score}")

    # ---------------------- Retrain on entire training data --------------------- #

    logger.info("Retraining on entire training data...")
    # Get training data (move to gpu memory)
    X_train, y_train = train_data[0], train_data[1]

    # Compute sample weights (using numpy)
    sample_weights = compute_sample_weight(class_weight="balanced", y=y_train.values)

    # Create pipeline and insert estimator
    model_pipeline = pipeline_func(
        inter_feat, poly_feat, hyperparameters["svd"], hyperparameters["n_components"]
    )
    # Apply pipeline to training data
    X_train = model_pipeline.fit_transform(X_train, y_train)

    # Train model and append the trained estimator to the pipeline as the final step
    lgb_estimator, lgb_early_stopping = estimator_func(hyperparameters=hyperparameters)
    # Use two eval sets since lightgbm ignores the training data (first of the eval_sets) when early stopping is used
    lgb_estimator.fit(
        X=X_train,
        y=y_train,
        sample_weight=sample_weights,
        eval_set=[(X_train, y_train), (X_train, y_train)],
        callbacks=[lgb_early_stopping],
    )
    model_pipeline.steps.append(["lgb_clf", lgb_estimator])

    return model_pipeline


if __name__ == "__main__":
    # ---------------------------------- Set up ---------------------------------- #

    additional_args = {
        "learning_rate": float,
        "num_iterations": int,
        "num_leaves": int,
        "max_depth": int,
        "max_bin": int,
        "max_delta_step": float,
        "min_gain_to_split": float,
        "min_sum_hessian_in_leaf": float,
        "min_data_in_leaf": int,
        "lambda_l1": float,
        "lambda_l2": float,
        "path_smooth": float,
        "bagging_fraction": float,
        "bagging_freq": int,
        "feature_fraction": float,
        "feature_fraction_bynode": float,
        "extra_trees": int,
        "svd": int,
        "n_components": int,
    }

    args = add_additional_args(parser, additional_args)()

    logger = get_logger(__name__)

    # Register root logger with lightgbm
    lgb.register_logger(logger)

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

    model_pipeline = train_lgb(
        pipeline_func=create_pipeline,
        estimator_func=create_lgb_estimator,
        inter_feat=inter_feat,
        poly_feat=poly_feat,
        hyperparameters={
            "svd": args.svd,
            "n_components": args.n_components,
            "learning_rate": args.learning_rate,
            "num_iterations": args.num_iterations,
            "num_leaves": args.num_leaves,
            "max_depth": args.max_depth,
            "max_bin": args.max_bin,
            "max_delta_step": args.max_delta_step,
            "min_gain_to_split": args.min_gain_to_split,
            "min_sum_hessian_in_leaf": args.min_sum_hessian_in_leaf,
            "min_data_in_leaf": args.min_data_in_leaf,
            "lambda_l1": args.lambda_l1,
            "lambda_l2": args.lambda_l2,
            "path_smooth": args.path_smooth,
            "bagging_fraction": args.bagging_fraction,
            "bagging_freq": args.bagging_freq,
            "feature_fraction": args.feature_fraction,
            "feature_fraction_bynode": args.feature_fraction_bynode,
            "extra_trees": bool(args.extra_trees),  # Lightgbm expects a boolean
        },
        train_data=[X, y],
        logger=logger,
    )

    # ------------------------------- Persist Model ------------------------------ #

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
