import os
from typing import Dict, Any, List, Union, Callable
import pickle
import argparse
import logging
import json
import boto3

import pandas as pd
import numpy as np
import cupy as cp
import cudf
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight
import xgboost as xgb

from custom_utils import get_logger, parser, add_additional_args

# ------------------------------ Baseline function --------------------------- #


def create_baseline(hyperparameters: Dict[str, Any]) -> Pipeline:
    """
    Create the baseline model pipeline using the hyperparameters provided.

    Parameters
    ----------
    hyperparameters : Dict[str, Any]
        The hyperparameters for the model.

    Returns
    -------
    Pipeline
        The untrained baseline model pipeline.
    """
    # Early stopping
    xgb_early_stopping = xgb.callback.EarlyStopping(
        rounds=50,
        metric_name="mlogloss",
        data_name="validation_1",
        maximize=False,
        save_best=True,
    )

    # Create model pipeline
    model_pipeline = Pipeline(
        [
            (
                "xgb_clf",
                xgb.XGBClassifier(
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
                ),
            )
        ]
    )

    return model_pipeline


# ------------------------------ Train function ------------------------------ #


def train_baseline(
    model_func: Callable,
    hyperparameters: Dict[str, Any],
    train_data: List[Union[pd.DataFrame, pd.Series]],
    logger: logging.Logger,
) -> Dict[str, Any]:
    """
    Train the baseline model pipeline using the training data.

    Parameters
    ----------
    model_func : Callable
        A callable that returns an untrained model pipeline.
    hyperparameters : Dict[str, Any]
        The hyperparameters for the model.
    train_data : List[Union[pd.Dataframe, pd.Series]]
        The training data.
    logger : logging.Logger
        The logger object.

    Returns
    -------
    Dict[str, Any]
        The feature importance of the trained model.
    """
    # Cross-validation
    skf = StratifiedKFold(n_splits=5, shuffle=True)

    # Container for the cross-validation scores
    logloss_scores = {}
    # Container for feature importances
    feature_importances = {}

    for fold, (train_index, val_index) in enumerate(
        skf.split(train_data[0], train_data[1])
    ):
        # Get train and validation data (move to gpu memory)
        X_train, X_val = (
            cudf.DataFrame.from_pandas(train_data[0].iloc[train_index]),
            cudf.DataFrame.from_pandas(train_data[0].iloc[val_index]),
        )
        y_train, y_val = (
            cudf.Series.from_pandas(train_data[1].iloc[train_index]),
            cudf.Series.from_pandas(train_data[1].iloc[val_index]),
        )

        # Compute sample weights (using numpy)
        fold_sample_weights = compute_sample_weight(
            class_weight="balanced", y=y_train.values.get()
        )

        # Fit model
        fold_pipeline = model_func(hyperparameters)

        logger.info(f"Fitting model for fold {fold + 1}...")

        fold_pipeline.fit(
            X=X_train,
            y=y_train,
            xgb_clf__eval_set=[(X_train, y_train), (X_val, y_val)],
            xgb_clf__sample_weight=fold_sample_weights,
            xgb_clf__verbose=200,
        )

        logger.info(
            f"Extracting best score and feature importance for fold {fold + 1}..."
        )
        logloss_scores[f"fold_{fold + 1}"] = np.max(
            fold_pipeline.named_steps["xgb_clf"].evals_result()["validation_1"][
                "mlogloss"
            ]
        )
        # Feature importance must be obtained from the booster object
        feature_importances[f"fold_{fold + 1}"] = (
            fold_pipeline["xgb_clf"]
            .get_booster()
            .get_score(importance_type="total_gain")
        )

    # Compute mean cross-validation score
    mean_logloss_score = np.mean([logloss_scores[key] for key in logloss_scores])
    logger.info(f"Mean logloss score: {mean_logloss_score}")

    return feature_importances


if __name__ == "__main__":
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
    }

    args = add_additional_args(parser, additional_args)()

    logger = get_logger(__name__)

    job_name = args.training_env["job_name"]

    # --------------------------------- Load data -------------------------------- #

    logger.info("Loading data...")

    data = pd.read_parquet(os.path.join(args.train, "train_base.parquet"))
    X, y = data.drop("target", axis=1), data.target

    logger.info(f"Input data matrix has shape: {X.shape}")

    # --------------------------------- Train model -------------------------------- #

    logger.info("Training baseline model with cv...")

    feature_importances = train_baseline(
        model_func=create_baseline,
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
        },
        train_data=[X, y],
        logger=logger,
    )

    # ------------------------------- Persist model ------------------------------ #

    logger.info("Saving feature importances to s3...")

    # Create s3 client
    s3_client = boto3.client("s3")

    # Save feature importances
    s3_client.put_object(
        Bucket=args.s3_bucket,
        Key=os.path.join(args.s3_key, f"eda/{job_name}-feature-importances.pickle"),
        Body=pickle.dumps(feature_importances),
    )

    logger.info("Finished feature importances to s3...")

    s3_client.close()
