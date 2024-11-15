import logging
import os
from typing import Any, Dict, List, Union

import joblib
import numpy as np
import pandas as pd
import s3fs
from cuml import LogisticRegression
from cuml.metrics import log_loss
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight
from src.model_utils import add_additional_args, get_logger, parser

# ------------------- Function to train the stacking model ------------------- #


def train_ensemble(
    base_estimators: List[Pipeline],
    final_estimator_hyperparameters: Dict[str, Any],
    train_data: List[Union[pd.DataFrame, pd.Series]],
    logger: logging.Logger,
) -> StackingClassifier:
    """
    Train the stacking ensemble model.

    Parameters
    ----------
    base_estimators: List[Pipeline]
        List of base estimators to be used in the stacking ensemble model.
    final_estimator_hyperparameters: Dict[str, Any]
        Hyperparameters for the final estimator.
    train_data: List[Union[pd.DataFrame, pd.Series]]
        Training data for the stacking ensemble model.
    logger: logging.Logger
        The logger object.

    Returns
    -------
    StackingClassifier
        The trained stacking ensemble model.
    """
    # ----------------------------- Cross-validation ----------------------------- #

    skf = StratifiedKFold(n_splits=5, shuffle=True)
    # Container for log loss scores
    log_loss_scores = []

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

        # Model training
        logger.info(f"Training fold {fold+1}...")
        fold_stack_model = StackingClassifier(
            estimators=base_estimators,
            final_estimator=LogisticRegression(
                penalty=final_estimator_hyperparameters["penalty"],
                C=final_estimator_hyperparameters["inverse_reg_c"],
                l1_ratio=final_estimator_hyperparameters["l1_ratio"],
                class_weight=final_estimator_hyperparameters["class_weight"],
            ),
            cv="prefit",
            stack_method="predict_proba",
            n_jobs=-1,
        )

        fold_stack_model.fit(X_train, y_train, sample_weight=fold_sample_weights)

        # Evaluate on validation set
        logger.info(f"Evaluating fold {fold+1}...")
        y_pred_prob = fold_stack_model.predict_proba(X_val)

        # Calculate log loss
        log_loss_scores.append(log_loss(y_true=y_val, y_pred=y_pred_prob))

    # Take the average score across all folds
    avg_log_loss = np.mean(log_loss_scores)
    logger.info(f"Average log loss: {avg_log_loss}")

    # -------------------- Retrain on the entire training set -------------------- #

    logger.info("Retraining on the entire training set...")

    # Compute sample weights
    sample_weights = compute_sample_weight(
        class_weight="balanced", y=train_data[1].values
    )

    # Model training
    stack_model = StackingClassifier(
        estimators=base_estimators,
        final_estimator=LogisticRegression(
            penalty=final_estimator_hyperparameters["penalty"],
            C=final_estimator_hyperparameters["inverse_reg_c"],
            l1_ratio=final_estimator_hyperparameters["l1_ratio"],
            class_weight=final_estimator_hyperparameters["class_weight"],
        ),
        cv="prefit",
        stack_method="predict_proba",
        n_jobs=-1,
    )

    stack_model.fit(train_data[0], train_data[1], sample_weight=sample_weights)

    return stack_model


def main() -> int:
    # ---------------------------------- Set up ---------------------------------- #

    additional_args = {
        "inverse_reg_c": float,
        "penalty": str,
        "l1_ratio": float,
        "xgb_base_learner": str,
        "lgb_base_learner": str,
    }

    args = add_additional_args(parser, additional_args)()

    logger = get_logger(__name__)

    # S3FileSystem instance
    s3_fs = s3fs.S3FileSystem()

    # --------------------------------- Load data -------------------------------- #

    logger.info("Loading data...")

    data = pd.read_parquet(os.path.join(args.train, "train_stacking.parquet"))
    X, y = data.drop("target", axis=1), data.target

    # ---------------------------- Load base learners ---------------------------- #

    base_learners = []
    for s3_uri in [args.xgb_base_learner, args.lgb_base_learner]:
        with s3_fs.open(s3_uri, "rb") as f:
            base_learners.append(joblib.load(f))

    # --------------------------- Train stacking model --------------------------- #

    stack_model = train_ensemble(
        base_estimators=[
            (name, estimator) for name, estimator in zip(["xgb", "lgb"], base_learners)
        ],
        final_estimator_hyperparameters={
            "penalty": args.penalty,
            "inverse_reg_c": args.inverse_reg_c,
            "class_weight": "balanced",
            "l1_ratio": args.l1_ratio,
        },
        train_data=[X, y],
        logger=logger,
    )

    # ------------------------------- Persist model ------------------------------ #

    joblib.dump(
        stack_model, os.path.join(args.model_dir, "stacking_ensemble_model.joblib")
    )

    return 0


if __name__ == "__main__":
    main()
