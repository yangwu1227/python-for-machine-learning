import argparse
import os
import pickle
import warnings
from typing import Tuple

import boto3
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from model_utils import create_pipeline, setup_logger
from optuna.trial import TrialState
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight

from src.model_utils import load_data

# --------------------- Parse argument from command line --------------------- #


def parser() -> argparse.Namespace:
    """
    Function that parses arguments from command line.

    Returns
    -------
    argparse.Namespace
        Namespace with arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n_trials",
        "-n_trials",
        type=int,
        default=100,
        help="Number of HPO trials to run",
    )
    parser.add_argument(
        "--k", "-k", type=int, default=5, help="Number of cv folds per trial"
    )
    parser.add_argument(
        "--seed", "-seed", type=int, default=12, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--s3_key",
        "-s3_key",
        type=str,
        default="customer_churn",
        help="Destination S3 key for uploading HPO results",
    )
    parser.add_argument(
        "--s3_bucket",
        "-s3_bucket",
        type=str,
        default="yang-ml-sagemaker",
        help="S3 bucket name",
    )
    args, _ = parser.parse_known_args()

    return args


# -------------- Define surrogate objective function for Optuna -------------- #


def xgboost_objective(
    trial: optuna.Trial,
    args: argparse.Namespace,
    train_data: Tuple[pd.DataFrame, np.ndarray],
) -> float:
    """
    Objective function for Optuna.

    Parameters
    ----------
    trial : optuna.Trial
        Trial object for sampling hyperparameters.
    args : argparse.Namespace
        Namespace with arguments.
    train_data : Tuple[pd.DataFrame, np.ndarray]
        Tuple of training data (X_train, y_train).


    Returns
    -------
    float
        Performance metric on validation set.
    """
    skf = StratifiedKFold(n_splits=args.k, shuffle=True, random_state=args.seed)
    X_train: pd.DataFrame = train_data[0]
    y_train: np.ndarray = train_data[1]

    # Parameter space
    search_space = {
        # Booster parameters
        "booster_params": {
            "booster": "gbtree",
            "objective": "binary:logistic",  # Outputs probabilities
            "eval_metric": "aucpr",  # Use area under PR curve as the evaluation metric on validation sets
            "learning_rate": trial.suggest_float(
                name="learning_rate", low=0.001, high=0.3, log=True
            ),  # Range: [0, 1], larger eta shrinks the feature weights more to make the boosting process more conservative, i.e., fewer trees (regularizer)
            "gamma": trial.suggest_float(
                "gamma", low=0.05, high=1
            ),  # Range: [0, inf], the larger the more conservative the algorithm (regularizer)
            "max_delta_step": trial.suggest_int(
                "max_delta_step", 0, 10
            ),  # Range: [0, inf], values from 1-10 might help control the update for imbalanced data (regularizer)
            "lambda": trial.suggest_float(
                "lambda", low=0.01, high=0.1
            ),  # Range: [0, inf], L2 regularization term on weights, the larger the more conservative the algorithm (regularizer)
            "alpha": trial.suggest_float(
                "alpha", low=0.01, high=0.1
            ),  # Range: [0, inf], L1 regularization term on weights, the larger the more conservative the algorithm (regularizer)
            "colsample_bylevel": trial.suggest_float(
                name="colsample_bylevel", low=0.3, high=0.9
            ),
            "colsample_bynode": trial.suggest_float(
                name="colsample_bynode", low=0.3, high=0.9
            ),
            "colsample_bytree": trial.suggest_float(
                name="colsample_bytree", low=0.3, high=0.9
            ),  # Range: (0, 1], subsample ratio of columns when constructing each tree, the smaller the more conservative the algorithm (regularizer)
            "subsample": trial.suggest_float(
                name="subsample", low=0.5, high=0.9
            ),  # Range: (0, 1], subsample ratio of the training instances every boosting iteration, the smaller the more conservative the algorithm (regularizer)
            "sampling_method": "uniform",  # Typically set subsample >= 0.5 for good results
            "max_depth": trial.suggest_categorical(
                "max_depth", np.arange(3, 15, dtype=np.int16).tolist()
            ),  # Range: [0, inf], deep trees boost predictive power but are more likely to overfit (bias reducer)
            "tree_method": "hist",
            "scale_pos_weight": (y_train == 0).sum()
            / y_train.sum(),  # Class weight for positive class (churn = 'yes')
        },
        # Non-booster parameters
        "num_boost_round": trial.suggest_int(
            "num_boost_round", low=100, high=1500, step=100
        ),  # Range: [0, inf], number of boosting iterations, the larger the more likely to overfit (bias reducer)
        "num_feat": trial.suggest_categorical("num_feat", [50, 60, 70, 80, 90, 100]),
        "step": trial.suggest_categorical("step", [0.1, 0.2, 0.3]),
    }

    ap_scores = np.empty(args.k)
    for fold, (train_index, val_index) in enumerate(skf.split(X_train, y_train), 1):
        # Split into train and validation sets
        fold_X_train, fold_y_train = X_train.iloc[train_index], y_train[train_index]
        fold_X_val, fold_y_val = X_train.iloc[val_index], y_train[val_index]

        # A fresh preprocessor for each fold
        preprocessor = create_pipeline(
            num_feat=search_space["num_feat"], step=search_space["step"]
        )
        # Fit and transform on training set
        fold_X_train = preprocessor.fit_transform(fold_X_train, fold_y_train)
        # Transform on validation set
        fold_X_val = preprocessor.transform(fold_X_val)

        # Compute sample weights
        sample_weights = compute_sample_weight(class_weight="balanced", y=fold_y_train)

        # Data for modeling
        feature_names = preprocessor["rfe"].get_feature_names_out().tolist()
        dtrain = xgb.DMatrix(
            data=fold_X_train,
            label=fold_y_train,
            feature_names=feature_names,
            weight=sample_weights,
        )
        dvalid = xgb.DMatrix(
            data=fold_X_val, label=fold_y_val, feature_names=feature_names
        )
        # Early stopping
        early_stopping_callback = xgb.callback.EarlyStopping(
            rounds=200,
            metric_name="aucpr",
            data_name="valid",
            save_best=True,
            maximize=True,
        )

        model = xgb.train(
            params=search_space["booster_params"],
            dtrain=dtrain,
            num_boost_round=search_space["num_boost_round"],
            evals=[(dtrain, "train"), (dvalid, "valid")],
            callbacks=[early_stopping_callback],
            maximize=True,  # Maximize 'aucpr'
            verbose_eval=200,
        )

        # Out-of-fold prediction
        oof_pred = model.predict(dvalid)
        ap_scores[fold] = average_precision_score(
            y_true=fold_y_val, y_score=oof_pred, average="weighted"
        )

    mean_ap_scores = float(np.mean(ap_scores))

    return mean_ap_scores


# ------------------------------- Optimization ------------------------------- #


def create_study() -> Tuple[optuna.study.Study, str]:
    """
    Create Optuna study instance.

    Returns
    -------
    Tuple[optuna.study.Study, str]
        An instance of optuna.study.Study and the storage path to the sqlite file.
    """
    study_name = "churn_prediction_xgboost"
    storage_path = f"/home/ec2-user/SageMaker/customer-churn/output/{study_name}.db"
    # Remove database if already exists (in case we need to run HPO multiple times)
    if os.path.exists(storage_path):
        os.remove(storage_path)
    storage_name = f"sqlite:///{storage_path}"

    study = optuna.create_study(
        storage=storage_name,
        sampler=optuna.samplers.TPESampler(),
        # pruner=optuna.pruners.HyperbandPruner(),
        study_name=study_name,
        direction="maximize",
    )

    return study, storage_path


def main() -> int:
    warnings.filterwarnings(
        "ignore", category=UserWarning
    )  # Ignore warnings from optuna callback

    logger = setup_logger(__name__)
    optuna.logging.enable_propagation()  # Propagate logs to the root logger
    optuna.logging.disable_default_handler()  # Stop showing logs in sys.stderr

    args = parser()

    boto3_session = boto3.Session()
    s3 = boto3_session.resource("s3")

    X_train, y_train = load_data(
        f"s3://{args.s3_bucket}/{args.s3_key}/train_test/train.csv", logger=logger
    )

    study, storage_path = create_study()

    def objective_wrapper(trial: optuna.Trial) -> float:
        return xgboost_objective(
            trial=trial,
            args=args,
            train_data=(X_train, y_train),
        )

    study.optimize(objective_wrapper, n_trials=args.n_trials, n_jobs=-1)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    logger.info(f"Number of finished trials: {len(study.trials)}")
    logger.info(f"Number of pruned trials: {len(pruned_trials)}")
    logger.info(f"Number of complete trials: {len(complete_trials)}")
    logger.info(f"Best trial RMSE: {study.best_trial.value}")

    # Upload study object to s3
    picked_study_obj = pickle.dumps(study)
    s3.Object(args.s3_bucket, os.path.join(args.s3_key, "hpo/study.pickle")).put(
        Body=picked_study_obj
    )

    # Upload sqlite db to s3
    s3.Bucket(args.s3_bucket).upload_file(
        storage_path, os.path.join(args.s3_key, "hpo/trial_history.db")
    )

    logger.info(f"Best hyperparameters: {study.best_params}")

    return 0


if __name__ == "__main__":
    main()
