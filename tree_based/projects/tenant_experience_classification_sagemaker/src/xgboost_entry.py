import logging
import os
from typing import Any, Callable, Dict, Tuple

import joblib
import numpy as np
import optuna
import polars as pl
import xgboost as xgb
from model_utils import (
    add_additional_args,
    create_study,
    get_db_url,
    get_logger,
    parser,
    study_report,
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.utils.class_weight import compute_sample_weight


def create_preprocessor() -> Pipeline:
    """
    Function that creates the preprocessor pipeline.

    Returns
    -------
    Pipeline
        The preprocessor pipeline.
    """
    preprocessor = Pipeline(steps=[("scaler", RobustScaler())])
    return preprocessor


def create_xgb_estimator(
    hyperparameters: Dict[str, Any],
    validation: bool,
    scale_pos_weight: float,
    test_mode: int,
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
    test_mode: int
        Whether to run in test mode, using cpu rather than gpu.

    Returns
    -------
    xgb.XGBClassifier
        The untrianed XGBoost estimator.
    """
    xgb_early_stopping = xgb.callback.EarlyStopping(
        rounds=50 if not test_mode else 10,
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
        eval_metric="logloss",
        n_jobs=-1,
        sampling_method="uniform",
        scale_pos_weight=scale_pos_weight,
        callbacks=[xgb_early_stopping],
        **hyperparameters,
    )

    return xgb_clf


def xgboost_objective(
    trial: optuna.Trial,
    aws_params: Dict[str, str],
    preprocessor_func: Callable,
    estimator_func: Callable,
    train_data: Tuple[pl.DataFrame, pl.Series],
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
    preprocessor_func : Callable
        Function that creates the processing pipeline.
    estimator_func : Callable
        Function that creates the estimator.
    train_data : Tuple[pl.DataFrame, pl.Series],
        Tuple containing train data and labels.
    test_mode: int
        Whether to run in test mode or not.
    logger : logging.Logger
        Logger object.

    Returns
    -------
    float
        The mean average precision from cross-validation.
    """
    # Hyperparameters space
    model_hyperparameter = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1500),
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        # Heuristic (2 to 10) / n_estimators
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "gamma": trial.suggest_float("gamma", 0.05, 1),
        "max_delta_step": trial.suggest_int("max_delta_step", 0, 10),
        # Smaller values for imbalanced dataset since small number of samples from a minority class can be in a leaf
        "min_child_weight": trial.suggest_float("min_child_weight", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.5, 0.9),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 0.9),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.3, 0.9),
        "colsample_bynode": trial.suggest_float("colsample_bynode", 0.3, 0.9),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.01, 0.1),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.01, 0.1),
        "max_bin": trial.suggest_categorical("max_bin", [2**i for i in range(8, 11)]),
        "grow_policy": trial.suggest_categorical(
            "grow_policy", ["depthwise", "lossguide"]
        ),
    }

    # ----------------------------- Cross-validation ----------------------------- #

    skf = StratifiedKFold(n_splits=2 if test_mode else 5, shuffle=True)

    # Container for the cross-validation scores
    avg_precision_scores = {}

    for fold, (train_val_index, cal_index) in enumerate(
        skf.split(train_data[0], train_data[1]), 1
    ):
        # Split the data into train_val and calibration sets
        X_train_val = train_data[0][train_val_index]
        y_train_val = train_data[1][train_val_index]
        X_cal = train_data[0][cal_index]
        y_cal = train_data[1][cal_index]

        # Further split train_val into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val,
            y_train_val,
            test_size=0.2,
            stratify=y_train_val,
            shuffle=True,
        )

        logger.info(
            f"Training set trial {trial.number} target distribution: {{0: {np.round(np.mean(y_train == 0), 2)}, 1: {np.round(np.mean(y_train == 1), 2)}}}"
        )
        logger.info(
            f"Validation set trial {trial.number} target distribution: {{0: {np.round(np.mean(y_val == 0), 2)}, 1: {np.round(np.mean(y_val == 1), 2)}}}"
        )
        logger.info(
            f"Calibration set trial {trial.number} target distribution: {{0: {np.round(np.mean(y_cal == 0), 2)}, 1: {np.round(np.mean(y_cal == 1), 2)}}}"
        )

        # Compute sample weights
        sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)
        # Compute the ratio of the number of negative class to the positive class
        scale_pos_weight = np.sum(y_train == 0) / np.sum(y_train == 1)

        # Create preprocessor
        fold_preprocessor = preprocessor_func()
        logger.info(
            f"Preprocessing training and validation data for trial {trial.number} fold {fold}..."
        )
        X_train = fold_preprocessor.fit_transform(X_train, y_train)
        X_val = fold_preprocessor.transform(X_val)

        # Create estimator
        fold_estimator = estimator_func(
            hyperparameters=model_hyperparameter,
            validation=True,
            scale_pos_weight=scale_pos_weight,
            test_mode=test_mode,
        )

        logger.info(f"Training estimator for trial {trial.number} fold {fold}...")

        # Train estimator with calibration
        fold_estimator.fit(
            X=X_train,
            y=y_train,
            sample_weight=sample_weights,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            verbose=200,
        )
        fold_calibrated_estimator = CalibratedClassifierCV(
            estimator=fold_estimator,
            method="isotonic",
            cv="prefit",
        )
        fold_calibrated_estimator.fit(X_cal, y_cal)

        # Compute average precision on validation set
        logger.info(
            f"Computing average precision for trial {trial.number} fold {fold}..."
        )
        # Obtain the 1-D positive class probabilities
        y_pred = fold_calibrated_estimator.predict_proba(X_val)[:, 1]
        avg_precision_scores[f"fold_{fold}"] = average_precision_score(y_val, y_pred)

    # Compute mean average precision
    mean_avg_precision = np.mean(list(avg_precision_scores.values()))
    logger.info(
        f"Mean average precision for trial {trial.number}: {mean_avg_precision}"
    )

    # ---------------------- Retrain on entire training data --------------------- #

    logger.info(f"Retraining on entire training data for trial {trial.number}...")
    X_train, y_train = train_data[0], train_data[1]

    # Sample weights
    sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)
    # Compute the ratio of the number of negative class to the positive class
    scale_pos_weight = np.sum(y_train == 0) / np.sum(y_train == 1)

    # Create preprocessor
    model_pipeline = create_preprocessor()
    X_train = model_pipeline.fit_transform(X_train, y_train)

    # Train model on entire training set
    xgb_estimator = estimator_func(
        hyperparameters=model_hyperparameter,
        validation=False,
        scale_pos_weight=scale_pos_weight,
        test_mode=test_mode,
    )
    calibrated_estimator = CalibratedClassifierCV(
        estimator=xgb_estimator,
        method="isotonic",
        cv=5,
    )
    calibrated_estimator.fit(
        X=X_train,
        y=y_train,
        sample_weight=sample_weights,
        verbose=200,
        eval_set=[(X_train, y_train)],
    )
    model_pipeline.steps.append(("xgb", calibrated_estimator))

    # -------------------------------- Save model -------------------------------- #

    logger.info(f"Saving model to tmp directory for trial {trial.number}...")
    local_model_dir = os.path.join("/tmp", f"model_trial_{trial.number}.joblib")
    joblib.dump(model_pipeline, local_model_dir)

    trial.set_user_attr("job_name", aws_params["job_name"])

    return mean_avg_precision


def main() -> int:
    # ---------------------------------- Set up ---------------------------------- #

    additional_args = {"study_name": str}

    args = add_additional_args(parser, additional_args)()

    logger = get_logger(name="training-job")

    job_name = args.training_env["job_name"]

    # --------------------------------- Load data -------------------------------- #

    logger.info("Loading data...")

    data = pl.read_parquet(os.path.join(args.train, "cluster_data.parquet"))
    data = data.with_columns(
        pl.when(pl.col("clusters") == "good_experience")
        .then(0)
        .otherwise(1)
        .alias("clusters")
    )
    if args.test_mode:
        data = data.sample(500)
    X, y = data.drop(["household_id", "clusters"]), data["clusters"].to_numpy()

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

    def objective_wrapper(trial: optuna.Trial) -> float:
        return xgboost_objective(
            trial=trial,
            aws_params={"job_name": job_name},
            preprocessor_func=create_preprocessor,
            estimator_func=create_xgb_estimator,
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
        best_model = joblib.load(
            os.path.join("/tmp", f"model_trial_{study.best_trial.number}.joblib")
        )
        # Save to model_dir for persistent storage, which will be uploaded to S3
        model_dir = os.path.join(args.model_dir, "best-model.joblib")
        joblib.dump(best_model, model_dir)
        logger.info(f"Best model saved to {model_dir}")
    except FileNotFoundError as error:
        logger.info(
            f"The best model does not exist in the current training job: {error}"
        )
    except Exception as error:
        logger.error(f"An error occurred while saving the best model: {error}")

    return 0


if __name__ == "__main__":
    main()
