import os 
import sys
from typing import Tuple, Dict, Any, Optional, List, Union, Callable
import logging

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import log_loss

from hydra import compose, initialize, core
from omegaconf import OmegaConf
import pandas as pd
import numpy as np
import optuna

from custom_utils import (
    get_logger, get_db_url, parser, add_additional_args, custom_log_loss, 
    create_study, FeatureImportanceHandler, study_report
)

# ------------------- Function to create baseline pipeline ------------------- #

def create_baseline(hyperparameters: Dict[str, Any], num_feat: List[str], cat_feat: List[str], scaling: int) -> Pipeline:
    """
    Create a baseline modeling pipeline.

    Parameters
    ----------
    hyperparameters : Dict[str, Any]
        Dictionary containing the hyperparameters to use.
    num_feat : List[str]
        List of numerical features.
    cat_feat : List[str]
        List of categorical features.
    scaling : int
        Whether to scale the numerical features or not.
    
    Returns
    -------
    Pipeline
        A sklearn pipeline.
    """
    # Preprocessing pipeline
    if scaling:
        preprocessing_pipeline = ColumnTransformer([
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scalar', RobustScaler())
            ]), num_feat),
            ('cat', OneHotEncoder(drop='first', sparse_output=False), cat_feat)
        ], remainder='passthrough')
    else:
        preprocessing_pipeline = ColumnTransformer([
            ('num', SimpleImputer(strategy='median'), num_feat),
            ('cat', OneHotEncoder(drop='first', sparse_output=False), cat_feat)
        ], remainder='passthrough')

    # Modeling pipeline
    model_pipeline = Pipeline([
        ('preprocessor', preprocessing_pipeline),
        ('rf_clf', RandomForestClassifier(
            criterion='log_loss',
            bootstrap=True,
            n_jobs=-1,
            verbose=0,
            class_weight='balanced_subsample',
            **hyperparameters
        ))
    ])
    # Configure all steps to output pandas dataframes
    model_pipeline.set_output(transform='pandas')

    return model_pipeline

# -------------------------- Optimization objective -------------------------- #

def baseline_objective(trial: optuna.Trial, 
                       aws_params: Dict[str, str],
                       pipeline_func: Callable, 
                       num_feat: List[str], 
                       cat_feat: List[str], 
                       train_data: Tuple[np.ndarray], 
                       test_mode: int,
                       logger: logging.Logger) -> float:
    """
    Surrogate function for optuna to optimize.

    Parameters
    ----------
    trial : optuna.Trial
        Optuna trial object.
    aws_params : Dict[str, str]
        Dictionary containing the AWS parameters: the S3 bucket and key and training job name.
    pipeline_func : Callable
        Function that creates the modeling pipeline.
    num_feat : List[str]
        List of numerical features.
    cat_feat : List[str]
        List of categorical features.
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
        'n_estimators': trial.suggest_int('n_estimators', 100, 1500),
        # As a rule of thumb, log_2(n_samples = 617) = 9.27
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        # Minimum number of samples required to split an internal node
        'min_samples_split': trial.suggest_float('min_samples_split', 0.01, 0.5), 
        # Minimum number of samples required to be at a leaf node 
        'min_samples_leaf': trial.suggest_float('min_samples_leaf', 0.01, 0.5), 
        'min_weight_fraction_leaf': trial.suggest_float('min_weight_fraction_leaf', 0.01, 0.5),
        'max_features': trial.suggest_float('max_features', 0.1, 0.9),
        # This can be dependent on other hyperparameters such as max_depth, min_samples_leaf, etc.
        'max_leaf_nodes': trial.suggest_int('max_leaf_nodes', 10, 300),
        # Similar to early stopping (larger means more conservative)
        'min_impurity_decrease': trial.suggest_float('min_impurity_decrease', 0.01, 0.5),
        'max_samples': trial.suggest_float('max_samples', 0.1, 0.9),
        'ccp_alpha': trial.suggest_float('ccp_alpha', 0.0, 0.2)
    }
    scaling = trial.suggest_categorical('scaling', [1, 0])

    # CV splitter: X_val.shape[0] = total_samples / n_splits and X_val.shape[0] = total_samples - (total_samples / n_splits)
    skf = StratifiedKFold(n_splits=2 if test_mode else 20, shuffle=True)

    feat_imp_handler = FeatureImportanceHandler(s3_key=aws_params['s3_key'], s3_bucket=aws_params['s3_bucket'], job_name=aws_params['job_name'], trial_number=trial.number)
    log_loss_scores = {}
    feature_importances = {}

    for fold, (train_idx, val_idx) in enumerate(skf.split(train_data[0], train_data[1])):

        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        logger.info(f'Training set trial {trial.number} target distribution: {{0: {np.round(np.mean(y_train == 0), 2)}, 1: {np.round(np.mean(y_train == 1), 2)}}}')
        logger.info(f'Validation set trial {trial.number} target distribution: {{0: {np.round(np.mean(y_val == 0), 2)}, 1: {np.round(np.mean(y_val == 1), 2)}}}')

        # Compute sample weights
        sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)

        # Train model on training set
        logger.info(f'Training model for trial {trial.number} fold {fold + 1}...')
        model_pipeline = pipeline_func(model_hyperparameter, num_feat, cat_feat, scaling)
        model_pipeline.fit(X_train, y_train, rf_clf__sample_weight=sample_weights)

        # Compute log loss on validation set
        logger.info(f'Computing log loss for trial {trial.number} fold {fold + 1}...')
        # Obtain the 1-D positive class probabilities
        y_pred = model_pipeline.predict_proba(X_val)[:, 1]
        log_loss_scores[f'fold_{fold + 1}'] = custom_log_loss(y_val, y_pred)
        
        # Feature importances
        logger.info(f'Computing feature importances trial {trial.number} fold {fold + 1}...')
        feat_imp = FeatureImportanceHandler.impure_feat_imp(model_pipeline=model_pipeline)
        feature_importances[f'fold_{fold + 1}'] = feat_imp

    logger.info(f'Uploading feature importances for trial {trial.number} fold {fold + 1}...')
    feat_imp_handler.upload(dictionary=feature_importances)

    # Mean log loss score
    mean_log_loss = np.mean(list(log_loss_scores.values()))
    logger.info(f'Mean log loss for trial {trial.number}: {mean_log_loss}')

    # Delete handler to close boto3 client  
    del feat_imp_handler

    return mean_log_loss

if __name__ == '__main__':

    # ---------------------------------- Set up ---------------------------------- #

    additional_args = {
        'study_name': str
    }

    args = add_additional_args(parser, additional_args)()

    logger = get_logger(name=__name__)

    job_name = args.training_env['job_name']

    # Hydra
    core.global_hydra.GlobalHydra.instance().clear()
    initialize(version_base='1.2', config_path='config', job_name='baseline')
    config = OmegaConf.to_container(compose(config_name='main'), resolve=True)

    # --------------------------------- Load data -------------------------------- #

    logger.info('Loading data...')

    data = pd.read_csv(os.path.join(args.train, 'train.csv'))
    if args.test_mode:
        data = data.sample(300)
    X, y = data.reset_index(drop=True).drop(['Class', 'Id'], axis=1), data['Class'].values

    logger.info(f'Training data shape: {X.shape}')
    logger.info(f'Class distribution: {{0: {np.sum(y == 0)}, 1: {np.sum(y == 1)}}}')

    # ------------------------------ Set up database ----------------------------- #

    logger.info('Setting up optuna database...')

    db_url = get_db_url(host=args.host, db_name=args.db_name, db_secret=args.db_secret, region_name=args.region_name)

    logger.info(f'Database URL: {db_url}')

    # ------------------------------- Optimization ------------------------------- #

    logger.info('Optimizing objective function...')

    def objective_wrapper(trial: optuna.Trial) -> Callable:
        return baseline_objective(
            trial=trial,
            aws_params={
                's3_bucket': config['s3_bucket'],
                's3_key': config['s3_key'],
                'job_name': job_name
            },
            pipeline_func=create_baseline,
            num_feat=list(config['num_feat']),
            cat_feat=list(config['cat_feat']),
            train_data=(X, y),
            test_mode=args.test_mode,
            logger=logger
        )

    study = create_study(study_name=args.study_name, storage=db_url, direction='minimize')
    study.optimize(objective_wrapper, n_trials=args.n_trials, n_jobs=-1)

    study_report(study=study, logger=logger)