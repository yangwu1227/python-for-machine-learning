import os
import argparse
from typing import Tuple, Union, List, Dict, Any, Optional, Callable
import logging
import sys
import json
import pickle
import base64
import ast
from IPython.display import Image
from itertools import combinations
import operator

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer
from sklearn.inspection import permutation_importance

import numpy as np
import pandas as pd
import optuna 
from optuna.trial import TrialState
import boto3
from botocore.exceptions import ClientError

# ---------------------------------- Logger ---------------------------------- #

def get_logger(name: str) -> logging.Logger:
    """
    Parameters
    ----------
    name : str
        A string that specifies the name of the logger.

    Returns
    -------
    logging.Logger
        A logger with the specified name.
    """
    logger = logging.getLogger(name)  # Return a logger with the specified name
    
    log_format = '%(asctime)s %(levelname)s %(name)s: %(message)s'
    formatter = logging.Formatter(log_format)
    # No matter how many processes we spawn, we only want one StreamHandler attached to the logger
    if not any(isinstance(handler, logging.StreamHandler) for handler in logger.handlers):
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.setLevel(logging.INFO)
    
    return logger

# --------------------- Parse argument from command line --------------------- #

def parser() -> argparse.ArgumentParser:
    """
    Function that parses arguments from command line.

    Returns
    -------
    argparse.ArgumentParser
        An ArgumentParser object that contains the arguments passed from command line.
    """
    parser = argparse.ArgumentParser()
    
    # Optuna database
    parser.add_argument('--host', type=str)
    parser.add_argument('--db_name', type=str, default='optuna')
    parser.add_argument('--db_secret', type=str, default='optuna/db')
    parser.add_argument('--region_name', type=str, default='us-east-1')
    parser.add_argument('--n_trials', type=int, default=20)

    # Data, model, and output directories 
    parser.add_argument('--model_dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--training_env', type=str, default=json.loads(os.environ['SM_TRAINING_ENV']))

    parser.add_argument('--test_mode', type=int, default=0)

    return parser

# ------ Function decorator for adding additional command line arguments ----- #

def add_additional_args(parser_func: Callable, additional_args: Dict[str, type]) -> Callable:
    """
    Function decorator that adds additional command line arguments to the parser.
    This allows for adding additional arguments without having to change the base
    parser.

    Parameters
    ----------
    parser_func : Callable
        The parser function to add arguments to.
    additional_args : Dict[str, type]
        A dictionary where the keys are the names of the arguments and the values
        are the types of the arguments, e.g. {'arg1': str, 'arg2': int}.

    Returns
    -------
    Callable
        A parser function that returns the ArgumentParser object with the additional arguments added to it.
    """
    def wrapper():
        # Call the original parser function to get the parser object
        parser = parser_func()

        for arg_name, arg_type in additional_args.items():
            parser.add_argument(f'--{arg_name}', type=arg_type)

        args, _ = parser.parse_known_args()

        return args

    return wrapper

# ----------------------- Function for database secret ----------------------- #

def get_secret(secret_name: str, region_name: str = 'ur-east-1') -> Union[Dict, bytes]:
    """
    Get secret from AWS Secrets Manager.

    Parameters
    ----------
    secret_name : str
        Name of the secret to retrieve.
    region_name : str, optional
        Region, by default 'ur-east-1'

    Returns
    -------
    Union[Dict, bytes]
        Secret retrieved from AWS Secrets Manager.
    """
    # Create a secrets manager client
    session = boto3.session.Session()
    client = session.client(
        service_name='secretsmanager', 
        region_name=region_name
    )
    try:
        get_secret_value_response = client.get_secret_value(SecretId=secret_name)
    except ClientError as e:
        if e.response['Error']['Code'] == 'DecryptionFailureException':
            # Secrets Manager can't decrypt the protected secret text using the provided KMS key
            raise e
        elif e.response['Error']['Code'] == 'InternalServiceErrorException':
            # An error occurred on the server side
            raise e
        elif e.response['Error']['Code'] == 'InvalidParameterException':
            # We provided an invalid value for a parameter
            raise e
        elif e.response['Error']['Code'] == 'InvalidRequestException':
            # We provided a parameter value that is not valid for the current state of the resource
            raise e
        elif e.response['Error']['Code'] == 'ResourceNotFoundException':
            # Can't find the resource that we asked for
            raise e
    else:
        # If the secret was a JSON-encoded dictionary string, convert it to dictionary
        if 'SecretString' in get_secret_value_response:
            secret = get_secret_value_response['SecretString']
            secret = ast.literal_eval(secret) # Convert string to dictionary
            return secret
        # If the secret was binary, decode it 
        else:
            decoded_binary_secret = base64.b64decode(get_secret_value_response['SecretBinary'])
            return decoded_binary_secret

# --------------------- Function for setting up database --------------------- #

def get_db_url(host: str, db_name: str, db_secret: str, region_name: str = 'us-east-1') -> str:
    """
    Set up database for Optuna.

    Parameters
    ----------
    host : str
        Host name of the database.
    db_name : str
        Name of the database.
    db_secret : str
        Name of the secret that contains the database credentials.
    region_name : str, optional
        Region, by default 'us-east-1'.

    Returns
    -------
    str
        Database URL.
    """
    secret = get_secret(db_secret, region_name)
    connector = 'pymysql'
    user_name = secret['username']
    password = secret['password']
    db_url = f'mysql+{connector}://{user_name}:{password}@{host}/{db_name}'

    return db_url

# ------------------------- Custom log loss function ------------------------- #

def custom_log_loss(y_true: np.array, y_pred: np.array) -> float:
    """
    Custom log loss function. Note that this function expects a 1-D
    array for both y_true and y_pred. In the case of y_pred, the
    probabilities of the positive class should be passed.

    Parameters
    ----------
    y_true : np.array
        The 1-D true labels.
    y_pred : np.array
        The 1-D predicted probabilities of the positive class.

    Returns
    -------
    float
        The log loss.
    """
    y_true = y_true.copy()
    # One hot encode the true labels
    y_true = np.eye(2)[y_true.astype(int)]

    # Clip probabilities to avoid 1 or 0, where log loss is undefined
    eps = np.finfo(y_pred.dtype).eps
    y_pred = np.clip(y_pred, eps, 1 - eps)

    # Compute the log loss for each class
    loss_0 = - np.sum(y_true[:, 0] * np.log((1 - y_pred))) / y_true[:, 0].sum()
    loss_1 = - np.sum(y_true[:, 1] * np.log(y_pred)) / y_true[:, 1].sum()

    # Compute the average log loss
    log_loss = (loss_0 + loss_1) / 2

    return log_loss

# ----------------------- Class for feature engineering ---------------------- #

class FeatureEngine(TransformerMixin, BaseEstimator):
    """
    A custom transformer that engineers new numerical features. It create pairwise interactions between the top 5
    most importance features (based on impurity-based feature importance) identified using the baseline random forest
    model. Next, it creates polynomial features for the top 15 most important features. Finally, it engineers new features 
    by taking the median, max, standard deviation, and sum of the top 5 and top 15 most important features. 
    """
    def __init__(self, top_5_feat: List[str], top_15_feat: List[str], cat_feat: List[str]):
        """
        Constructor for the FeatureEngine class.

        Parameters
        ----------
        top_5_feat : List[str]
            List of the top 5 most important features.
        top_15_feat : List[str]
            List of the top 15 most important features.
        cat_feat : List[str]
            List of categorical features.
        """
        self.top_5_feat = top_5_feat
        self.top_15_feat = top_15_feat
        self.cat_feat = cat_feat
        
    def fit(self, X: pd.DataFrame, y: Union[np.ndarray, pd.Series] = None):
        """
        Fit the FeatureEngine transformer. This is a no-op.

        Parameters
        ----------
        X : pd.DataFrame
            Data matrix.
        y : Union[np.ndarray, pd.Series], optional
            Ignored, present here for API consistency by convention, by default None.

        Returns
        -------
        self: FeatureEngine
            A fitted FeatureEngine transformer.
        """
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the data matrix by engineering new features.

        Parameters
        ----------
        X : pd.DataFrame
            Data matrix.

        Returns
        -------
        pd.DataFrame
            Transformed data matrix.
        """
        X = X.copy()
        
        # Polynomial features without interactions for top 5 features
        X[[col + '_squared' for col in self.top_5_feat]] = X[self.top_5_feat].pow(2)
        X[[col + '_cubed' for col in self.top_5_feat]] = X[self.top_5_feat].pow(3)
        X[[col + '_sqrt' for col in self.top_5_feat]] = X[self.top_5_feat].pow(0.5)
        
        # Math operations for the top 5 most important features
        X['top_five_product'] = X[self.top_5_feat].prod(axis=1)
        X['top_five_sum'] = X[self.top_5_feat].sum(axis=1)
        X['top_five_median'] = X[self.top_5_feat].median(axis=1)
        X['top_five_max'] = X[self.top_5_feat].max(axis=1)
        X['top_five_std'] = X[self.top_5_feat].std(axis=1)

        # Math operations for the top 15 most important features
        X['top_fifteen_product'] = X[self.top_15_feat].prod(axis=1)
        X['top_fifteen_sum'] = X[self.top_15_feat].sum(axis=1)
        X['top_fifteen_max'] = X[self.top_15_feat].max(axis=1)
        X['top_fifteen_median'] = X[self.top_15_feat].median(axis=1)
        X['top_fifteen_std'] = X[self.top_15_feat].std(axis=1)

        # Group by categorical feature and apply aggregations to the top 5 most important features
        for group in self.cat_feat:
            for agg_func in ['mean', 'max', 'sum']:
                X[[col + f'_{agg_func}_by_{group}' for col in self.top_5_feat]] = X.groupby(group)[self.top_5_feat].transform(agg_func)
    
        # List of tuples (col_i, col_j) for top 5 most important features
        col_pairs = list(combinations(self.top_5_feat, 2))
        # List of tuples (col_q, col_t, col_k) for top 5 most important features
        col_triplets = list(combinations(self.top_5_feat, 3))

        py_operators = {
            'add': operator.add,
            'sub': operator.sub, 
            'mul': operator.mul
        }

        # Calculate the number of columns for pairwise and triplet interactions
        num_pairwise_cols = len(py_operators) * len(col_pairs)
        num_triplet_cols = len(py_operators) * len(col_triplets)

        # Create column names for pairwise and triplet interactions
        pairwise_cols = [f'{col_i}_{func_key}_{col_j}' for func_key in py_operators for col_i, col_j in col_pairs]
        triplet_cols = [f'{col_q}_{func_key}_{col_t}_{func_key}_{col_k}' for func_key in py_operators for col_q, col_t, col_k in col_triplets]

        # Preallocate memory for pairwise and triplet interactions
        pairwise_interactions = pd.DataFrame(index=X.index, columns=pairwise_cols, dtype=float)
        triplet_interactions = pd.DataFrame(index=X.index, columns=triplet_cols, dtype=float)

        # Pairwise interactions for the top 5 most important features
        for func_key in py_operators:
            for col_i, col_j in col_pairs:
                pairwise_interactions[f'{col_i}_{func_key}_{col_j}'] = py_operators[func_key](X[col_i], X[col_j])

        # Triplet interactions for the top 5 most important features
        for func_key in py_operators:
            for col_q, col_t, col_k in col_triplets:
                triplet_interactions[f'{col_q}_{func_key}_{col_t}_{func_key}_{col_k}'] = py_operators[func_key](X[col_q], py_operators[func_key](X[col_t], X[col_k]))

        # Concatenate the original DataFrame with the new interaction DataFrames
        X = pd.concat([X, pairwise_interactions, triplet_interactions], axis=1)

        return X

# ------------------------ Function to create pipeline ----------------------- #

def create_preprocessor(top_5_feat: List[str], top_15_feat: List[str], num_feat: List[str], cat_feat: List[str]) -> Pipeline:
    """
    Create a preprocessing pipeline.

    Parameters
    ----------
    top_5_feat : List[str]
        List of the top 5 most important features.
    top_15_feat : List[str]
        List of the top 15 most important features.
    num_feat : List[str]
        List of numerical features.
    cat_feat : List[str]
        List of categorical features.
    
    Returns
    -------
    Pipeline
        A sklearn pipeline.
    """
    # Preprocessing pipeline
    preprocessor = ColumnTransformer([
        ('num', SimpleImputer(strategy='median'), num_feat),
        ('cat', OrdinalEncoder(dtype=np.int16, handle_unknown='use_encoded_value', unknown_value=-999, encoded_missing_value=-999), cat_feat)
    ], remainder='passthrough')
    # Configure all preprocessing steps to output pandas dataframes
    preprocessor.set_output(transform='pandas')
    # Feature engine expects a pandas dataframe
    cat_feat = [f'cat__{col}' for col in cat_feat]
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('feature_engine', FeatureEngine(top_5_feat, top_15_feat, cat_feat))
    ])

    return pipeline

# ------- Class for uploading and downloading dictionary to and from S3 ------ #

class FeatureImportanceHandler:
    """
    Class for uploading and downloading feature importance dictionary to and from S3.
    """
    def __init__(self, s3_key: str, s3_bucket: str, job_name: str, trial_number: int) -> None:
        """
        Parameters
        ----------
        s3_key : str
            S3 key.
        s3_bucket : str
            S3 bucket.
        job_name : str
            Training job name to differentiate between different training jobs.
        trial_number : int
            Trial number to differentiate between different trials.
        """
        self.s3_key = s3_key
        self.s3_bucket = s3_bucket
        self.job_name = job_name
        self.client = boto3.client('s3')
        self.trial_number = trial_number

    @staticmethod
    def perm_feat_imp(model_pipeline: Pipeline, scorer: Callable, n_repeats: int, val_data: Tuple[np.ndarray]) -> Dict[str, float]:
        """
        Compute the permutation feature importance for a given model pipeline over 10 iterations.
        
        Parameters
        ----------
        model_pipeline : Pipeline
            Model pipeline (must be fitted).
        scorer : Callable
            Scorer function.
        n_repeats : int
            Number of iterations for computing the permutation feature importance.
        val_data : Tuple[np.ndarray]
            Validation data.

        Returns
        -------
        Dict[str, float]
            Dictionary of feature importances where the keys are the feature names and the values are the feature importances.
        """
        feature_names = model_pipeline.named_steps['rf_clf'].feature_names_in_
        log_loss_scorer = make_scorer(
            score_func=scorer, 
            greater_is_better=False, # Loss loss is a loss function (minimize)
            needs_proba=True
        )

        perm_imp_result = permutation_importance(
            estimator=model_pipeline,
            X=val_data[0],
            y=val_data[1],
            scoring=log_loss_scorer,
            n_repeats=n_repeats
        )

        perm_imp_dict = dict(zip(feature_names, perm_imp_result['importances_mean']))

        return perm_imp_dict

    @staticmethod
    def impure_feat_imp(model_pipeline: Pipeline) -> Dict[str, float]:
        """
        Extract the impurity-based feature importance for a given model pipeline.

        Parameters
        ----------
        model_pipeline : Pipeline
            Model pipeline (must be fitted).

        Returns
        -------
        Dict[str, float]
            Dictionary of feature importances where the keys are the feature names and the values are the feature importances.
        """
        feature_names = model_pipeline.named_steps['rf_clf'].feature_names_in_
        impurity_imp_dict = dict(zip(feature_names, model_pipeline.named_steps['rf_clf'].feature_importances_))

        return impurity_imp_dict

    def upload(self, dictionary: Dict[str, Any]) -> None:
        """
        Upload feature dictionary to S3.

        Parameters
        ----------
        dictionary : Dict[str, Any]
            Dictionary to upload.
        """
        self.client.put_object(
            Bucket=self.s3_bucket,
            Key=f'{self.s3_key}/eda/{self.job_name}-trial-{self.trial_number}-feature-importance.pickle',
            Body=pickle.dumps(dictionary)
        )

        return None

    def download(self) -> Dict[str, Any]:
        """
        Download dictionary from S3.

        Returns
        -------
        Dict[str, Any]
            Dictionary downloaded from S3.
        """
        dictionary = pickle.loads(
            self.client.get_object(
                Bucket=self.s3_bucket,
                Key=f'{self.s3_key}/eda/{self.job_name}-trial-{self.trial_number}-feature-importance.pickle'
                )['Body'].read()
        )

        return dictionary

    def __del__(self) -> None:
        """
        When the object is deleted, close the boto3 s3 client.
        """
        self.client.close()

        return None

# ------------------------ Function for creating study ----------------------- #

def create_study(study_name: str, storage: str, direction: str = 'minimize') -> optuna.study.Study:
    """
    Create Optuna study instance.
    
    Parameters
    ----------
    study_name : str
        Name of the study.
    storage : str
        Database url.
    direction: str
        Direction of the metric--- maximize or minimize.
        
    Returns
    -------
    optuna.study.Study
        Optuna study instance.
    """
    study = optuna.create_study(
        storage=storage,
        sampler=optuna.samplers.TPESampler(),
        study_name=study_name,
        direction=direction,
        load_if_exists=True
    )

    return study

# ------------------- Function for reporting study results ------------------- #

def study_report(study: optuna.study.Study, logger: logging.Logger) -> None:
    """
    Report study results.

    Parameters
    ----------
    study : optuna.study.Study
        Optuna study instance.
    logger : logging.Logger
        The logger object.
    """
    pruned_trials = study.get_trials(
        deepcopy=False,
        states=[TrialState.PRUNED]
    )
    complete_trials = study.get_trials(
        deepcopy=False,
        states=[TrialState.COMPLETE]
    )

    best_trial = study.best_trial
    
    logger.info(f'Number of pruned trials: {len(pruned_trials)}')
    logger.info(f'Number of complete trials: {len(complete_trials)}')
    logger.info(f'Best trial score: {best_trial.value}')
    logger.info(f'Best trial params: {best_trial.params}')

    return None

# ---------------- Class for visualizing hyperparameter tuning --------------- #

class StudyVisualizer:
    """
    Class for visualizing hyperparameter tuning via Optuna
    """

    def __init__(self, study: optuna.study.Study) -> None:
        """
        Parameters
        ----------
        study : optuna.study.Study
            Optuna study instance.
        """
        self.study = study
        self.plot_func_dict = plot_functions = {
            'plot_optimization_history': optuna.visualization .plot_optimization_history,
            'plot_slice': optuna.visualization .plot_slice,
            'plot_parallel_coordinate': optuna.visualization .plot_parallel_coordinate,
            'plot_contour': optuna.visualization .plot_contour,
            'plot_param_importances': optuna.visualization .plot_param_importances
        }

    def _static_plot(self, plot_func: str, figsize: Tuple[float, float], **kwargs) -> Image:
        """
        Create static plot.

        Parameters
        ----------
        figsize : Tuple[float, float]
            Figure size.
        **kwargs
            Keyword arguments to pass to the plot function.
        """
        fig = self.plot_func_dict[plot_func](self.study, **kwargs)
        fig.update_layout(width=figsize[0], height=figsize[1])
        fig_bytes = fig.to_image(format='png')

        return Image(fig_bytes)

    def plot_optimization_history(self, figsize: Tuple[float]) -> Image:
        """
        Plot optimization history.

        Parameters
        ----------
        figsize : Tuple[float]
            Figure size.

        Returns
        -------
        Image
            Image of the plot.
        """
        return self._static_plot('plot_optimization_history', figsize)

    def plot_param_importances(self, figsize: Tuple[float]) -> Image:
        """
        Plot parameter importances.

        Parameters
        ----------
        figsize : Tuple[float]
            Figure size.

        Returns
        -------
        Image
            Image of the plot.
        """
        return self._static_plot('plot_param_importances', figsize)

    def plot_parallel_coordinate(self, params: List[str], figsize: Tuple[float]) -> Image:
        """
        Plot parallel coordinate.

        Parameters
        ----------
        params : List[str]
            List of parameters to plot.
        figsize : Tuple[float]
            Figure size.

        Returns
        -------
        Image
            Image of the plot.
        """
        return self._static_plot('plot_parallel_coordinate', figsize, params=params)

    def plot_contour(self, params: List[str], figsize: Tuple[float]) -> Image:
        """
        Plot contour.

        Parameters
        ----------
        params : List[str]
            List of parameters to plot.
        figsize : Tuple[float]
            Figure size.
        """
        return self._static_plot('plot_contour', figsize, params=params)

    def plot_slice(self, params: List[str], figsize: Tuple[float]) -> Image:
        """
        Plot slice.

        Parameters
        ----------
        params : List[str]
            List of parameters to plot.
        figsize : Tuple[float]
            Figure size.

        Returns
        -------
        Image
            Image of the plot.
        """
        return self._static_plot('plot_slice', figsize, params=params)