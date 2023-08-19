import os
import argparse
from typing import Tuple, Union, List, Dict, Any, Optional, Callable
from collections import OrderedDict
import logging
import sys
import json
import base64
import ast
import s3fs
from IPython.display import Image

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Nopep8
import tensorflow as tf

import polars as pl
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
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(log_format))
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
    parser.add_argument('--val', type=str, default=os.environ['SM_CHANNEL_VAL'])
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

# ------------------------------- Data from csv ------------------------------ #

def dataset_from_csv(file_path: str, 
                     config: Dict[str, Any],
                     train: bool,
                     batch_size: int,
                     **kwargs) -> Tuple[int, tf.data.Dataset]:
    """
    Create tf.data.Dataset from csv file.

    Parameters
    ----------
    file_path : str
        Path to the csv file.
    config: Dict[str, Any]
        Configuration dictionary.
    train : bool
        Whether to create training dataset (shuffle) or validation or test (no shuffle).
    batch_size: int
        Batch size.
    **kwargs
        Keyword arguments to pass to tf.data.experimental.make_csv_dataset.

    Returns
    -------
    Tuple[int, tf.data.Dataset]
        Tuple of number of batches and tf.data.Dataset.
    """
    target_label_lookup = tf.keras.layers.StringLookup(
        vocabulary=config['tf_keras']['target_labels'], 
        # OOV inputs will cause an error when calling the layer
        num_oov_indices=config['tf_keras']['num_oov_indices'],
        mask_token=None
    )

    # List of default values based on feature dtype
    column_defaults = [
        config['tf_keras']['default_num'] if name in config['tf_keras']['num_feat'] + [config['tf_keras']['weight_feat']] else config['tf_keras']['default_cat']
        for name in config['csv_header']
    ]

    def process_sample(features: OrderedDict, label: tf.Tensor) -> Tuple[OrderedDict, tf.Tensor, tf.Tensor]:
        """
        Function to process samples. The reason for defining this function inside
        dataset_from_csv's local scope is that it needs to access the target_label_lookup 
        layer and the weight feature name.

        Parameters
        ----------
        features : OrderedDict
            Dictionary of feature names -> tensors.
        label : tf.Tensor
            Tensor of labels.

        Returns
        -------
        Tuple[OrderedDict, tf.Tensor, tf.Tensor]
            Tuple of features, labels, and weights.
        """
        # Integer encode the label
        label = target_label_lookup(label)
        # Remove weight column in place and assign at the same time
        weights = features.pop(config['tf_keras']['weight_feat'])
        return features, label, weights

    dataset = tf.data.experimental.make_csv_dataset(
        file_pattern=file_path,
        batch_size=batch_size,
        column_names=config['csv_header'],
        column_defaults=column_defaults,
        header=config['header'],
        label_name=config['target'],
        num_epochs=config['tf_keras']['num_epochs'],
        na_value=config['tf_keras']['na_value'],
        shuffle=True if train else False,
        **kwargs
    )

    # Number of batches
    num_batches = dataset.reduce(0, lambda x, _: x + 1).numpy()

    # Apply process_sample function to each batch
    dataset = dataset.map(
        map_func=process_sample,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=config['tf_keras']['deterministic']
    )

    return num_batches, dataset.cache()

# ----------------------------- Stratified sample ---------------------------- #

def stratified_sample(data: pl.DataFrame, cat_feat: List[str]) -> pl.DataFrame:
    """
    Stratified sample of a dataframe ensuring representation of all unique values from 
    categorical features. This is useful for creating a small sample of the data for
    testing purposes. If not all unique values are present in the dataframe, then the
    'tf.keras.layers.StringLookup' layer will throw an error.

    Parameters
    ----------
    data : pl.DataFrame
        The original dataframe from which to sample.
    cat_feat : List[str]
        List of the names of the categorical feature columns.

    Returns
    -------
    pl.DataFrame
        A new dataframe which is a stratified sample from the original dataframe.
    """
    # Initialize an empty DataFrame to store the result
    samples = []
    
    # Iterate over each categorical feature
    for feat in cat_feat:
        # Get unique values in the categorical feature
        unique_values = data[feat].unique()
        
        # Iterate over each unique value and get a sample for each
        for value in unique_values:
            sample = data.filter(data[feat] == value).sample(n=1)
            samples.append(sample)
            # If there is '?'in this categorical feature, then sample it too
            na_data = data.filter(data[feat] == '?')
            if na_data.shape[0] != 0:
                na_sample = na_data.sample(n=1)
                samples.append(na_sample)

    # Row-bind all the samples together
    result = pl.concat(samples, how='vertical')
    
    return result

# ---------------------------- Sample for testing ---------------------------- #

def test_sample(file_path: str, 
                config: Dict[str, Any],
                train: bool,
                batch_size: int,
                **kwargs) -> Tuple[int, tf.data.Dataset]:
    """
    Read csv files directly from S3, stratified sample, and create tf.data.Dataset. 
    This dataset is used for testing purposes using SageMaker local mode.

    Parameters
    ----------
    file_path : str
        S3 Path to the csv file.
    config: Dict[str, Any]
        Configuration dictionary.
    train : bool
        Whether to create training dataset (shuffle) or validation or test (no shuffle).
    batch_size: int
        Batch size.
    **kwargs
        Keyword arguments to pass to tf.data.experimental.make_csv_dataset.

    Returns
    -------
    Tuple[int, tf.data.Dataset]
        Tuple of number of batches and tf.data.Dataset.
    """
    s3_fs = s3fs.S3FileSystem(anon=False)
    with s3_fs.open(file_path, 'rb') as f:
        data = pl.read_csv(f, new_columns=config['csv_header'])

    sampled_data = stratified_sample(data, list(config['tf_keras']['cat_feat_vocab'].keys()))

    sampled_data.write_csv(
        '/tmp/sample.csv',
        has_header=config['header']
    )

    num_batches, dataset = dataset_from_csv('/tmp/sample.csv', config, train, batch_size, **kwargs)

    return num_batches, dataset