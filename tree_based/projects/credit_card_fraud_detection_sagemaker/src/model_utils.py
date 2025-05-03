import argparse
import ast
import json
import logging
import os
import sys
from collections.abc import Callable
from typing import Any, Dict, Optional
from urllib.parse import quote_plus

import boto3
import joblib
import numpy as np
import optuna
import s3fs
from botocore.exceptions import ClientError
from mypy_boto3_secretsmanager.type_defs import GetSecretValueResponseTypeDef
from optuna.trial import TrialState
from sklearn.inspection import permutation_importance
from sklearn.utils import Bunch


def save_or_load_permutation_importance(
    estimator: Any,
    X,
    y,
    model_s3_path: str,
    write_mode: bool,
    scoring_function: Optional[Callable] = None,
    random_state: Optional[int] = None,
) -> None | Bunch:
    """
    Save or load permutation feature importance results for an estimator.

    Parameters
    ----------
    estimator : object
        The trained model or estimator.
    X : pl.DataFrame
        Feature data matrix.
    y : np.ndarray
        Target vector.
    model_s3_path : str
        S3 path to the model file, which should contain the feature importance results.
    write_mode : bool
        If True, compute and save permutation importance. If False, load the results from S3.
    scoring_function : Callable, optional
        Custom scoring function for permutation importance.
    random_state : int, optional
        Random state for reproducibility. Default is None.

    Returns
    -------
    None | Bunch
        Permutation importance results if write_mode is False.
    """
    fs = s3fs.S3FileSystem(anon=False)

    if write_mode:
        if scoring_function is None:
            raise ValueError(
                "scoring_function must be provided when write_mode is True."
            )
        perm_feat_imp_result = permutation_importance(
            estimator=estimator,
            X=X,
            y=y,
            scoring=scoring_function,
            n_repeats=10,
            random_state=random_state,
            n_jobs=-1,
        )
        with fs.open(
            os.path.join(
                os.path.dirname(model_s3_path),
                "permutation_feature_importance.joblib",
            ),
            "wb",
        ) as f:
            joblib.dump(perm_feat_imp_result, f)
        return None
    else:
        with fs.open(
            os.path.join(
                os.path.dirname(model_s3_path),
                "permutation_feature_importance.joblib",
            ),
            "rb",
        ) as f:
            perm_feat_imp_result = joblib.load(f)

        return perm_feat_imp_result


def setup_logger(name: str) -> logging.Logger:
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

    log_format = "%(asctime)s %(levelname)s %(name)s: %(message)s"
    formatter = logging.Formatter(log_format)
    # No matter how many processes we spawn, we only want one StreamHandler attached to the logger
    if not any(
        isinstance(handler, logging.StreamHandler) for handler in logger.handlers
    ):
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.setLevel(logging.INFO)

    return logger


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
    parser.add_argument("--host", type=str)
    parser.add_argument("--db_name", type=str, default="optuna")
    parser.add_argument("--db_secret", type=str, default="optuna/db")
    parser.add_argument("--region_name", type=str, default="us-east-1")
    parser.add_argument("--n_trials", type=int, default=20)

    # Data, model, and output directories
    parser.add_argument("--model_dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--train", type=str, default=os.environ["SM_CHANNEL_TRAIN"])
    parser.add_argument(
        "--training_env", type=str, default=json.loads(os.environ["SM_TRAINING_ENV"])
    )

    parser.add_argument("--test_mode", type=int, default=0)

    return parser


def add_additional_args(
    parser_func: Callable, additional_args: Dict[str, type]
) -> Callable:
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
            parser.add_argument(f"--{arg_name}", type=arg_type)

        args, _ = parser.parse_known_args()

        return args

    return wrapper


def get_secret(secret_name: str, region_name: str = "us-east-1") -> Dict[str, str]:
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
    Dict[str, str]
        Secret retrieved from AWS Secrets Manager.

    Raises
    ------
    ClientError
        If an error occurs on the client side.
    Exception
        If an unknown error occurs.
    """
    # Create a secrets manager client
    session = boto3.session.Session()
    client = session.client(service_name="secretsmanager", region_name=region_name)
    try:
        get_secret_value_response: GetSecretValueResponseTypeDef = (
            client.get_secret_value(SecretId=secret_name)
        )
        # If the secret was a JSON-encoded dictionary string, convert it to dictionary
        if "SecretString" in get_secret_value_response:
            secret = get_secret_value_response["SecretString"]
            secret_dict = ast.literal_eval(secret)  # Convert string to dictionary
            return secret_dict
    except ClientError as error:
        if error.response["Error"]["Code"] == "DecryptionFailureException":
            # Secrets Manager can't decrypt the protected secret text using the provided KMS key
            raise error
        elif error.response["Error"]["Code"] == "InternalServiceErrorException":
            # An error occurred on the server side
            raise error
        elif error.response["Error"]["Code"] == "InvalidParameterException":
            # We provided an invalid value for a parameter
            raise error
        elif error.response["Error"]["Code"] == "InvalidRequestException":
            # We provided a parameter value that is not valid for the current state of the resource
            raise error
        elif error.response["Error"]["Code"] == "ResourceNotFoundException":
            # Can't find the resource that we asked for
            raise error
        else:
            raise error
    except Exception as error:
        raise error


# --------------------- Function for setting up database --------------------- #


def get_db_url(
    host: str, db_name: str, db_secret: str, region_name: str = "us-east-1"
) -> str:
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
    connector = "pymysql"
    user_name = secret["username"]
    password = quote_plus(secret["password"])
    db_url = f"mysql+{connector}://{user_name}:{password}@{host}/{db_name}"

    return db_url


# ------------------------ Function for creating study ----------------------- #


def create_study(
    study_name: str, storage: str, direction: str = "minimize"
) -> optuna.study.Study:
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
        load_if_exists=True,
    )

    return study


def study_report(study: optuna.study.Study, logger: logging.Logger) -> None:
    """
    Report Optuna study results.

    Parameters
    ----------
    study : optuna.study.Study
        Optuna study instance.
    logger : logging.Logger
        The logger object.
    """
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    best_trial = study.best_trial

    logger.info(f"Number of pruned trials: {len(pruned_trials)}")
    logger.info(f"Number of complete trials: {len(complete_trials)}")
    logger.info(f"Best trial score: {best_trial.value}")
    logger.info(f"Best trial params: {best_trial.params}")

    return None


def compute_class_distribution(y: np.ndarray) -> Dict[int, float]:
    """
    Compute the class distribution of a binary target.

    Parameters
    ----------
    y : np.ndarray
        Binary target.

    Returns
    -------
    Dict[int, float]
        Class distribution.
    """
    class_distribution = {0: np.round(1 - np.mean(y), 4), 1: np.round(np.mean(y), 4)}
    return class_distribution
