import argparse
import json
import logging
import os
import sys
from collections.abc import Callable
from typing import Dict, Tuple

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Nopep8
import numpy as np

# ------------------------------ Logger function ----------------------------- #


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


# ------------------- Parse arguments from the command line ------------------ #


def parser() -> argparse.ArgumentParser:
    """
    Parse arguments from the command line.

    Returns
    -------
    argparse.ArgumentParser
        The parser object.
    """
    parser = argparse.ArgumentParser()

    # Data, model, and output directories
    parser.add_argument("--model_dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--train", type=str, default=os.environ["SM_CHANNEL_TRAIN"])
    parser.add_argument(
        "--training_env", type=str, default=json.loads(os.environ["SM_TRAINING_ENV"])
    )

    # Other
    parser.add_argument("--test_mode", type=int, default=0)

    return parser


# --------- Decorator for adding additional arguments to base parser --------- #


def add_additional_args(parser_func: Callable, additional_args: Dict[str, type]):
    """
    Add additional arguments to the parser function, which returns the parser, that are specific to each script.
    This function decorator returns a callable parser function that can be called to parse additional arguments,
    and finaly returning the namespace object containging those arguments.

    Parameters
    ----------
    parser_func : Callable
        The base parser function.
    additional_args : Dict[str, type]
        A dictionary with the additional arguments to add to the parser function. Each key is the name of the argument and the value is the type of the argument, e.g. {'arg1': str, 'arg2': int}.

    Returns
    -------
    Callable
        The parser function with additional arguments.
    """

    def wrapper():
        # Call the original parser function to get the parser object
        parser = parser_func()

        for arg_name, arg_type in additional_args.items():
            parser.add_argument(f"--{arg_name}", type=arg_type)

        args, _ = parser.parse_known_args()

        return args

    return wrapper


# ------------------------------ Standardization ----------------------------- #


def standardize(
    X: np.ndarray, y: np.ndarray
) -> Tuple[Tuple[np.ndarray, np.ndarray], Dict[str, float]]:
    """
    This function standardizes the training data and the target by subtracting the mean and dividing by
    the standard deviation. The mean and standard deviation are computed by flattening the training data
    and the target into 1D-vectors (densities across all counties). The two statistics are stored in a
    dictionary and returned along with the standardized training data and target.

    Parameters
    ----------
    X : np.ndarray
        The training data.
    y : np.ndarray
        The target.

    Returns
    -------
    Tuple[Tuple[np.ndarray, np.ndarray], Dict[str, float]]
        A tuple of numpy arrays representing the standardized training data and the target
        and a dictionary containing the mean and standard deviation.
    """
    mean_ratio = np.mean(X)
    std_ratio = np.std(X)

    X_train_out = (X - mean_ratio) / std_ratio
    y_train_out = (y - mean_ratio) / std_ratio

    return (X_train_out, y_train_out), {"mean": mean_ratio, "std": std_ratio}
