import argparse
import json
import logging
import operator
import os
import sys
from itertools import combinations
from typing import Callable, Dict, List, Union

import cudf
import cupy as cp
from cuml import TruncatedSVD
from cuml.preprocessing import MaxAbsScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

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

    # AWS
    parser.add_argument("--s3_key", type=str, default="otto-product-classification")
    parser.add_argument("--s3_bucket", type=str, default="yang-ml-sagemaker")

    # Data, model, and output directories
    parser.add_argument("--model_dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--train", type=str, default=os.environ["SM_CHANNEL_TRAIN"])
    parser.add_argument(
        "--training_env", type=str, default=json.loads(os.environ["SM_TRAINING_ENV"])
    )

    # Other parameters
    parser.add_argument("--local_test_mode", type=int, default=0)

    return parser


# ------ Function decorator for adding additional command line arguments ----- #


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


# ---------------------------- Custom transformer ---------------------------- #


class FeatureEngine(TransformerMixin, BaseEstimator):
    """
    A custom transformer that engineers new numerical features. It create pairwise interactions between the top 5
    most importance features (based on gains) identified using the baseline XGBoost model. Next, it creates
    polynomial features for the top 15 most important features. Finally, it engineers new features by taking the
    median, max, standard deviation, and sum of the top 5 and top 15 most important features.
    """

    def __init__(self, inter_feat: List[str], poly_feat: List[str]):
        """
        Constructor for the FeatureEngine class.

        Parameters
        ----------
        inter_feat : List[str]
            List of features to generate interactions in the data matrix.
        poly_feat : List[str]
            List of features to generate polynomials in the data matrix.
        """
        self.inter_feat = inter_feat
        self.poly_feat = poly_feat

    def fit(self, X: cudf.DataFrame, y: Union[cp.ndarray, cudf.Series] = None):
        """
        Fit the FeatureEngine transformer. This is a no-op.

        Parameters
        ----------
        X : cudf.DataFrame
            Data matrix.
        y : Union[cp.ndarray, cudf.Series], optional
            Ignored, present here for API consistency by convention, by default None.

        Returns
        -------
        self: FeatureEngine
            A fitted FeatureEngine transformer.
        """
        return self

    def transform(self, X: cudf.DataFrame) -> cudf.DataFrame:
        """
        Transform the data matrix by engineering new features.

        Parameters
        ----------
        X : cudf.DataFrame
            Data matrix.

        Returns
        -------
        cudf.DataFrame
            Transformed data matrix.
        """
        X = X.copy()

        # Polynomial features with out interactions
        X[[col + "_squared" for col in self.poly_feat]] = X[self.poly_feat].pow(2)
        X[[col + "_cubed" for col in self.poly_feat]] = X[self.poly_feat].pow(3)
        X[[col + "_sqr_root" for col in self.poly_feat]] = X[self.poly_feat].pow(1 / 2)

        # Math operations for the top 5 most important features
        X["top_five_product"] = X[self.inter_feat].prod(axis=1)
        X["top_five_sum"] = X[self.inter_feat].sum(axis=1)
        X["top_five_max"] = X[self.inter_feat].max(axis=1)
        X["top_five_median"] = X[self.inter_feat].median(axis=1)
        X["top_five_std"] = X[self.inter_feat].std(axis=1)

        # Math operations for the top 15 most important features
        X["top_fifteen_product"] = X[self.poly_feat].prod(axis=1)
        X["top_fifteen_sum"] = X[self.poly_feat].sum(axis=1)
        X["top_fifteen_max"] = X[self.poly_feat].max(axis=1)
        X["top_fifteen_median"] = X[self.poly_feat].median(axis=1)
        X["top_fifteen_std"] = X[self.poly_feat].std(axis=1)

        # List of tuples (col_i, col_j) for top 5 most important features
        col_pairs = list(combinations(self.inter_feat, 2))

        py_operators = {"add": operator.add, "sub": operator.sub, "mul": operator.mul}

        # Pairwise interactions for the top 5 most important features
        for func_key in py_operators:
            for col_i, col_j in col_pairs:
                X[f"{col_i}_{func_key}_{col_j}"] = py_operators[func_key](
                    X[col_i], X[col_j]
                )

        return X


# --------------------------- Function for pipeline -------------------------- #


def create_pipeline(
    inter_feat: List[str], poly_feat: List[str], svd: bool, n_components: int
) -> Pipeline:
    """
    Create a pipeline for the model.

    Parameters
    ----------
    inter_feat : List[str]
        List of features to generate interactions in the data matrix.
    poly_feat : List[str]
        List of features to generate polynomials in the data matrix.
    svd : bool
        Whether to apply TruncatedSVD to the data matrix.
    n_components : int
        Number of components to keep after applying TruncatedSVD.

    Returns
    -------
    Pipeline
        A pipeline that transforms the data matrix but does not include a final estimator.
    """
    pipeline = Pipeline([("feature_engine", FeatureEngine(inter_feat, poly_feat))])

    if svd:
        pipeline.steps.append(("scaler", MaxAbsScaler()))
        pipeline.steps.append(("svd", TruncatedSVD(n_components=n_components)))

    return pipeline
