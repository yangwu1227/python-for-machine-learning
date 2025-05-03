import logging
import pickle
import sys
from re import sub
from typing import Any, List, Optional, Tuple

import boto3
import numpy as np
import pandas as pd
from category_encoders.cat_boost import CatBoostEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import RFE
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier


class FeatureEngine(TransformerMixin, BaseEstimator):
    """
    A custom transformer that engineers new numerical features. It groups by categorical features in the
    data matrix and applies aggregation functions of the numerical features.
    """

    def __init__(self, num_feat: List[str], cat_feat: List[str]):
        """
        Constructor for the FeatureEngine class.

        Parameters
        ----------
        num_feat : List[str]
            List of numerical features in the data matrix.
        cat_feat : List[str]
            List of categorical features in the data matrix.
        """
        self.num_feat = num_feat
        self.cat_feat = cat_feat

    def fit(self, X: pd.DataFrame, y: Optional[np.ndarray] = None):
        """
        Fit the FeatureEngine transformer. This is a no-op.

        Parameters
        ----------
        X : pd.DataFrame
            Data matrix.
        y : np.ndarray, optional
            Ignored, present here for API consistency by convention, by default None

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
        for group in self.cat_feat:
            for agg_func in ["std", "mean", "max", "sum"]:
                X[[col + f"_{agg_func}_by_{group}" for col in self.num_feat]] = (
                    X.groupby(group)[self.num_feat].transform(agg_func)
                )
        return X


def create_pipeline(num_feat: int, step: float) -> Pipeline:
    """
    Function that returns a preprocessing pipeline.

    Parameters
    ----------
    num_feat : int
        Number of features to select.
    step : float
        A value within (0.0, 1.0) that corresponds to the percentage (rounded down) of features to remove at each iteration.

    Returns
    -------
    Pipeline
        An instance of Pipeline class that encapsulates the preprocessing logic.
    """
    # Categorical features for grouping
    cat_feat_list = [
        "gender",
        "offer",
        "phone_service",
        "internet_type",
        "online_security",
        "multiple_lines",
        "online_backup",
        "device_protection_plan",
        "premium_tech_support",
        "contract",
        "paperless_billing",
        "payment_method",
    ]
    # Numerical features to aggregate
    num_feat_list = [
        "age",
        "number_of_referrals",
        "tenure_in_months",
        "avg_monthly_long_distance_charges",
        "avg_monthly_gb_download",
        "monthly_charge",
        "total_charges",
        "total_refunds",
        "total_extra_data_charges",
        "total_long_distance_charges",
        "satisfaction_score",
        "cltv",
    ]
    # Categorical features for encoding
    encode_feat_list = [
        "gender",
        "under_30",
        "senior_citizen",
        "married",
        "dependents",
        "offer",
        "phone_service",
        "multiple_lines",
        "internet_service",
        "internet_type",
        "online_security",
        "online_backup",
        "device_protection_plan",
        "premium_tech_support",
        "streaming_tv",
        "streaming_movies",
        "streaming_music",
        "unlimited_data",
        "contract",
        "paperless_billing",
        "payment_method",
    ]

    preprocessor = Pipeline(
        [
            (
                "feat_engine",
                FeatureEngine(num_feat=num_feat_list, cat_feat=cat_feat_list),
            ),
            ("cat_boost_encode", CatBoostEncoder(cols=encode_feat_list)),
            (
                "rfe",
                RFE(
                    estimator=DecisionTreeClassifier(random_state=None),
                    n_features_to_select=num_feat,
                    step=step,
                ),
            ),
        ]
    )

    return preprocessor


class S3Pickle:
    """
    A class for uploading and downloading Python objects to and from S3.
    """

    def __init__(self, s3_client=None):
        """
        Constructor for S3Pickle class.

        Parameters
        ----------
        s3_client : _type_, optional
            A boto3 S3 client. The default is None.
        """
        if s3_client is None:
            self.s3_client = boto3.client("s3")
        else:
            self.s3_client = s3_client

    def upload_pickle(self, obj: Any, bucket_name: str, key_name: str) -> None:
        """
        Upload a Python object to S3 as a pickle byte string.

        Parameters
        ----------
        obj : Any
            A Python object.
        bucket_name : str
            S3 bucket name.
        key_name : str
            S3 key name.
        """
        # Serialize the object to a pickle byte string
        pickle_byte_string = pickle.dumps(obj)

        # Upload the pickle byte string to S3
        self.s3_client.put_object(
            Body=pickle_byte_string, Bucket=bucket_name, Key=key_name
        )

        return None

    def download_pickle(self, bucket_name: str, key_name: str) -> Any:
        """
        Download a Python object from S3 as a pickle byte string.

        Parameters
        ----------
        bucket_name : str
            S3 bucket name.
        key_name : str
            S3 key name.
        """
        # Download the pickle byte string from S3
        response = self.s3_client.get_object(Bucket=bucket_name, Key=key_name)
        pickle_byte_string = response["Body"].read()

        # Deserialize the pickle byte string to a Python object
        obj = pickle.loads(pickle_byte_string)

        return obj


# ----------------------------------- Data ----------------------------------- #


def load_data(
    data_s3_url: str, logger: Optional[logging.Logger] = None
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
     Load data from S3 bucket and return X and y.

    Parameters
    ----------
    data_s3_url : str
        S3 url of data.
    logger : Optional[logging.Logger]
        Logger object.

    Returns
    -------
    Tuple[pd.DataFrame, np.ndarray]
        Feature matrix and target array.
    """
    data = pd.read_csv(data_s3_url, index_col=0)

    # Drop ID column and 'churn category' column (not useful for prediction)
    data.drop(["Customer ID", "Churn Category"], axis=1, inplace=True)

    # Change column names to lower case and relace white spaces with underscore
    data.columns = [sub(r"\s", "_", col.lower()) for col in data.columns]

    X, y = data.drop(["churn_value"], axis=1), data.churn_value.values

    if logger is not None:
        logger.info("Data Loaded")
        logger.info(f"The shape of training set: {(X.shape, y.shape)}")

    return X, y


# ----------------------- Custom metric for evaluation ----------------------- #


def weighted_ap_score(predt: np.ndarray, data: np.ndarray) -> Tuple[str, float]:
    y_true = data
    y_score = predt
    weighted_ap_score = average_precision_score(
        y_true=y_true, y_score=y_score, average="weighted", pos_label=1
    )
    return "avgAP", weighted_ap_score


# ------------------------ Stratified train/test split ----------------------- #


def stratified_split(
    X_train: pd.DataFrame, y_train: np.ndarray
) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]:
    """
    Split the training set into train and validation sets, stratifying on the target variable.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features.
    y_train : np.ndarray
        Training target.

    Returns
    -------
    Tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]
        X_train, y_train, X_val, y_val.
    """
    ssf = StratifiedShuffleSplit(n_splits=1, test_size=0.2)

    for train_index, val_index in ssf.split(X_train, y_train):
        X_train, X_val = X_train.iloc[train_index], X_train.iloc[val_index]
        y_train, y_val = y_train[train_index], y_train[val_index]

    return X_train, y_train, X_val, y_val


# ---------------------------------- Logger ---------------------------------- #


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
