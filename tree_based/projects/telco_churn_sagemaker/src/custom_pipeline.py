from typing import List

import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from category_encoders.cat_boost import CatBoostEncoder
from sklearn.feature_selection import RFE
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

    def fit(self, X: pd.DataFrame, y: np.ndarray = None):
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
