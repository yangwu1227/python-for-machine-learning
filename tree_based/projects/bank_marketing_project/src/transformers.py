# ---------------------------------- Imports --------------------------------- #

import numpy as np
import pandas as pd
from feature_engine.encoding import CountFrequencyEncoder, OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import FunctionTransformer

# --------------------------- Categorical features --------------------------- #


def cat_feature_engineer(X, encode_type="count"):
    """
    This function assumes that the dataframe has already been processed such that select_dtypes('object') returns only string columns.
    In practice, we may need more upstream processing to ensure this is the case.

    We can use this function to engineer categorical features in two different ways:

    - Count: Replace the string with the count of that string in the column

    - One-hot: Replace the string with a one-hot encoding of that string in the column

    The 'encode_type' argument can then be a hyperparameter to the pipeline.
    """
    # Categorical features
    cat_features = X.select_dtypes("object").columns.tolist()

    if encode_type == "count":
        return CountFrequencyEncoder(
            variables=cat_features, encoding_method="count"
        ).fit_transform(X)
    elif encode_type == "onehot":
        return OneHotEncoder(
            variables=cat_features, drop_last=True, drop_last_binary=True
        ).fit_transform(X)
    else:
        raise ValueError('encode_type must be "count" or "onehot"')


# ---------------------------- Numerical features ---------------------------- #


def num_feature_engineer(X, switch=True):
    """
    This function assumes that the dataframe has already been processed such that select_dtypes('number') returns only numerical columns.
    In practice, we need to include all the processing in the 'eda.ipynb' notebook in the pipeline, especially when new
    numerical feature may be added in the future. The 'switch' argument is a hyperparameter to the pipeline, turning this transformation on or
    off.
    """
    if switch:
        # Numerical features
        num_features = X.select_dtypes("number").columns.tolist()
        num_features.remove("age")

        # Group by client 'age' and generate new columns ['mean', 'std', 'min', 'max', 'last'] for each col in 'num_features' (5 x len(num_features) total)
        # Could also use df.groupby("customer_ID")[num_features].describe(), but that includes quantiles as well so it may more more wasteful
        num_agg = X.groupby("age")[num_features].agg(
            ["mean", "std", "min", "max", "last"]
        )
        # The num_agg.columns is a 'pandas.core.indexes.multi.MultiIndex' instance (an iterable)
        # Its iterator returns tuples of the form ('parent_col', 'child_col') e.g., ('num_var1', 'mean'), ('num_var2', 'std'), ...
        # Join the elements of this tuple using an underscore to create new column names
        num_agg.columns = ["_".join(x) for x in num_agg.columns]

        # Reset index to make 'age' a column again
        num_agg.reset_index(drop=False, inplace=True)

        # Merge the new numerical features onto the original dataframe
        X = pd.merge(X, num_agg, on="age", how="left")

        # Impute missing values for the new numerical features, in case there are any
        # If a certain descriptive statistic is not available for a given age (e.g., sample too small), then we impute the missing value with 0
        X.fillna(value=0, inplace=True)
        return X
    else:
        return X


# ------------------ Custom transformer for dropping columns ----------------- #


class ColumnDropperTransformer(BaseEstimator, TransformerMixin):
    """
    Drop columns from a dataframe.
    """

    def __init__(self, columns):
        self.columns = columns

    def transform(self, X, y=None):
        return X.drop(self.columns, axis=1)

    def fit(self, X, y=None):
        return self
