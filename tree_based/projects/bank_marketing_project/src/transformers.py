import pandas as pd
from feature_engine.encoding import CountFrequencyEncoder, OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Union, List, Self

# --------------------------- Categorical features --------------------------- #


def cat_feature_engineer(X: pd.DataFrame, encode_type: str = "count") -> pd.DataFrame:
    """
    Engineer categorical features using specified encoding methods.

    Parameters
    ----------
    X : pd.DataFrame
        The input DataFrame with categorical features.
    encode_type : str, optional
        Type of encoding to apply to categorical features. Options are "count" for count encoding
        and "onehot" for one-hot encoding. Default is "count".

    Returns
    -------
    pd.DataFrame
        Transformed DataFrame with categorical features encoded.

    Raises
    ------
    ValueError
        If `encode_type` is not "count" or "onehot".
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


def num_feature_engineer(X: pd.DataFrame, switch: bool = True) -> pd.DataFrame:
    """
    Engineer numerical features based on aggregation grouped by a key column.

    Parameters
    ----------
    X : pd.DataFrame
        The input DataFrame with numerical features.
    switch : bool, optional
        If True, applies feature engineering transformations. Default is True.

    Returns
    -------
    pd.DataFrame
        Transformed DataFrame with new numerical features based on aggregation statistics.
    """
    if switch:
        # Numerical features
        num_features = X.select_dtypes("number").columns.tolist()
        num_features.remove("age")

        # Group by client 'age' and generate new columns ['mean', 'std', 'min', 'max', 'last'] for each col in 'num_features'
        num_agg = X.groupby("age")[num_features].agg(
            ["mean", "std", "min", "max", "last"]
        )
        # Creating column names from multi-index columns
        num_agg.columns = ["_".join(x) for x in num_agg.columns]

        # Reset index to make 'age' a column again
        num_agg.reset_index(drop=False, inplace=True)

        # Merge the new numerical features onto the original dataframe
        X = pd.merge(X, num_agg, on="age", how="left")

        # Impute missing values for the new numerical features
        X.fillna(value=0, inplace=True)
        return X
    else:
        return X


# ------------------ Custom transformer for dropping columns ----------------- #


class ColumnDropperTransformer(BaseEstimator, TransformerMixin):
    """
    Custom transformer to drop specified columns from a DataFrame.

    Parameters
    ----------
    columns : Union[List[str], str]
        List of column names or a single column name to be dropped from the DataFrame.

    Methods
    -------
    transform(X)
        Drop specified columns from the DataFrame.
    fit(X, y=None)
        Fit method (no-op).
    """

    def __init__(self, columns: Union[List[str], str]):
        self.columns = columns

    def transform(self, X: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:
        """
        Transform method to drop specified columns from the DataFrame.

        Parameters
        ----------
        X : pd.DataFrame
            Input DataFrame.
        y : pd.Series, optional
            Target values (ignored).

        Returns
        -------
        pd.DataFrame
            DataFrame with specified columns dropped.
        """
        return X.drop(self.columns, axis=1)

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> Self:
        """
        Fit method (no-op).

        Parameters
        ----------
        X : pd.DataFrame
            Input DataFrame.
        y : pd.Series, optional
            Target values (ignored).

        Returns
        -------
        ColumnDropperTransformer
            Instance of itself.
        """
        return self
