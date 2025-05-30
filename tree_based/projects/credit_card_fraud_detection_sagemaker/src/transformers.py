from functools import partial
from itertools import combinations
from typing import Any, Dict, List, Optional, Self, Tuple, Union

import numpy as np
import polars as pl
from scipy.stats import vonmises
from sklearn.base import BaseEstimator, TransformerMixin, _fit_context
from sklearn.utils.validation import (
    check_is_fitted,
    validate_data,
)

ConfidenceLevels = Union[List[float], Tuple[float, ...], np.ndarray]
GroupByColumns = Union[List[str], Tuple[str, ...], np.ndarray]


class TransactionTimesFeatures(TransformerMixin, BaseEstimator):
    """
    This transformer generates binary features indicating whether the latest transaction
    falls within the confidence interval of transaction times over rolling time windows.
    The times are modeled using the von Mises distribution, which is well-suited for
    periodic data (e.g., hours of the day).

    Features are generated for multiple rolling windows and confidence levels, allowing
    detection of anomalies in transaction timings.

    Reference
    ---------
    https://en.wikipedia.org/wiki/Von_Mises_distribution
    https://doi.org/10.1016/j.eswa.2015.12.030
    """

    _parameter_constraints: Dict[str, Any] = {
        "account_id_column": [str],
        "transaction_datetime_column": [str],
        "confidence_levels": ["array-like"],
    }

    def __init__(
        self,
        account_id_column: str,
        transaction_datetime_column: str,
        confidence_levels: ConfidenceLevels,
    ) -> None:
        """
        Initialize the transformer.

        Parameters
        ----------
        account_id_column : str
            The column name in the input data that contains the account identifier.
        transaction_datetime_column : str
            The column name in the input data that contains the transaction datetime.
        confidence_levels : array-like
            The confidence levels for the von Mises distribution (e.g., [0.95, 0.99]).

        Returns
        -------
        None
        """
        self.account_id_column = account_id_column
        self.transaction_datetime_column = transaction_datetime_column
        self.confidence_levels = confidence_levels

    def __sklearn_is_fitted__(self) -> bool:
        """
        Check if the transformer is fitted.

        Returns
        -------
        bool
            True if the transformer is fitted, False otherwise.
        """
        return hasattr(self, "_ci_funcs")

    def _transaction_within_ci(self, series: List[pl.Series], level: float) -> int:
        """
        Determines whether the last transaction falls within the confidence interval
        of transaction times modeled by the von Mises distribution.

        Parameters
        ----------
        series : List[pl.Series]
            A list containing a single Polars Series of transaction times in radians.
        level : float
            The confidence level for the interval.

        Returns
        -------
        int
            `1` if the last transaction falls within the confidence interval, `0` otherwise.
        """
        times_in_radians: pl.Series = series[0]
        # If there is only one transaction, i.e., the current transaction, return False
        if len(times_in_radians) == 1:
            return 0
        kappa, theta, _ = vonmises.fit(data=times_in_radians)
        # Adjust theta (mean) to be between [0, 2π]
        theta %= 2 * np.pi
        # Density of the most recent transaction
        density = vonmises.pdf(times_in_radians[-1], kappa, loc=theta)
        quantile = vonmises.ppf((1 - level) / 2, kappa, loc=theta)
        cutoff_density = vonmises.pdf(quantile, kappa, loc=theta)
        return int(density >= cutoff_density)

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X: pl.DataFrame, y: Optional[np.ndarray] = None) -> Self:
        """
        Fit the transformer by validating input data and precomputing functions
        for checking confidence intervals.

        Parameters
        ----------
        X : polars.DataFrame
            The input data.
        y : np.ndarray, optional
            Not used; only present for compatibility with the scikit-learn API.

        Returns
        -------
        Self
            The fitted transformer.
        """
        _ = validate_data(
            self,
            X=X,
        )
        # Check that all confidence levels are between 0 and 1
        if not all(0 <= level <= 1 for level in self.confidence_levels):
            raise ValueError("Confidence levels must be between 0 and 1")
        # Construct functions for each confidence level
        self._ci_funcs = {
            int(level * 100): partial(self._transaction_within_ci, level=level)
            for level in self.confidence_levels
        }
        return self

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """
        Transform the input data by generating binary features that indicate
        whether the most recent transaction falls within the confidence interval
        of transaction times over rolling windows.

        Parameters
        ----------
        X : polars.DataFrame
            The input data.

        Returns
        -------
        polars.DataFrame
            The transformed data with added binary features.
        """
        check_is_fitted(self)

        # Convert datetimes to radians once
        X = X.with_columns(
            (
                (
                    X[self.transaction_datetime_column].dt.hour().cast(pl.Float32)
                    + X[self.transaction_datetime_column].dt.minute().cast(pl.Float32)
                    / 60
                    + X[self.transaction_datetime_column].dt.second().cast(pl.Float32)
                    / 3600
                )
                * (2 * np.pi / 24.0)
                % (2 * np.pi)
            ).alias("transactionTimeRadians")
        )

        # Windows in days
        periods = [f"{i}d" for i in [7, 14, 30]]

        # Sort once for all rolling aggregations
        X_sorted = X.sort(
            by=[self.account_id_column, self.transaction_datetime_column],
            descending=[False, False],
        )

        # Cheap operation for mental clarity, this is the driver onto which new features will be joined
        X_transformed = X_sorted.clone()

        for period in periods:
            # Build all the map groups expressions for this period
            map_groups_expr: List[pl.Expr] = []
            for level_pct, func in self._ci_funcs.items():
                feature_name = f"{period}TransactionWithinCI{level_pct}"
                map_groups_expr.append(
                    pl.map_groups(
                        [pl.col("transactionTimeRadians")],
                        func,
                        return_dtype=pl.UInt8,
                        returns_scalar=True,
                    ).alias(feature_name)
                )

            # Perform rolling aggregation for this period
            binary_features_period_levels = X_sorted.rolling(
                index_column=pl.col(self.transaction_datetime_column),
                period=period,
                group_by=[self.account_id_column],
            ).agg(map_groups_expr)

            # Join keys are account_id and transaction_datetime
            join_keys = [self.account_id_column, self.transaction_datetime_column]

            # Join the features back to the driver
            X_transformed = X_transformed.join(
                other=binary_features_period_levels,
                on=join_keys,
                how="inner",
            )

        # Drop the temporary radians column
        X_transformed = X_transformed.drop("transactionTimeRadians")

        return X_transformed


class TransactionAmountsFeatures(TransformerMixin, BaseEstimator):
    """
    This transformer generates rolling aggregation features based
    on transaction amounts, as proposed in:

    Alejandro Correa Bahnsen, Djamila Aouada, Aleksandar Stojanovic, Björn Ottersten.
    "Feature engineering strategies for credit card fraud detection." Expert Systems
    with Applications, 2016.

    Reference
    ---------
    https://doi.org/10.1016/j.eswa.2015.12.030
    """

    _parameter_constraints: Dict[str, Any] = {
        "account_id_column": [str],
        "transaction_datetime_column": [str],
        "transaction_amount_column": [str],
        "group_by_columns": ["array-like"],
    }

    def __init__(
        self,
        account_id_column: str,
        transaction_datetime_column: str,
        transaction_amount_column: str,
        group_by_columns: GroupByColumns,
    ) -> None:
        """
        Initialize the transformer.

        Parameters
        ----------
        account_id_column : str
            The column name in the input data that contains the account identifier.
        transaction_datetime_column : str
            The column name in the input data that contains the transaction datetime.
        transaction_amount_column : str
            The column name in the input data that contains the transaction amount.
        group_by_columns : array-like
            The columns to group by when aggregating the transaction amounts.

        Returns
        -------
        None
        """
        self.account_id_column = account_id_column
        self.transaction_datetime_column = transaction_datetime_column
        self.transaction_amount_column = transaction_amount_column
        self.group_by_columns = group_by_columns

    def __sklearn_is_fitted__(self) -> bool:
        """
        Check if the transformer is fitted.

        Returns
        -------
        bool
            True if the transformer is fitted, False otherwise.
        """
        return (
            hasattr(self, "group_by_columns_pairs_")
            and self.group_by_columns_pairs_ is not None
        )

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X: pl.DataFrame, y: Optional[np.ndarray] = None) -> Self:
        """
        Fit the transformer. Validates the input data, verifies that `group_by_columns`
        contains strings, and generates all pairwise combinations of these columns.

        Parameters
        ----------
        X : polars.DataFrame
            The input data.
        y : np.ndarray, optional
            Not used; only present for compatibility with the scikit-learn API.

        Returns
        -------
        Self
            The fitted transformer.
        """
        _ = validate_data(
            self,
            X=X,
        )
        # Check that the group_by_columns are strings
        if not all(isinstance(col, str) for col in self.group_by_columns):
            raise ValueError(
                "The `group_by_columns` parameter must contain valid column names in the input data X"
            )
        # Generate all pairs of columns from group_by_columns
        self.group_by_columns_pairs_: List[Tuple[str, str]] = list(
            combinations(self.group_by_columns, 2)
        )
        return self

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """
        Transform the input data by adding rolling aggregation features over
        various hourly time windows: `[1h, 3h, 6h, 12h, 18h, 24h, 72h, 168h]`.
        The aggregations include rolling sum, median, and count of transaction
        amounts. Two sets of aggregations are computed: one by account ID only
        and one by account ID plus pairs of other grouping columns.

        Specifically, for each transaction `i`:

        - Define a time window `t_p` (e.g., 24 hours).
        - Identify all past transactions from the same account `x_i^{id}` that
        occurred within `t_p` hours of transaction `i`.
        - Compute:
            - `wTransactionCount`: Number of transactions in the last `t_p` hours.
            - `wTransactionSum`: Sum of transaction amounts in the last `t_p` hours.
            - `wTransactionMedian`: Median transaction amount in the last `t_p` hours.

        Additionally, for each pair of grouping columns `(c1, c2)`, the same set
        of features is computed but only considering transactions that match both
        `c1` and `c2` values.

        Parameters
        ----------
        X : polars.DataFrame
            The input data.

        Returns
        -------
        polars.DataFrame
            The transformed data with the added rolling aggregation features.
        """
        check_is_fitted(self)

        # Windows in hours
        periods = [f"{i}h" for i in [1, 3, 6, 12, 24, 72, 168]]

        # Sort once for all the rolling aggregations
        X_sorted = X.sort(
            by=[self.account_id_column, self.transaction_datetime_column],
            descending=[False, False],
        )

        # Cheap operation for mental clarity, this is the driver onto which new features will be joined
        X_transformed = X_sorted.clone()

        for period in periods:
            # Rolling aggregation by account only
            agg_features_by_account = X_sorted.rolling(
                index_column=pl.col(self.transaction_datetime_column),
                period=period,
                group_by=[self.account_id_column],
            ).agg(
                [
                    pl.sum(self.transaction_amount_column)
                    .cast(pl.Float32)
                    .alias(f"{period}TransactionSum"),
                    pl.median(self.transaction_amount_column)
                    .cast(pl.Float32)
                    .alias(f"{period}TransactionMedian"),
                    pl.len().cast(pl.UInt16).alias(f"{period}TransactionCount"),
                ]
            )

            # Join keys for account-only aggregation
            join_keys_account = [
                self.account_id_column,
                self.transaction_datetime_column,
            ]

            # Join the account-level features
            X_transformed = X_transformed.join(
                other=agg_features_by_account,
                on=join_keys_account,
                how="inner",
            )

            # Rolling aggregation by account and pairs of columns from group_by_columns
            for pair in self.group_by_columns_pairs_:
                pair_name: str = "".join(pair)

                agg_features_by_account_and_pair = X_sorted.rolling(
                    index_column=pl.col(self.transaction_datetime_column),
                    period=period,
                    group_by=[self.account_id_column] + list(pair),
                ).agg(
                    [
                        pl.sum(self.transaction_amount_column)
                        .cast(pl.Float32)
                        .alias(f"{period}{pair_name}TransactionSum"),
                        pl.median(self.transaction_amount_column)
                        .cast(pl.Float32)
                        .alias(f"{period}{pair_name}TransactionMedian"),
                        pl.len()
                        .cast(pl.UInt16)
                        .alias(f"{period}{pair_name}TransactionCount"),
                    ]
                )

                # Join keys include account_id, transaction_datetime, and the pair columns
                join_keys_pair = [
                    self.account_id_column,
                    self.transaction_datetime_column,
                ] + list(pair)

                # Join the account + group-by features
                X_transformed = X_transformed.join(
                    other=agg_features_by_account_and_pair,
                    on=join_keys_pair,
                    how="inner",
                )

        return X_transformed


class DateTimesFeatures(TransformerMixin, BaseEstimator):
    """
    This transformer generates temporal features based on datetime columns.

    It computes raw temporal features (e.g., days since account open, day of the week, etc.),
    cyclic encodings for periodic features (e.g., hour of day, day of week, etc.), and scaled features.

    Parameters
    ----------
    transaction_datetime_column : str
        The column name in the input data that contains the transaction datetime.
    account_open_date_column : str
        The column name in the input data that contains the account open date.
    current_exp_date_column : str
        The column name in the input data that contains the card expiry date.
    last_address_change_date_column : str
        The column name in the input data that contains the date of the last address change.

    Returns
    -------
    Self
        The fitted transformer.
    """

    _parameter_constraints: Dict[str, Any] = {
        "account_id_column": [str],
        "transaction_datetime_column": [str],
        "account_open_date_column": [str],
        "current_exp_date_column": [str],
        "last_address_change_date_column": [str],
    }

    def __init__(
        self,
        transaction_datetime_column: str,
        account_open_date_column: str,
        current_exp_date_column: str,
        last_address_change_date_column: str,
    ) -> None:
        self.transaction_datetime_column = transaction_datetime_column
        self.account_open_date_column = account_open_date_column
        self.current_exp_date_column = current_exp_date_column
        self.last_address_change_date_column = last_address_change_date_column

    def __sklearn_is_fitted__(self) -> bool:
        """
        Check if the transformer is fitted.

        Returns
        -------
        bool
            True if the transformer is fitted, False otherwise.
        """
        return hasattr(self, "_feature_columns")

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X: pl.DataFrame, y: Optional[np.ndarray] = None) -> Self:
        """
        Fit the transformer. Validates the input data.

        Parameters
        ----------
        X : polars.DataFrame
            The input data.
        y : np.ndarray, optional
            Not used; only present for compatibility with the scikit-learn API.

        Returns
        -------
        Self
            The fitted transformer.
        """
        _ = validate_data(self, X=X)
        self._feature_columns = [
            self.transaction_datetime_column,
            self.account_open_date_column,
            self.current_exp_date_column,
            self.last_address_change_date_column,
        ]
        return self

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """
        Transform the input DataFrame by adding temporal features.

        Parameters
        ----------
        X : polars.DataFrame
            The input data.

        Returns
        -------
        polars.DataFrame
            The transformed data with added temporal features.
        """
        check_is_fitted(self)

        # Add duration features
        X = X.with_columns(
            (
                pl.col(self.transaction_datetime_column)
                - pl.col(self.account_open_date_column)
            )
            .dt.total_days()
            .cast(pl.UInt16)
            .alias("daysSinceAccountOpen"),
            (
                pl.col(self.current_exp_date_column)
                - pl.col(self.transaction_datetime_column)
            )
            .dt.total_days()
            .cast(pl.Int16)  # Negative values may be possible
            .alias("daysUntilCardExpiry"),
            (
                pl.col(self.transaction_datetime_column)
                - pl.col(self.last_address_change_date_column)
            )
            .dt.total_days()
            .cast(pl.UInt16)
            .alias("daysSinceLastAddressChange"),
        )

        # Add cyclic and scaled temporal features
        X = X.with_columns(
            # Month of the year (1-12)
            pl.col(self.transaction_datetime_column)
            .dt.month()
            .alias("transactionMonthOfYearRaw"),
            # Day of the month (1-31)
            pl.col(self.transaction_datetime_column)
            .dt.day()
            .alias("transactionDayOfMonthRaw"),
            # Day of the week (1=Monday, 7=Sunday)
            pl.col(self.transaction_datetime_column)
            .dt.weekday()
            .alias("transactionDayOfWeekRaw"),
            # Hour of day (0-23)
            pl.col(self.transaction_datetime_column)
            .dt.hour()
            .alias("transactionHourOfDayRaw"),
        )

        X = X.with_columns(
            # Cyclic encodings
            (2 * np.pi * pl.col("transactionMonthOfYearRaw") / 12)
            .sin()
            .cast(pl.Float32)
            .alias("transactionMonthOfYearSin"),
            (2 * np.pi * pl.col("transactionMonthOfYearRaw") / 12)
            .cos()
            .cast(pl.Float32)
            .alias("transactionMonthOfYearCos"),
            (2 * np.pi * pl.col("transactionDayOfWeekRaw") / 7)
            .sin()
            .cast(pl.Float32)
            .alias("transactionDayOfWeekSin"),
            (2 * np.pi * pl.col("transactionDayOfWeekRaw") / 7)
            .cos()
            .cast(pl.Float32)
            .alias("transactionDayOfWeekCos"),
            (2 * np.pi * pl.col("transactionHourOfDayRaw") / 24)
            .sin()
            .cast(pl.Float32)
            .alias("transactionHourOfDaySin"),
            (2 * np.pi * pl.col("transactionHourOfDayRaw") / 24)
            .cos()
            .cast(pl.Float32)
            .alias("transactionHourOfDayCos"),
            # Scaled features
            ((pl.col("transactionMonthOfYearRaw") - 1) / 11)
            .cast(pl.Float32)
            .alias("transactionMonthOfYearScaled"),
            ((pl.col("transactionDayOfMonthRaw") - 1) / 30)
            .cast(pl.Float32)
            .alias("transactionDayOfMonthScaled"),
            ((pl.col("transactionDayOfWeekRaw") - 1) / 6)
            .cast(pl.Float32)
            .alias("transactionDayOfWeekScaled"),
            (pl.col("transactionHourOfDayRaw") / 23)
            .cast(pl.Float32)
            .alias("transactionHourOfDayScaled"),
        )

        # Drop original date and raw features
        X = X.drop(
            self._feature_columns + [col for col in X.columns if col.endswith("Raw")]
        )

        return X


def check_estimators(estimators: List[BaseEstimator]) -> None:
    """
    Run scikit-learn's `check_estimator` for a list of estimators.

    Parameters
    ----------
    estimators : list of BaseEstimator
        List of estimators to check.

    Returns
    -------
    None
    """
    for estimator in estimators:
        try:
            print(f"Checking {estimator.__class__.__name__}...\n")

            results = check_estimator(
                estimator=estimator,
                expected_failed_checks={
                    key: "Transformer only works on polars DataFrames; these tests use numpy array inputs"
                    for key in [
                        "check_fit2d_1sample",
                        "check_fit2d_1feature",
                        "check_fit1d",
                        "check_fit2d_predict1d",
                        "check_fit_score_takes_y",
                        "check_n_features_in_after_fitting",
                        "check_dtype_object",
                        "check_pipeline_consistency",
                        "check_estimators_nan_inf",
                        "check_estimators_pickle",
                        "check_f_contiguous_array_estimator",
                        "check_transformer_data_not_an_array",
                        "check_transformer_general",
                        "check_transformer_preserve_dtypes",
                        "check_estimators_dtypes",
                        "check_dtype_object",
                        "check_methods_subset_invariance",
                        "check_dict_unchanged",
                        "check_fit_idempotent",
                        "check_methods_sample_order_invariance",
                    ]
                },
            )

            for result in results:
                if result["status"] == "passed":
                    print(f"{result['check_name']}: {result['status']}")
                elif result["status"] == "xfail":
                    print(
                        f"{result['check_name']}: {result['expected_to_fail_reason']}"
                    )

            print("\n")
            print("-" * 80)
            print("\n")

        except Exception as error:
            print(f"Error with {estimator.__class__.__name__}: {error}")

    return None


if __name__ == "__main__":
    from sklearn.utils.estimator_checks import check_estimator

    account_id_column = "accountNumber"
    transaction_datetime_column = "transactionDateTime"
    confidence_levels = [0.90, 0.95]
    transaction_times_features = TransactionTimesFeatures(
        account_id_column=account_id_column,
        transaction_datetime_column=transaction_datetime_column,
        confidence_levels=confidence_levels,
    )

    transaction_amount_column = "transactionAmount"
    group_by_columns = ["posEntryMode", "transactionType"]
    transaction_amounts_features = TransactionAmountsFeatures(
        account_id_column=account_id_column,
        transaction_datetime_column=transaction_datetime_column,
        transaction_amount_column=transaction_amount_column,
        group_by_columns=group_by_columns,
    )

    account_open_date_column = "accountOpenDate"
    current_exp_date_column = "cardExpiryDate"
    last_address_change_date_column = "lastAddressChangeDate"
    datetime_features = DateTimesFeatures(
        transaction_datetime_column=transaction_datetime_column,
        account_open_date_column=account_open_date_column,
        current_exp_date_column=current_exp_date_column,
        last_address_change_date_column=last_address_change_date_column,
    )

    check_estimators(
        [transaction_times_features, transaction_amounts_features, datetime_features]
    )
