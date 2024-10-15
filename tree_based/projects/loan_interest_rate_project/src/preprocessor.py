import pandas as pd
import numpy as np
import cudf
import cupy as cp
from re import sub


def extract_date_features(data: pd.DataFrame) -> cudf.DataFrame:
    """
    Extract year and months from date columns. Upstream imputers should have already handled missing values in the date columns,
    if any, by encoding them as 'missing'. This function will coerce 'missing' to `pandas._libs.tslibs.nattype.NaTType`, which will
    subsequently become -1 when accessing the year and month components with the dt accessor.

    Parameters
    ----------
    data : pd.DataFrame

    Returns
    -------
    cudf.DataFrame
    """
    date_cols = ["loan_issued_date", "borrower_earliest_credit_open_date"]
    year_cols = [sub("_date", "", col) + "_year" for col in date_cols]
    month_cols = [sub("_date", "", col) + "_month" for col in date_cols]
    # Year extraction
    data[year_cols] = data[date_cols].apply(
        lambda col: pd.to_datetime(col, errors="coerce").dt.to_period("M").dt.year
    )
    data[month_cols] = data[date_cols].apply(
        lambda col: pd.to_datetime(col, errors="coerce").dt.to_period("M").dt.month
    )

    return cudf.DataFrame(data.drop(date_cols, axis=1))


def num_feat_eng(data: cudf.DataFrame) -> cudf.DataFrame:
    """
    Group by aggregation of numerical features based on categorical features.

    Parameters
    ----------
    data : cudf.DataFrame

    Returns
    -------
    cudf.DataFrame
    """
    num_cols = [
        "loan_amt_requested",
        "loan_amt_investor_funded_portion",
        "borrower_annual_income",
        "monthly_debt_to_income_ratio",
        "num_of_past_dues",
        "num_of_creditor_inquiries",
        "num_of_open_credit_line",
        "num_of_derog_publib_rec",
        "total_credit_rev_balance",
        "rev_line_util_rate",
        "total_credit_line",
    ]

    group_cols = [
        "num_of_payment_months",
        "loan_subgrade",
        "num_of_years_employed",
        "home_ownership_status",
        "verify_income_or_source",
        "loan_issued_year",
        "borrower_provided_loan_category",
        "zip_first_three",
        "borrower_state",
        "borrower_earliest_credit_open_year",
        "init_loan_status",
    ]

    for group in group_cols:
        for agg_func in ["std", "mean", "max", "sum"]:
            data[[col + f"_{agg_func}_by_{group}" for col in num_cols]] = data.groupby(
                group
            )[num_cols].transform(agg_func)
    return data.to_pandas()


def restore_columns(data: pd.DataFrame) -> pd.DataFrame:
    """
    Restore the original column order of the data frame.

    Parameters
    ----------
    data : cudf.DataFrame

    Returns
    -------
    cudf.DataFrame
    """
    data.columns = [sub(r"^(num__|cat__)", "", col) for col in data.columns]
    return data
