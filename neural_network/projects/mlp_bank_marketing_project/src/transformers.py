from re import sub

import pandas as pd

# ---------------------------- Numerical features ---------------------------- #


def NumTransformer(df: pd.DataFrame) -> pd.DataFrame:
    """
    Group by aggregation of numerical columns based on characteristics--- age, job, education, contact, marital, housing, day_of_week, month.
    """
    # Define list of numerical columns to aggregate
    num_col = [
        "campaign",
        "pdays",
        "previous",
        "emp_var_rate",
        "cons_price_idx",
        "cons_conf_idx",
        "euribor3m",
        "nr_employed",
    ]

    for group in [
        "age",
        "job",
        "marital",
        "contact",
        "education",
        "housing",
        "day_of_week",
        "month",
        "poutcome",
    ]:
        for agg_func in ["mean", "median", "min", "max", "last"]:
            df[[col + f"_{agg_func}_by_{group}" for col in num_col]] = df.groupby(
                group
            )[num_col].transform(agg_func, engine="cython")

    return df


# ------------------ Convert columns to categorical datatype ----------------- #


def CatTransformer(df: pd.DataFrame) -> pd.DataFrame:
    """
    Function to convert columns to categorical datatype.
    """
    # Convert categorical features to 'category' datatype
    cat_col = [
        "age",
        "job",
        "marital",
        "education",
        "default",
        "housing",
        "loan",
        "contact",
        "month",
        "day_of_week",
        "poutcome",
    ]

    for col in cat_col:
        df[col] = df[col].astype("category")

    return df


# --------------------------- Restore column names --------------------------- #


def RestoreColNames(df: pd.DataFrame) -> pd.DataFrame:
    """
    Function to restore column names.
    """
    # Restore column names
    df.columns = [sub(r"scaler__|remainder__", "", col) for col in df.columns]

    return df
