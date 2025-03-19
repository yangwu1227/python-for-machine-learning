import re
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import polars.selectors as cs
from mapie.metrics import regression_coverage_score, regression_mean_width_score
from mapie.regression import MapieQuantileRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm import tqdm


def clean_col(name: str) -> str:
    """
    Clean a column name by lowercasing and replacing whitespace and non-alphanumeric characters with underscores.

    Parameters
    ----------
    name : str
        The original column name.

    Returns
    -------
    str
        The cleaned column name.
    """
    name = name.lower()
    name = re.sub(r"\s+", "_", name)
    name = re.sub(r"[^a-z0-9_]", "_", name)
    name = re.sub(r"_+", "_", name)
    return name.strip("_")


def load_and_preprocess_data(filename: str) -> pl.DataFrame:
    """
    Load and preprocess rent data from a CSV file.

    The preprocessing steps include:
      - Lowercasing all string columns.
      - Splitting the 'amenities' and 'pets_allowed' columns by comma.
      - Filling null values in these columns with empty lists.
      - Binarizing multi-label columns using MultiLabelBinarizer.
      - Dropping the original multi-label columns and concatenating the binarized data.
      - Renaming columns using the clean_col function.

    Parameters
    ----------
    filename : str
        The path to the CSV file.

    Returns
    -------
    pl.DataFrame
        The preprocessed DataFrame.
    """
    data = pl.read_parquet(filename)

    # Lowercase all string columns
    data = data.with_columns(cs.by_dtype(pl.String).str.to_lowercase())

    # Split multi-label columns
    data = data.with_columns(
        pl.col("amenities").str.split(","),
        pl.col("pets_allowed").str.split(","),
    )

    # Fill null values with empty lists
    data = data.with_columns(
        [
            pl.col("amenities").fill_null(pl.lit([])).alias("amenities"),
            pl.col("pets_allowed").fill_null(pl.lit([])).alias("pets_allowed"),
        ]
    )

    # Binarize multi-label columns for amenities
    mlb_amenities = MultiLabelBinarizer()
    amenities = mlb_amenities.fit_transform(data["amenities"])
    amenities_data = pl.DataFrame(amenities, schema=mlb_amenities.classes_.tolist())

    # Binarize multi-label columns for pets_allowed
    mlb_pets = MultiLabelBinarizer()
    pets = mlb_pets.fit_transform(data["pets_allowed"])
    pets_data = pl.DataFrame(pets, schema=mlb_pets.classes_.tolist())

    # Drop original columns and concatenate the new binary columns
    data = data.drop(["amenities", "pets_allowed"])
    data = pl.concat([data, amenities_data, pets_data], how="horizontal")

    # Rename columns using the clean_col function
    data = data.rename({col: clean_col(col) for col in data.columns})
    return data


def split_data(
    X: pl.DataFrame, y: pl.Series, random_state: np.random.RandomState
) -> Tuple[
    pl.DataFrame,
    pl.DataFrame,
    pl.DataFrame,
    pl.DataFrame,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Split the data into training, validation, calibration, and test sets.

    Parameters
    ----------
    X : pl.DataFrame
        The feature DataFrame.
    y : pl.Series
        The target Series.
    random_state : np.random.RandomState
        The random state for reproducibility.

    Returns
    -------
    Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame,
          np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        A tuple containing (X_train, X_val, X_calib, X_test, y_train, y_val, y_calib, y_test).
    """
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.5, random_state=random_state
    )
    X_val, X_temp, y_val, y_temp = train_test_split(
        X_temp, y_temp, test_size=0.7, random_state=random_state
    )
    X_calib, X_test, y_calib, y_test = train_test_split(
        X_temp, y_temp, test_size=0.2, random_state=random_state
    )
    return X_train, X_val, X_calib, X_test, y_train, y_val, y_calib, y_test


def train_models(
    X_train: pl.DataFrame,
    y_train: np.ndarray,
    random_state: np.random.RandomState,
    alphas: List[float],
) -> List[GradientBoostingRegressor]:
    """
    Train GradientBoostingRegressor models for quantile regression with different alpha values.

    Parameters
    ----------
    X_train : pl.DataFrame
        The training feature DataFrame.
    y_train : np.ndarray
        The training target array.
    random_state : np.random.RandomState
        The random state for reproducibility.
    alphas : List[float]
        List of alpha values to use for the quantile regressors.

    Returns
    -------
    List[GradientBoostingRegressor]
        A list of trained GradientBoostingRegressor models.
    """
    models: List[GradientBoostingRegressor] = []
    for alpha in tqdm(alphas, desc="Training models"):
        model = GradientBoostingRegressor(
            loss="quantile", alpha=alpha, random_state=random_state
        )
        model_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("model", model),
            ]
        )
        model_pipeline.fit(X_train, y_train)
        models.append(model_pipeline)
    return models


def plot_interval_width_histogram(widths: np.ndarray) -> None:
    """
    Plot a histogram of the prediction interval widths.

    Parameters
    ----------
    widths : np.ndarray
        Array of interval widths.
    """
    plt.hist(widths)
    plt.xlabel("Interval width")
    plt.ylabel("Frequency")
    plt.show()


def plot_quantile_by_feature(
    X: pl.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_qrs: np.ndarray,
    feature_name: str,
) -> None:
    """
    Plot predicted quantiles and true values against a specific feature.

    This function creates a scatter plot of the feature values vs. the predicted and true target values,
    and overlays the lower and upper quantile estimates with dashed lines along with a shaded region between them.

    Parameters
    ----------
    X : pl.DataFrame
        DataFrame containing the features.
    y_true : np.ndarray
        True target values.
    y_pred : np.ndarray
        Predicted target values.
    y_qrs : np.ndarray
        Array of predicted quantile ranges with shape (n_samples, 2) where the columns are the lower and upper quantiles.
    feature_name : str
        The feature name to plot on the x-axis.
    """
    feature: np.ndarray = X[feature_name].to_numpy()
    order = np.argsort(feature)
    plt.scatter(feature, y_pred, alpha=1 / 3)
    plt.scatter(feature, y_true, color="black", alpha=1 / 3)
    plt.plot(feature[order], y_qrs[order][:, 0], color="orange", ls="--")
    plt.plot(feature[order], y_qrs[order][:, 1], color="orange", ls="--")
    plt.fill_between(
        feature[order].ravel(),
        y_qrs[order][:, 0].ravel(),
        y_qrs[order][:, 1].ravel(),
        alpha=0.2,
    )
    plt.xlabel(feature_name)
    plt.ylabel("Predicted Rent Per Square Meter")
    plt.show()


def main() -> int:
    """
    Main function to execute the rent prediction and quantile estimation pipeline.

    Returns
    -------
    int
        Exit status code.
    """
    random_state = np.random.RandomState(1227)

    # Load and preprocess the data
    data = load_and_preprocess_data(
        "/Users/yang_wu/Desktop/python/python_for_machine_learning/data/regression/rent_data.parquet"
    )
    data = data.drop_nulls(subset=["monthly_price", "square_feet"])

    # Extract target variable and features
    y = data["monthly_price"] / data["square_feet"]
    X = data.drop(
        [
            "monthly_price",
            "square_feet",
            "address",
            "state",
            "cityname",
            "latitude",
            "longitude",
            "id",
            "has_photo",
        ]
    )

    # Split the data into training, validation, calibration, and test sets
    X_train, X_val, X_calib, X_test, y_train, y_val, y_calib, y_test = split_data(
        X, y, random_state
    )
    print(f"X_train: {X_train.shape}")
    print(f"X_val: {X_val.shape}")
    print(f"X_calib: {X_calib.shape}")
    print(f"X_test: {X_test.shape}")

    # Train models for quantile regression
    alphas = [1 / 6, 5 / 6, 0.5]
    models = train_models(X_train, y_train, random_state, alphas)

    # Initialize and fit the MapieQuantileRegressor with prefit models
    cqr = MapieQuantileRegressor(models, alpha=1 / 3, cv="prefit")
    cqr.fit(X_calib, y_calib)

    # Predict quantiles on the test set
    y_pred, y_qr = cqr.predict(X_test)

    # Plot histogram of the interval widths
    widths = y_qr[:, 1] - y_qr[:, 0]
    plot_interval_width_histogram(widths)

    # Compute and display average interval width and coverage score
    avg_width = regression_mean_width_score(y_qr[:, 0], y_qr[:, 1])
    print(f"Average interval width: {avg_width:.2f}")
    cov = regression_coverage_score(y_test, y_qr[:, 0], y_qr[:, 1])
    print(f"Coverage: {cov:.2f}")

    # Plot quantile predictions against the feature "area"
    plot_quantile_by_feature(X_test, y_test, y_pred, y_qr, "bedrooms")

    return 0


if __name__ == "__main__":
    main()
