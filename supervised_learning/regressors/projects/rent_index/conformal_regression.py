import re
from pathlib import Path
from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import polars.selectors as cs
from mapie.regression import ConformalizedQuantileRegressor
from scipy.stats import loguniform, randint, uniform
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import make_scorer, mean_pinball_loss
from sklearn.model_selection import RandomizedSearchCV, train_test_split
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


def load_and_preprocess_data(filename: Union[str, Path]) -> pl.DataFrame:
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
    filename : Union[str, Path]
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
    Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, np.ndarray, np.ndarray, np.ndarray]
        A tuple containing (X_train, X_calib, X_test, y_train, y_calib, y_test).
    """
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=random_state
    )
    X_calib, X_test, y_calib, y_test = train_test_split(
        X_temp, y_temp, test_size=0.2, random_state=random_state
    )
    return (
        X_train,
        X_calib,
        X_test,
        y_train.to_numpy(),
        y_calib.to_numpy(),
        y_test.to_numpy(),
    )


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
    search_space = {
        "model__learning_rate": loguniform(1e-3, 0.1),
        "model__subsample": loguniform(0.7, 1),
        "model__min_samples_split": uniform(0.1, 0.4),
        "model__min_samples_leaf": uniform(0.1, 0.4),
        "model__max_depth": randint(6, 15),
        "model__max_features": loguniform(0.6, 0.9),
    }

    models: List[GradientBoostingRegressor] = []
    for alpha in tqdm(alphas, desc="Training models"):
        # By default, scores are maximized, so set greater_is_better=False to minimize the loss
        score_func = make_scorer(
            score_func=mean_pinball_loss,
            greater_is_better=False,
            alpha=alpha,
        )
        model = GradientBoostingRegressor(
            n_estimators=1500,
            loss="quantile",
            alpha=alpha,
            random_state=random_state,
            validation_fraction=0.2,
            n_iter_no_change=300,
        )
        model_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("model", model),
            ]
        )
        randomized_search = RandomizedSearchCV(
            estimator=model_pipeline,
            param_distributions=search_space,
            n_iter=50,
            cv=5,
            n_jobs=-1,
            verbose=2,
            refit=True,
            scoring=score_func,
            random_state=random_state,
        )
        randomized_search.fit(X=X_train, y=y_train)
        models.append(randomized_search.best_estimator_)
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


def _coverage_score(y_true: np.ndarray, y_intervals_2d: np.ndarray) -> float:
    """
    Compute the empirical coverage of prediction intervals.

    Parameters
    ----------
    y_true : np.ndarray
        True target values of shape (n,).
    y_intervals_2d : np.ndarray
        Prediction intervals of shape (n, 2), where the first column is the lower bound
        and the second column is the upper bound.

    Returns
    -------
    float
    """
    lower = y_intervals_2d[:, 0]
    upper = y_intervals_2d[:, 1]
    return float(np.mean((y_true >= lower) & (y_true <= upper)))


def _mean_width_score(y_intervals_2d: np.ndarray) -> float:
    """
    Compute the mean width of prediction intervals.

    Parameters
    ----------
    y_intervals_2d : np.ndarray
        Prediction intervals of shape (n, 2), where the first column is the lower bound
        and the second column is the upper bound.

    Returns
    -------
    float
    """
    return float(np.mean(y_intervals_2d[:, 1] - y_intervals_2d[:, 0]))


def _uniform_subsample_indices(
    n: int,
    sample: Optional[Union[int, float]],
) -> np.ndarray:
    """
    Return evenly spaced indices of length k from [0, n - 1].

    Parameters
    ----------
    n : int
        Total number of points.
    sample : Optional[Union[int, float]]
        If None or invalid, returns all indices.
        If 0 < sample < 1, uses ceil(sample * n) points.
        If sample >= 1, uses min(int(sample), n) points.

    Returns
    -------
    np.ndarray
        Indices of shape (k,).
    """
    if sample is None or n <= 0:
        return np.arange(n)

    if isinstance(sample, float):
        if not (0.0 < sample < 1.0):
            return np.arange(n)
        k = int(np.ceil(sample * n))
    else:
        k = int(sample)
        if k <= 0 or k >= n:
            return np.arange(n)

    # Even spacing preserves coverage across the x range better than random picks
    return np.linspace(0, n - 1, num=k, dtype=int)


def plot_prediction_intervals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_intervals: np.ndarray,
    dataset_name: str,
    confidence_level: float = 2 / 3,
    legend_outside: bool = True,
    true_sample: Optional[Union[int, float]] = 0.25,
    true_alpha: float = 0.20,
    true_size: float = 10.0,
    ribbon_alpha: float = 0.45,
) -> None:
    """
    Plot prediction intervals with lighter, subsampled true points and a more pronounced ribbon.

    Parameters
    ----------
    y_true : np.ndarray
        True target values, shape (n,).
    y_pred : np.ndarray
        Predicted point estimates, shape (n,).
    y_intervals : np.ndarray
        Prediction intervals, shape (n, 2), (n, 2, 1), or (n, 2, k).
    dataset_name : str
        Label for the title.
    confidence_level : float, default 2/3
        Displayed level in the legend.
    legend_outside : bool, default True
        Place legend outside the axes on the right.
    true_sample : Optional[Union[int, float]], default 0.25
        Subsample of true points for plotting only.
        If float in (0,1), interpreted as fraction of n.
        If int >= 1, caps the number of points.
        If None, plots all true points.
    true_alpha : float, default 0.20
        Alpha for the true scatter points to make them lighter.
    true_size : float, default 10.0
        Marker size for true points.
    ribbon_alpha : float, default 0.45
        Alpha for the interval ribbon to make it more pronounced.

    Returns
    -------
    None
    """
    # Prepare arrays
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    y_int2d = np.asarray(y_intervals).squeeze()

    # Metrics on unsorted data
    coverage = _coverage_score(y_true, y_int2d)
    avg_width = _mean_width_score(y_int2d)

    # Sort by prediction for visualization only
    order = np.argsort(y_pred)
    y_true_s = y_true[order]
    y_pred_s = y_pred[order]
    y_int_s = y_int2d[order]
    x = np.arange(len(y_pred_s))

    # Subsample true points to avoid covering the ribbon
    idx_true = _uniform_subsample_indices(n=len(y_true_s), sample=true_sample)

    # Plot
    plt.figure(figsize=(12, 7))

    # Ribbon first so it sits behind points
    plt.fill_between(
        x,
        y_int_s[:, 0],
        y_int_s[:, 1],
        alpha=ribbon_alpha,
        label=f"{confidence_level * 100:.0f}% interval",
        zorder=1,
    )

    plt.plot(x, y_pred_s, linewidth=1.8, label="Predicted", zorder=3)

    # Lighter, subsampled true points
    plt.scatter(
        x[idx_true],
        y_true_s[idx_true],
        s=true_size,
        alpha=true_alpha,
        label="True (subsampled)",
        zorder=4,
    )

    plt.xlabel("Sample index (sorted by prediction)")
    plt.ylabel("Target")
    plt.title(
        f"{dataset_name} Prediction Intervals\n"
        f"Coverage: {coverage:.3f} | Avg width: {avg_width:.3f}"
    )

    if legend_outside:
        plt.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0.0)
        plt.tight_layout(rect=(0, 0, 0.82, 1))
    else:
        plt.legend()
        plt.tight_layout()

    plt.grid(True, alpha=0.3)
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
    project_root = Path(__file__).parents[4]

    # Load and preprocess the data
    data = load_and_preprocess_data(project_root / "data/regression/rent_data.parquet")
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

    # Split the data into training, calibration, and test sets
    X_train, X_calib, X_test, y_train, y_calib, y_test = split_data(X, y, random_state)
    print(f"X_train: {X_train.shape}")
    print(f"X_calib: {X_calib.shape}")
    print(f"X_test: {X_test.shape}")

    # Train models for quantile regression
    alphas = [1 / 6, 5 / 6, 0.5]
    models = train_models(X_train, y_train, random_state, alphas)

    # Initialize and fit the ConformalizedQuantileRegressor with prefit models
    conf_level = 2 / 3
    cqr = ConformalizedQuantileRegressor(
        estimator=models, confidence_level=conf_level, prefit=True
    )
    cqr.conformalize(X_conformalize=X_calib, y_conformalize=y_calib)

    # Predict on all datasets
    y_pred_train, y_pi_train = cqr.predict_interval(X=X_train)
    y_pred_calib, y_pi_calib = cqr.predict_interval(X=X_calib)
    y_pred_test, y_pi_test = cqr.predict_interval(X=X_test)
    true_sample = 0.01  # Fraction of true points to plot
    true_alpha = 0.40  # Alpha for true points
    true_size = 50.0  # Size for true points
    ribbon_alpha = 0.50  # Alpha for the interval ribbon

    plot_prediction_intervals(
        y_true=y_train,
        y_pred=y_pred_train,
        y_intervals=y_pi_train,
        dataset_name="Training",
        confidence_level=conf_level,
        true_sample=true_sample,
        true_alpha=true_alpha,
        true_size=true_size,
        ribbon_alpha=ribbon_alpha,
    )

    plot_prediction_intervals(
        y_true=y_calib,
        y_pred=y_pred_calib,
        y_intervals=y_pi_calib,
        dataset_name="Calibration",
        confidence_level=conf_level,
        true_sample=true_sample,
        true_alpha=true_alpha,
        true_size=true_size,
        ribbon_alpha=ribbon_alpha,
    )

    plot_prediction_intervals(
        y_true=y_test,
        y_pred=y_pred_test,
        y_intervals=y_pi_test,
        dataset_name="Test",
        confidence_level=conf_level,
        true_sample=true_sample,
        true_alpha=true_alpha,
        true_size=true_size,
        ribbon_alpha=ribbon_alpha,
    )

    # Plot histogram of the interval widths
    widths = y_pi_test[:, 1] - y_pi_test[:, 0]
    plot_interval_width_histogram(widths)

    # Plot quantile predictions against the feature "area"
    features_to_plot = ["bathrooms", "bedrooms"]
    for feature in features_to_plot:
        plot_quantile_by_feature(X_test, y_test, y_pred_test, y_pi_test, feature)

    return 0


if __name__ == "__main__":
    main()
