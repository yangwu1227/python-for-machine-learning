from typing import Any, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import optuna
import polars as pl
from IPython.display import Image
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
    precision_recall_curve,
)
from sklearn.utils import Bunch


def plot_prediction_vs_groundtruth(
    y_prob: np.ndarray,
    threshold: float,
    y_true: np.ndarray,
    figsize: Tuple[float, float] = (14, 8),
) -> None:
    """
    Plot the distribution of predicted labels and ground truth labels side-by-side.

    Parameters
    ----------
    y_prob : np.ndarray
        Predicted probabilities for the positive class.
    threshold : float
        Threshold for converting probabilities to binary predictions.
    y_true : np.ndarray
        True binary labels.
    figsize : Tuple[float, float], default=(14, 8)
        Figure size for the plot.

    Returns
    -------
    None
        Displays side-by-side histogram plots for predicted and ground truth labels.
    """
    # Convert probabilities to binary predictions using the threshold
    y_pred = (y_prob >= threshold).astype(int)

    predicted_labels = y_pred
    ground_truth_labels = y_true

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    axes[0].hist(
        predicted_labels,
        bins=np.histogram_bin_edges(predicted_labels, bins="auto"),
        color="cornflowerblue",
        alpha=0.7,
        edgecolor="black",
        rwidth=0.9,
    )
    axes[0].set_title("Distribution of Predicted Labels")
    axes[0].set_xticks([0, 1])
    axes[0].set_xlabel("Predicted Label")
    axes[0].set_ylabel("Frequency")

    axes[1].hist(
        ground_truth_labels,
        bins=np.histogram_bin_edges(ground_truth_labels, bins="auto"),
        color="coral",
        alpha=0.7,
        edgecolor="black",
        rwidth=0.9,
    )
    axes[1].set_title("Distribution of Ground Truth Labels")
    axes[1].set_xticks([0, 1])
    axes[1].set_xlabel("Ground Truth Label")
    axes[1].set_ylabel("Frequency")

    plt.tight_layout()
    plt.show()


def plot_error_numerical_distribution(
    data: pl.DataFrame,
    numerical_feature: str,
    bins: int = 20,
    figsize: Tuple[float, float] = (14, 6),
) -> None:
    """
    Plot the distribution of a numerical feature for false negatives and false positives side-by-side.

    Parameters
    ----------
    data : pl.DataFrame
        The Polars DataFrame containing the data, must have "falseNegative" and "falsePositive" columns.
    numerical_feature : str
        The name of the numerical feature to analyze.
    bins : int, default=20
        Number of bins for the histogram.
    figsize : Tuple[float, float], default=(14, 6)
        Figure size for the plot.

    Returns
    -------
    None
        Displays side-by-side histogram plots for the numerical feature.
    """
    false_negative_data = data.filter(pl.col("falseNegative") == 1)
    false_positive_data = data.filter(pl.col("falsePositive") == 1)

    false_negative_values = false_negative_data[numerical_feature].to_numpy()
    false_positive_values = false_positive_data[numerical_feature].to_numpy()

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    axes[0].hist(
        false_negative_values,
        bins=bins,
        color="salmon",
        alpha=0.7,
        edgecolor="black",
    )
    axes[0].set_title(f"Distribution of '{numerical_feature}' (False Negatives)")
    axes[0].set_xlabel(numerical_feature)
    axes[0].set_ylabel("Frequency")

    axes[1].hist(
        false_positive_values,
        bins=bins,
        color="skyblue",
        alpha=0.7,
        edgecolor="black",
    )
    axes[1].set_title(f"Distribution of '{numerical_feature}' (False Positives)")
    axes[1].set_xlabel(numerical_feature)
    axes[1].set_ylabel("Frequency")

    plt.tight_layout()
    plt.show()


def plot_error_category_distribution(
    data: pl.DataFrame,
    categorical_feature: str,
    top_n: int = 10,
    figsize: Tuple[float, float] = (14, 6),
) -> None:
    """
    Plot the most frequent categories for false negatives and false positives
    side-by-side for a given categorical feature.

    Parameters
    ----------
    data : pl.DataFrame
        The Polars DataFrame containing the data, must have "falseNegative" and "falsePositive" columns.
    categorical_feature : str
        The name of the categorical feature to analyze.
    top_n : int, default=10
        Number of top categories to display in the plot.
    figsize : Tuple[float, float], default=(14, 6)
        Figure size for the plot.

    Returns
    -------
    None
        Displays side-by-side bar plots of the most frequent categories.
    """
    data = data.with_columns(pl.col(categorical_feature).fill_null("Unknown"))

    false_negative_data = data.filter(pl.col("falseNegative") == 1)
    false_negative_counts = (
        false_negative_data.group_by(categorical_feature)
        .agg(pl.len().alias("count"))
        .sort("count", descending=True)
        .head(top_n)
    )

    false_positive_data = data.filter(pl.col("falsePositive") == 1)
    false_positive_counts = (
        false_positive_data.group_by(categorical_feature)
        .agg(pl.len().alias("count"))
        .sort("count", descending=True)
        .head(top_n)
    )

    false_negative_counts_pd = false_negative_counts.to_pandas()
    false_positive_counts_pd = false_positive_counts.to_pandas()

    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)

    axes[0].barh(
        false_negative_counts_pd[categorical_feature],
        false_negative_counts_pd["count"],
        color="salmon",
    )
    axes[0].invert_yaxis()  # Highest frequency at the top
    axes[0].set_title(f"Top {top_n} Categories for False Negatives")
    axes[0].set_xlabel("Count")
    axes[0].set_ylabel(categorical_feature)

    axes[1].barh(
        false_positive_counts_pd[categorical_feature],
        false_positive_counts_pd["count"],
        color="skyblue",
    )
    axes[1].invert_yaxis()  # Highest frequency at the top
    axes[1].set_title(f"Top {top_n} Categories for False Positives")
    axes[1].set_xlabel("Count")

    plt.tight_layout()
    plt.show()


def plot_cost_curve(
    y_prob: np.ndarray,
    y_true: np.ndarray,
    transaction_amounts: np.ndarray,
    false_positive_cost: float,
    thresholds: np.ndarray = np.linspace(0, 1, 101),
    figsize: Tuple[float, float] = (10, 6),
) -> None:
    """
    Plot the cost curve for different thresholds based on cost-sensitive optimization.

    Parameters
    ----------
    y_prob : np.ndarray
        Predicted probabilities for the positive class.
    y_true : np.ndarray
        True binary labels.
    transaction_amounts : np.ndarray
        Array of transaction amounts associated with each sample.
    false_positive_cost : float
        Cost of a false positive (e.g., manual inspection cost in hours).
    thresholds : np.ndarray, default=np.linspace(0, 1, 101)
        Array of thresholds to evaluate.
    figsize : Tuple[float, float], default=(10, 6)
        Figure size for the plot.

    Returns
    -------
    None
        Displays the cost curve plot.
    """
    costs: np.ndarray = np.zeros(len(thresholds))

    for i, threshold in enumerate(thresholds):
        predictions = (y_prob >= threshold).astype(int)
        # False positives and false negatives
        false_positives = np.sum((predictions == 1) & (y_true == 0))
        false_negatives = np.sum((predictions == 0) & (y_true == 1))
        # Total cost
        cost = false_positives * false_positive_cost + np.sum(
            transaction_amounts[(predictions == 0) & (y_true == 1)]
        )
        costs[i] = cost

    plt.figure(figsize=figsize)
    plt.plot(thresholds, costs, label="Total Cost", marker="o")
    plt.title("Cost Curve for Threshold Optimization")
    plt.xlabel("Threshold")
    plt.ylabel("Total Cost")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_precision_recall_curve(
    y_true: np.ndarray, y_score: np.ndarray, figsize: Tuple[float, float] = (10, 6)
):
    """
    Plot a precision-recall curve.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels.
    y_score : np.ndarray
        Predicted scores or probabilities for the positive class.
    figsize : Tuple[float, float], default=(10, 6)
        Figure size.

    Returns
    -------
    None
        Displays the precision-recall curve.
    """
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    plt.figure(figsize=figsize)
    plt.plot(recall, precision, marker=".", label="Precision-Recall Curve")
    plt.title("Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc="best")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(
    model: Any,
    X: pl.DataFrame,
    y: np.ndarray,
    figsize: Tuple[float, float] = (10, 8),
    threshold: Optional[float] = None,
    **kwargs: Any,
):
    """
    Plot a confusion matrix for a classification model.

    Parameters
    ----------
    model : Any
        A fitted estimator or model pipeline with a `predict` and `predict_proba` method.
    X : pl.DataFrame
        Feature data for prediction.
    y : np.ndarray
        True binary labels.
    figsize : Tuple[float, float], default=(10, 8)
        Figure size.
    threshold : Optional[float], default=None
        Decision threshold for classification. If None, the default decision threshold is used.
    **kwargs : Dict[str, Any]
        Additional keyword arguments to pass to the `confusion_matrix` function.

    Returns
    -------
    None
        Displays a confusion matrix.
    """
    if threshold is not None:
        # Use predict_proba to calculate predictions based on threshold
        y_score = model.predict_proba(X)[:, 1]
        predictions = (y_score >= threshold).astype(int)
    else:
        # Default decision threshold (e.g., 0.5)
        predictions = model.predict(X)

    fig, ax = plt.subplots(figsize=figsize)
    labels = np.unique(y)
    cm = confusion_matrix(y, predictions, labels=labels, **kwargs)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues", xticks_rotation="vertical", ax=ax)
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    plt.show()


def plot_calibration_curve(
    model: Any,
    X: pl.DataFrame,
    y_true: np.ndarray,
    figsize: Tuple[float, float] = (15, 5),
    **kwargs: Any,
):
    """
    Plot a calibration curve for a binary classification model.

    Parameters
    ----------
    model: Any,
        A fitted estimator or model pipeline with a `predict_proba` method.
    X : pl.DataFrame
        Feature data for prediction.
    y_true : np.ndarray
        True binary labels.
    figsize : Tuple[float, float], default=(15, 5)
        Figure size.
    **kwargs : Dict[str, Any]
        Additional keyword arguments to pass to the `calibration_curve` function. See
        sklearn.calibration.calibration_curve for more information.

    Returns
    -------
    None
        Displays a calibration curve plot.
    """
    # Get predicted probabilities for the positive class
    y_prob = model.predict_proba(X)[:, 1]
    prob_true, prob_pred = calibration_curve(y_true=y_true, y_prob=y_prob, **kwargs)

    plt.figure(figsize=figsize)
    plt.plot(prob_pred, prob_true, "s-", label="Calibration curve")
    plt.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
    plt.title("Calibration Curve")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.legend(loc="best")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_permutation_importances(
    result: Bunch,
    feature_names: List[str],
    top_k: Optional[int] = None,
    metric_name: Optional[str] = None,
    figsize: Tuple[float, float] = (15, 5),
) -> pl.DataFrame:
    """
    Plot permutation importances and return a Polars DataFrame containing the importances.

    Parameters
    ----------
    result : sklearn.utils.Bunch
        The result of `permutation_importance`, containing importances for each feature.
    feature_names : List[str]
        The names of the features corresponding to the columns of the input data.
    top_k : Optional[int], default=None
        Number of top features to plot based on importance. If None, all features are plotted.
    metric_name : Optional[str], default=None
        Name of the metric for the x-axis label.
    figsize : Tuple[float, float], default=(15, 5)
        Figure size.

    Returns
    -------
    pl.DataFrame
        A Polars DataFrame containing the permutation importances for each feature.
    """
    # Sort feature indices based on mean importance
    sorted_importances_idx = result.importances_mean.argsort()
    sorted_features = [feature_names[i] for i in sorted_importances_idx]

    importances = pl.DataFrame(
        {
            sorted_features[i]: result.importances[sorted_importances_idx[i]]
            for i in range(len(sorted_importances_idx))
        }
    )

    if top_k is not None:
        top_features = sorted_features[-top_k:]
        importances = importances.select(top_features)

    fig, ax = plt.subplots(figsize=figsize)
    importances.to_pandas().plot.box(vert=False, whis=10, ax=ax)
    ax.set_title(f"Permutation Importances (Top {top_k})")
    ax.axvline(x=0, color="k", linestyle="--")
    ax.set_xlabel("Decrease in " + metric_name if metric_name else "Metric")
    plt.show()
    return importances


class StudyVisualizer(object):
    """
    Class for visualizing hyperparameter tuning via Optuna
    """

    def __init__(self, study: optuna.study.Study) -> None:
        """
        Parameters
        ----------
        study : optuna.study.Study
            Optuna study instance.
        """
        self.study = study
        self.plot_func_dict = {
            "plot_optimization_history": optuna.visualization.plot_optimization_history,
            "plot_slice": optuna.visualization.plot_slice,
            "plot_parallel_coordinate": optuna.visualization.plot_parallel_coordinate,
            "plot_contour": optuna.visualization.plot_contour,
            "plot_param_importances": optuna.visualization.plot_param_importances,
        }

    def _static_plot(
        self, plot_func: str, figsize: Tuple[float, float], **kwargs
    ) -> Image:
        """
        Create static plot.

        Parameters
        ----------
        figsize : Tuple[float, float]
            Figure size.
        **kwargs
            Keyword arguments to pass to the plot function.
        """
        fig = self.plot_func_dict[plot_func](self.study, **kwargs)
        fig.update_layout(width=figsize[0], height=figsize[1])
        fig_bytes = fig.to_image(format="png")

        return Image(fig_bytes)  # type: ignore[no-untyped-call]

    def plot_optimization_history(self, figsize: Tuple[float, float]) -> Image:
        """
        Plot optimization history.

        Parameters
        ----------
        figsize : Tuple[float, float]
            Figure size.

        Returns
        -------
        Image
            Image of the plot.
        """
        return self._static_plot("plot_optimization_history", figsize)

    def plot_param_importances(self, figsize: Tuple[float, float]) -> Image:
        """
        Plot parameter importances.

        Parameters
        ----------
        figsize : Tuple[float, float]
            Figure size.

        Returns
        -------
        Image
            Image of the plot.
        """
        return self._static_plot("plot_param_importances", figsize)

    def plot_parallel_coordinate(
        self, params: List[str], figsize: Tuple[float, float]
    ) -> Image:
        """
        Plot parallel coordinate.

        Parameters
        ----------
        params : List[str]
            List of parameters to plot.
        figsize : Tuple[float, float]
            Figure size.

        Returns
        -------
        Image
            Image of the plot.
        """
        return self._static_plot("plot_parallel_coordinate", figsize, params=params)

    def plot_contour(self, params: List[str], figsize: Tuple[float, float]) -> Image:
        """
        Plot contour.

        Parameters
        ----------
        params : List[str]
            List of parameters to plot.
        figsize : Tuple[float, float]
            Figure size.
        """
        return self._static_plot("plot_contour", figsize, params=params)

    def plot_slice(self, params: List[str], figsize: Tuple[float, float]) -> Image:
        """
        Plot slice.

        Parameters
        ----------
        params : List[str]
            List of parameters to plot.
        figsize : Tuple[float, float]
            Figure size.

        Returns
        -------
        Image
            Image of the plot.
        """
        return self._static_plot("plot_slice", figsize, params=params)
