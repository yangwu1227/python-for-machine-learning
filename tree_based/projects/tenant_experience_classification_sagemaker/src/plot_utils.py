from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import polars as pl
import seaborn as sns
from IPython.display import Image
from matplotlib.lines import Line2D
from sklearn.metrics import confusion_matrix


def plot_violin_plots(
    cluster_data: pl.DataFrame,
    cluster_label: str,
    lower_quantile: Optional[float] = 0.05,
    upper_quantile: Optional[float] = 0.95,
) -> None:
    """
    Plot violin plots for numerical columns in a Polars DataFrame grouped by a specified cluster label.
    Displays mean and median markers on the plots for each group, with an optional filter for a specified
    quantile range.

    Parameters
    ----------
    cluster_data : pl.DataFrame
        The Polars DataFrame containing the data to plot.
    cluster_label : str
        The name of the column representing the cluster labels to group data by.
    lower_quantile : float, optional, default=0.05
        The lower quantile to filter data (e.g., 0.05 for 5th percentile).
    upper_quantile : float, optional, default=0.95
        The upper quantile to filter data (e.g., 0.95 for 95th percentile).

    Returns
    -------
    None
    """
    if isinstance(cluster_data, pl.DataFrame):
        data = cluster_data.to_pandas()
    cluster_groups = data.groupby(cluster_label)

    numerical_cols = [col for col in data.columns if col != cluster_label]
    num_plots = len(numerical_cols)
    cols = int(np.sqrt(num_plots))
    rows = int(np.ceil(num_plots / cols))

    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(15, rows * 4))
    axes = axes.flatten()

    for idx, col in enumerate(numerical_cols):
        # Filter the column values based on the specified quantile range
        col_values = data[col]
        lower_bound = col_values.quantile(lower_quantile)
        upper_bound = col_values.quantile(upper_quantile)
        filtered_data = data.loc[
            (col_values >= lower_bound) & (col_values <= upper_bound)
        ]

        ax = axes[idx]
        sns.violinplot(
            data=filtered_data,
            x=cluster_label,
            y=col,
            hue=cluster_label,
            ax=ax,
            inner=None,
            palette="Set2",
        )

        # Calculate mean and median for each cluster in the filtered data
        filtered_groups = filtered_data.groupby(cluster_label)
        means = filtered_groups[col].mean()
        medians = filtered_groups[col].median()

        # Add mean and median markers
        for i, cluster in enumerate(means.index):
            mean = means[i]
            median = medians[i]
            ax.scatter(x=[i] * 2, y=[mean, median], c=["blue", "red"], marker="o")

            ymin, ymax = ax.get_ylim()
            offset = (ymax - ymin) * 0.02  # Adjust the offset as needed

            # Add text annotations for mean and median
            ax.text(
                i, mean + offset, f"{mean:.2f}", color="blue", ha="center", va="bottom"
            )
            ax.text(
                i, median - offset, f"{median:.2f}", color="red", ha="center", va="top"
            )

        ax.set_title(f"{col.replace('_', ' ').title()}")
        ax.set_xlabel(cluster_label)
        ax.set_ylabel(col)

        # Custom legend for mean and median markers
        legend_elements = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="blue",
                markersize=8,
                label="Mean",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="red",
                markersize=8,
                label="Median",
            ),
        ]
        ax.legend(handles=legend_elements)

    # Hide unused subplots if any
    for idx in range(num_plots, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(
    x_cluster_labels: pd.Series,
    y_cluster_labels: pd.Series,
    x_label: str,
    y_label: str,
    figsize: Tuple[int, int] = (10, 8),
    cmap: str = "Blues",
) -> None:
    """
    Plots a confusion matrix between two clustering label sets.

    Parameters
    ----------
    x_cluster_labels : pd.Series
        The column containing the first set of cluster labels (to be plotted on the x-axis).
    y_cluster_labels : pd.Series
        The column containing the second set of cluster labels (to be plotted on the y-axis).
    x_label : str
        The label for the x-axis.
    y_label : str
        The label for the y-axis.
    figsize : Tuple[int, int], optional
        Figure size for the plot (default is (10, 8)).
    cmap : str, optional
        Colormap for the heatmap (default is 'Blues').

    Returns
    -------
    None
    """
    # Determine unique labels from both cluster label sets
    unique_labels = sorted(
        list(set(x_cluster_labels.unique()) | set(y_cluster_labels.unique()))
    )

    # Compute the confusion matrix with y_true as y_cluster_labels (for y-axis)
    # And y_pred as x_cluster_labels (for x-axis)
    conf_matrix = confusion_matrix(
        y_true=y_cluster_labels, y_pred=x_cluster_labels, labels=unique_labels
    )

    conf_matrix_data = pd.DataFrame(
        conf_matrix,
        index=[f"{y_label}: {label}" for label in unique_labels],
        columns=[f"{x_label}: {label}" for label in unique_labels],
    )
    conf_matrix_data = conf_matrix_data.loc[
        ~(conf_matrix_data == 0).all(axis=1), ~(conf_matrix_data == 0).all(axis=0)
    ]

    plt.figure(figsize=figsize)
    sns.heatmap(conf_matrix_data, annot=True, fmt="d", cmap=cmap, cbar=False)
    plt.title(
        f"Confusion Matrix between {y_label.capitalize()} and {x_label.capitalize()} Cluster Labels"
    )
    plt.yticks(rotation=0)
    plt.xticks(rotation=45)
    plt.xlabel(x_label)  # x_pred labels on x-axis
    plt.ylabel(y_label)  # y_true labels on y-axis
    plt.tight_layout()
    plt.show()


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
