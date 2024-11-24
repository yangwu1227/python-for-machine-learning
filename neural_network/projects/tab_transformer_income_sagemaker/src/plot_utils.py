from typing import List, Tuple

import optuna
from IPython.display import Image


class StudyVisualizer:
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
        self.plot_func_dict = plot_functions = {
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
        figsize : Tuple[float]
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
        figsize : Tuple[float]
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
        figsize : Tuple[float]
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
        figsize : Tuple[float]
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
        figsize : Tuple[float]
            Figure size.

        Returns
        -------
        Image
            Image of the plot.
        """
        return self._static_plot("plot_slice", figsize, params=params)
