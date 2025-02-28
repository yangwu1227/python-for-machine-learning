import os
import sys
from typing import List, Tuple, Union

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import clear_output, display
from scipy.stats import norm
from seaborn import boxplot, kdeplot
from sktime.utils.plotting import plot_correlations
from statsmodels.tsa.stattools import grangercausalitytests


class ExploratoryDataAnalyzer(object):
    """
    Class for exploratory data analysis.
    """

    def __init__(self, data: pd.DataFrame) -> None:
        """
        Initialize the class with the data source.

        Parameters
        ----------
        data : pd.DataFrame
            Data source containing time series data.
        """
        self.data = data.copy()

    def _check_freq(self, freq: str) -> None:
        """
        Type and value check for the frequency parameter.

        Parameters
        ----------
        freq : str
            The frequency of the time series--- 'M' for month end, 'Q' for quarter end, 'Y' for year end.
        """
        if not (isinstance(freq, str) and (freq in ["M", "Q", "Y"])):
            raise TypeError("The freq must be a string of either M, Q, or Y")
        return None

    def _check_agg_func(self, agg_func: str) -> None:
        """
        Type and value check for the aggregation function parameter.

        Parameters
        ----------
        agg_func : str
            The aggregation function to use--- mean or median.
        """
        if not (isinstance(agg_func, str) and (agg_func in ["mean", "median"])):
            raise TypeError("The agg_func must be a string of either mean or median")
        return None

    @staticmethod
    def plot_correlations(
        series: pd.Series,
        lags: int,
        suptitle: str,
        fig_size: Tuple[float, float],
        zero_lag: bool = False,
    ) -> None:
        """
        Plot autocorrelation and partial autocorrelation plots for a given time series.

        Parameters
        ----------
        series : pd.Series
            The time series to plot.
        lags : int
            The number of lags to plot.
        suptitle : str
            The title of the plot.
        fig_size : Tuple[float, float]
            The size of the figure.
        zero_lag : bool, optional
            Whether to include the zero lag, by default False.

        Returns
        -------
        None
            The function displays the plot and doesn't return any value.
        """
        # This method return a tuple of (fig, axes), so we unpack it
        fig, _ = plot_correlations(
            series=series, lags=lags, zero_lag=zero_lag, suptitle=suptitle
        )
        fig.set_size_inches(*fig_size)
        plt.show()

    @staticmethod
    def plot_ccf(
        x: Union[pd.Series, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        maxlags: int,
        title: str,
        fig_size: Tuple[float, float],
        ci: float = 0.95,
    ) -> None:
        """
        Plot the cross-correlation between two time series. For a white noise time series, the auto-correlations are approximately normally distributed with a mean of zero and
        a variance of 1/n, where n is the number of observations. This is under the assumption that the data are IID (independent and identically distributed).

        Parameters
        ----------
        x : Union[pd.Series, np.ndarray]
            The first time series.
        y : Union[pd.Series, np.ndarray]
            The second time series.
        maxlags : int
            The maximum number of lags to plot.
        title : str
            The title of the plot.
        fig_size : Tuple[float, float]
            The size of the figure.
        ci : float, optional
            The confidence interval, by default 0.95.
        """
        x_standardized = (x - x.mean()) / x.std()
        y_standardized = (y - y.mean()) / y.std()

        # CI for white noise
        ci_lim = norm.ppf((1 + ci) / 2) / np.sqrt(len(x))

        fig, ax = plt.subplots(figsize=fig_size)
        ax.xcorr(x=x_standardized, y=y_standardized, maxlags=maxlags)
        ax.axhline(
            ci_lim,
            color="blue",
            linestyle="--",
            label=f"{ci * 100}% Significance Level (White Noise)",
        )
        ax.axhline(-ci_lim, color="blue", linestyle="--")
        ax.set_title(title)
        ax.set_xlabel("Lag")
        ax.legend()
        plt.show()

    @staticmethod
    def plot_granger_causality(
        data: pd.DataFrame, maxlags: int, fig_size: Tuple[float, float]
    ) -> None:
        """
        Plot the p-values of the Granger Causality tests for different lags.

        Parameters
        ----------
        data : pd.DataFrame
            The data to test for Granger Causality.
        maxlags : int
            The maximum number of lags to plot.
        fig_size : Tuple[float, float]
            The size of the figure.
        """
        # Suppress the standard output (temporary hack before statsmodels fixes the issue of printing to stdout)
        original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")
        gc_res = grangercausalitytests(x=data, maxlag=maxlags)
        # Restore the standard output
        sys.stdout = original_stdout

        # Extract p-values for each test
        ssr_ftest_pvals = np.array(
            [gc_res[i + 1][0]["ssr_ftest"][1] for i in range(len(gc_res))]
        )
        ssr_chi2test_pvals = np.array(
            [gc_res[i + 1][0]["ssr_chi2test"][1] for i in range(len(gc_res))]
        )
        lrtest_pvals = np.array(
            [gc_res[i + 1][0]["lrtest"][1] for i in range(len(gc_res))]
        )
        params_ftest_pvals = np.array(
            [gc_res[i + 1][0]["params_ftest"][1] for i in range(len(gc_res))]
        )

        plt.figure(figsize=fig_size)
        lags = np.arange(1, len(gc_res) + 1)
        plt.plot(lags, ssr_ftest_pvals, "-o", label="SSR F-test")
        plt.plot(lags, ssr_chi2test_pvals, "-o", label="SSR Chi2-test")
        plt.plot(lags, lrtest_pvals, "-o", label="LR test")
        plt.plot(lags, params_ftest_pvals, "-o", label="Params F-test")

        plt.axhline(
            0.05, color="red", linestyle="--", label="Significance Level (0.05)"
        )
        plt.suptitle("P-values of Granger Causality Tests for Different Lags")
        plt.title(f"X1: {data.columns[0]}, X2: {data.columns[1]}")
        plt.xlabel("Lags")
        plt.ylabel("P-value")
        plt.gca().invert_yaxis()  # Smaller p-values at the top
        plt.legend(loc="upper right")
        plt.grid(True, which="both", linestyle="--", linewidth=0.5)
        plt.tight_layout()

        plt.show()

    def plot_correlations_within_year(
        self,
        selections: List[int],
        var: str,
        lags: int,
        suptitle: str,
        fig_size: Tuple[float, float],
        zero_lag: bool = False,
    ) -> None:
        """
        Display an interactive plot based on user selections using ipywidgets. The plot
        shows the autocorrelation and partial autocorrelation plots for a given year.

        Parameters
        ----------
        selections : List[int]
            The list of years to choose from.
        var : str
            The variable to plot.
        lags: str
            The number of lags to plot.
        suptitle : str
            The title of the plot.
        fig_size : Tuple[float, float]
            The size of the figure.
        zero_lag : bool, optional
            Whether to include the zero lag, by default False.

        Returns
        -------
        None
            The function displays interactive widgets and doesn't return any value.
        """

        def display_plot(year: int) -> None:
            year_subset = self.data.loc[self.data["service_date"].dt.year == year, var]
            self.__class__.plot_correlations(
                series=year_subset,
                lags=lags,
                zero_lag=zero_lag,
                suptitle=f"{suptitle} ({year})",
                fig_size=fig_size,
            )

        dropdown = widgets.Dropdown(
            options=selections, value=selections[0], description="Year:"
        )

        def on_dropdown_change(change) -> None:
            clear_output(wait=True)  # type: ignore[no-untyped-call]
            display(dropdown)  # type: ignore[no-untyped-call]
            display_plot(change["new"])

        display_plot(dropdown.value)

        dropdown.observe(on_dropdown_change, names="value")
        display(dropdown)  # type: ignore[no-untyped-call]

    def plot_correlations_between_years(
        self,
        var: str,
        lags: int,
        agg_func: str,
        freq: str,
        suptitle: str,
        fig_size: Tuple[float, float],
        zero_lag: bool = False,
    ) -> None:
        """
        This plot shows the autocorrelation and partial autocorrelation plots for a monthly series. The user can choose
        to aggregate the series using different aggregation functions--- mean or median. The user can also choose the
        frequency of the series--- monthly, quarterly, or yearly.

        Parameters
        ----------
        var : str
            The variable to plot.
        lags: str
            The number of lags to plot.
        agg_func : str
            The aggregation function to use--- mean or median.
        freq : str
            The frequency of the series.-- M for month end, Q for quarter end, Y for year end.
        suptitle : str
            The title of the plot.
        fig_size : Tuple[float, float]
            The size of the figure.
        zero_lag : bool, optional
            Whether to include the zero lag, by default False.

        Returns
        -------
        None
        """
        self._check_agg_func(agg_func)
        self._check_freq(freq)

        series = (
            self.data[["service_date", var]]
            .set_index("service_date")
            .resample(rule=freq)
            .agg(agg_func)
        )

        self.__class__.plot_correlations(
            series=series,
            lags=lags,
            zero_lag=zero_lag,
            suptitle=f"{suptitle} (Aggregation = {agg_func.title()}, Resampling Freq = {freq.upper()})",
            fig_size=fig_size,
        )

    def plot_series_between_years(
        self,
        var: str,
        agg_func: str,
        freq: str,
        title: str,
        fig_size: Tuple[float, float],
    ) -> None:
        """
        Plot a time series for a given variable between years. The user can choose to aggregate the series using different
        aggregation functions--- mean or median. The user can also choose the frequency of the series--- monthly, quarterly,
        or yearly.

        Parameters
        ----------
        var : str
            The variable to plot.
        agg_func : str
            The aggregation function to use--- mean or median.
        freq : str
            The frequency of the series.-- M for month end, Q for quarter end, Y for year end.
        title : str
            The title of the plot.
        fig_size : Tuple[float, float]
            The size of the figure.
        """
        self._check_agg_func(agg_func)
        self._check_freq(freq)

        series = (
            self.data[["service_date", var]]
            .set_index("service_date")
            .resample(rule=freq)
            .agg(agg_func)
        )

        fig, ax = plt.subplots(figsize=fig_size)
        ax.plot(series)
        ax.set_title(
            f"{title} (Aggregation = {agg_func.title()}, Resampling Freq = {freq.upper()})"
        )
        plt.show()

    def plot_conditional_distribution_and_timeplot(
        self, selections: List[int], var: str, fig_size: Tuple[float, float]
    ) -> None:
        """
        Display an interactive plot based on user selections using ipywidgets. The plot
        shows the conditional distribution and time plot of a variable grouped by day_type for a given year.

        Parameters
        ----------
        selections : List[int]
            The list of years to choose from.
        var : str
            The variable to plot.
        fig_size : Tuple[float, float]
            The size of the figure.

        Returns
        -------
        None
            The function displays interactive widgets and doesn't return any value.
        """

        def display_plot(year: int) -> None:
            year_subset = self.data[self.data["service_date"].dt.year == year]

            fig, axes = plt.subplots(1, 2, figsize=fig_size)

            # Conditional Distribution Plot
            kdeplot(data=year_subset, x=var, hue="day_type", ax=axes[0])
            axes[0].set_title(f"Conditional Distribution of {var} ({year})")

            # Time Plot
            for day_type, group in year_subset.groupby("day_type"):
                axes[1].plot(group["service_date"], group[var], label=day_type)
            axes[1].set_title(f"Time Plot of {var} ({year})")

            plt.tight_layout()
            plt.show()

        dropdown = widgets.Dropdown(
            options=selections, value=selections[0], description="Year:"
        )

        def on_dropdown_change(change) -> None:
            clear_output(wait=True)  # type: ignore[no-untyped-call]
            display(dropdown)  # type: ignore[no-untyped-call]
            display_plot(change["new"])

        display_plot(dropdown.value)

        dropdown.observe(on_dropdown_change, names="value")
        display(dropdown)  # type: ignore[no-untyped-call]

    def plot_monthly_distribution_by_year(
        self, selections: List[int], fig_size: Tuple[int, int]
    ) -> None:
        """
        Display an interactive plot based on user selections using ipywidgets. The plot
        shows the conditional distribution of bus and rail boardings grouped by month for a given year.

        Parameters
        ----------
        selections : List[int]
            The list of years to choose from.
        fig_size : Tuple[int, int]
            The size of the figure.

        Returns
        -------
        None
            The function displays interactive widgets and doesn't return any value.
        """

        def display_plot(year: int) -> None:
            year_subset_bus = self.data[self.data["service_date"].dt.year == year][
                "bus"
            ]
            year_subset_rail = self.data[self.data["service_date"].dt.year == year][
                "rail_boardings"
            ]
            year_subset_service_date = self.data[
                self.data["service_date"].dt.year == year
            ]["service_date"]

            fig, axes = plt.subplots(2, 1, figsize=fig_size)

            boxplot(
                x=year_subset_service_date.dt.month,
                y=year_subset_bus.values,
                ax=axes[0],
            )
            axes[0].set_title(f"Bus Boardings by Month ({year})")
            axes[0].set_xlabel("Month")
            axes[0].set_ylabel("Bus Boardings")

            boxplot(
                x=year_subset_service_date.dt.month,
                y=year_subset_rail.values,
                ax=axes[1],
            )
            axes[1].set_title(f"Rail Boardings by Month ({year})")
            axes[1].set_xlabel("Month")
            axes[1].set_ylabel("Rail Boardings")

            plt.tight_layout()
            plt.show()

        # Dropdown widget for year selection
        dropdown = widgets.Dropdown(
            options=selections, value=selections[0], description="Year:"
        )

        # Handler for dropdown value change
        def on_dropdown_change(change) -> None:
            clear_output(wait=True)  # type: ignore[no-untyped-call]
            display(dropdown)  # type: ignore[no-untyped-call]
            display_plot(change["new"])

        display_plot(dropdown.value)

        dropdown.observe(on_dropdown_change, names="value")
        display(dropdown)  # type: ignore[no-untyped-call]
