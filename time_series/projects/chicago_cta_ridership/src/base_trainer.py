import logging
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from IPython.display import Image
from plotly.graph_objs import Figure
from plotly.subplots import make_subplots
from scipy.stats import anderson, shapiro
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.compose import ColumnEnsembleForecaster
from sktime.performance_metrics.forecasting import (
    MeanAbsoluteError,
    MeanAbsolutePercentageError,
    MeanSquaredError,
)
from sktime.split import SlidingWindowSplitter
from src.custom_utils import S3Helper, SetUp
from statsmodels.stats.diagnostic import acorr_ljungbox


class BaseTrainer(object):
    """
    Forecasting base trainer. All trainers should inherit from this class.

    Attributes
    ----------
    horizon: str
        Horizon of the trainer, one of 'short', 'medium', or 'long'.
    logger: logging.Logger
        Logger object.
    config: Dict[str, Any]
        Config dictionary.
    metric: str
        Metric to use for evaluation.
    model: Any
        Model object from sktime.
    y_train: pd.DataFrame
        Training set targets.
    y_test: pd.DataFrame
        Testing set targets.
    y_full: pd.DataFrame
        Full set targets.
    X_train: pd.DataFrame
        Training set features.
    X_test: pd.DataFrame
        Testing set features.
    X_full: pd.DataFrame
        Full set features.
    test_fh: ForecastingHorizon
        Forecasting horizon for the testing set.
    cv: SlidingWindowSplitter
        Cross-validation splitter.
    s3_helper: S3Helper
        S3Helper object for interacting with S3.
    """

    def __init__(
        self,
        horizon: str,
        config_path: str,
        logger_name: str,
        config_name: str,
        s3_helper: S3Helper = None,
    ) -> None:
        """
        Initialize the trainer object.

        Parameters
        ----------
        horizon: str
            Horizon of the trainer, one of 'short', 'medium', or 'long'.
        config_path: str
            Path to the config file.
        logger_name: str
            Name of the logger.
        config_name: str
            Name of the config.
        s3_helper: S3Helper, optional
            S3Helper object to upload the model to S3.
        """
        self.horizon = horizon
        self.logger, self.config = SetUp(
            logger_name=logger_name, config_name=config_name, config_path=config_path
        ).setup()

        if s3_helper is None:
            self.s3_helper = S3Helper()
        else:
            self.s3_helper = s3_helper

        self.metric = {
            "mse": MeanSquaredError(square_root=False),
            "rmse": MeanSquaredError(square_root=True),
            "mae": MeanAbsoluteError(),
            "mape": MeanAbsolutePercentageError(symmetric=False),
            "smape": MeanAbsolutePercentageError(symmetric=True),
        }[self.config[self.horizon]["metric"]]

        self.model = None
        self.y_train = None
        self.X_train = None
        self.y_test = None
        self.X_test = None
        self.y_full = None
        self.X_full = None
        self.test_fh = None
        self.cv = None

    @property
    def horizon(self) -> str:
        """
        Horizon of the trainer.
        """
        return self._horizon

    @horizon.setter
    def horizon(self, horizon: str) -> None:
        """
        Validate and set the horizon of the trainer.

        Parameters
        ----------
        horizon: str
            Horizon of the trainer, one of 'short', 'medium', or 'long'.
        """
        if not isinstance(horizon, str):
            raise TypeError(f"Expected type str for horizon, got {type(horizon)}")
        if horizon not in ["short", "medium", "long"]:
            raise ValueError(f'The horizon must be one of "short", "medium", or "long"')
        self._horizon = horizon

    def _attribute_is_none(self, attribute: str) -> bool:
        """
        Check if an attribute is None.

        Parameters
        ----------
        attribute: str
            Name of the attribute.

        Returns
        -------
        bool
            True if the attribute is not None, False otherwise.
        """
        return getattr(self, attribute) is None

    def _load_data(self, obj_key: str) -> pd.DataFrame:
        """
        Load data from s3.

        Parameters
        ----------
        obj_key: str
            Key of the object in S3.

        Returns
        -------
        data: pd.DataFrame
            Dataframe of the data.
        """
        return self.s3_helper.read_parquet(obj_key)

    def _process_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Process the data by setting the index to 'service_date' with daily frequency and sorting the data by 'service_date'.

        Parameters
        ----------
        data: pd.DataFrame
            Dataframe of the data.

        Returns
        -------
        data: pd.DataFrame
            Dataframe of the data.
        """
        data["service_date"] = pd.to_datetime(data["service_date"])
        data.set_index("service_date", inplace=True)
        data.index.freq = self.config[self.horizon]["freq"]
        data.sort_index(inplace=True)
        return data.astype(
            np.float32
        )  # Should be float32 for sktime or else it will fail

    def load_and_process_data(self) -> Dict[str, pd.DataFrame]:
        """
        Load processed train and test data from S3 and process it.

        Returns
        -------
        data_dict: Dict[str, pd.DataFrame]
            Dictionary of dataframes for training and testing data.
        """
        data_dict = {}
        for data_name in ["train", "test"]:
            s3_obj_key = os.path.join(
                self.config[self.horizon]["output_key"], f"{data_name}.parquet"
            )
            data = self._load_data(obj_key=s3_obj_key)
            data = self._process_data(data=data)
            data_dict[data_name] = data

        return data_dict

    def setup_cross_validation(self) -> None:
        """
        Set up for cross-validation: ingest data if not already ingested, create the cross-validation splitter if not already created,
        and create the forecasting horizon. The number of train-val splits depends on a few factors:

            - The number of training examples, n
            - The size of the sliding window, w
            - The size of the forecasting horizon, h
            - The size of the step length, s

        Given n, w, h, and s, the number of train-val splits is given by:

            ((n - w - h) // s) + 1

        Where // is the floor division operator.
        """
        if (self._attribute_is_none("y_train")) and (self._attribute_is_none("y_test")):
            self.logger.info("Ingesting data...")
            data_dict = self.load_and_process_data()
            self.X_train = data_dict["train"][self.config["predictor"]]
            self.y_train = data_dict["train"][self.config["targets"]]
            self.X_test = data_dict["test"][self.config["predictor"]]
            self.y_test = data_dict["test"][self.config["targets"]]
            self.y_full = pd.concat([self.y_train, self.y_test], axis=0)
            self.X_full = pd.concat([self.X_train, self.X_test], axis=0)
        else:
            self.logger.info("Data already ingested, skipping ingestion...")

        if self._attribute_is_none("cv"):
            self.logger.info("Creating cross-validation splitter...")
            self.test_fh = ForecastingHorizon(self.y_test.index, is_relative=False)
            self.cv = SlidingWindowSplitter(
                fh=np.arange(1, len(self.test_fh)),
                window_length=self.config[self.horizon]["window_length"],
                step_length=self.config[self.horizon]["step_length"],
            )
        else:
            self.logger.info(
                "Cross-validation splitter already created, skipping creation..."
            )

    def cross_validate(self):
        """
        Time series cross-validation with either a sliding window splitter (short term horizon) or
        expanding window splitter (long term horizons).

        Parameters
        ----------
        verbose: int
            Verbosity level.
        n_jobs: int, optional
            Number of jobs to run in parallel, by default -1 (all processors are used).
        refit: bool, optional
            Refit the model on the entire data using the best params, by default True.
        """
        raise NotImplementedError(
            "Must implement cross_validation() method in child class"
        )

    def forecast(
        self, level: float = 0.95
    ) -> Dict[str, Union[pd.DataFrame, Dict[str, pd.DataFrame]]]:
        """
        Implements prediction logic for the model.

        Parameters
        ----------
        level: float, optional
            Prediction interval level, by default 0.95.

        Returns
        -------
        Dict[str, Union[pd.DataFrame, Dict[str, pd.DataFrame]]]
            Dictionary containing train, test, and prediction DataFrames--- one
            column for each target. The `pi` is a dictionary containing prediction
            intervals for each target.
        """
        raise NotImplementedError("Must implement forecast() method in child class")

    def refit_and_forecast(
        self, level: float = 0.95
    ) -> Dict[str, Union[pd.DataFrame, Dict[str, pd.DataFrame]]]:
        """
        Refit the model on the entire data (train + test) using the best params. Forecast
        the target for the out-of-sample horizon.

        Parameters
        ----------
        level: float, optional
            Prediction interval level, by default 0.95.

        Returns
        -------
        Dict[str, Union[pd.DataFrame, Dict[str, pd.DataFrame]]]
            Dictionary containing train and forecast DataFrames--- one
            column for each target. The `pi` is a dictionary containing prediction
            intervals for each target.
        """
        raise NotImplementedError(
            "Must implement refit_and_forecast() method in child class"
        )

    def diagnostics(
        self, full_model: bool, lags: int = None, auto_lag: bool = None
    ) -> pd.DataFrame:
        """
        Perform diagnostics tests on residuals.

        Parameters
        ----------
        full_model: bool
            Whether to use the full model or the best model from cross-validation.
        lags : int, optional
            Number of lags to use for Ljung-Box test, by default None.
        auto_lag : bool, optional
            Whether to automatically determine the number of lags to use for Ljung-Box test, by default None.

        Returns
        -------
        Dict[str, pd.DataFrame]
            Dictionary where keys are target names and values are DataFrames with test results.
        """
        raise NotImplementedError("Must implement diagnostics() method in child class")

    def __getstate__(self) -> Dict[str, Any]:
        """
        This method is called when pickling the object. It returns a dictionary of the attributes that should be pickled.
        If any sub-class of BaseTrainer has attributes that should be pickled, then this method should be overridden if
        they are not python built-in types.

        Returns
        -------
        Dict[str, Any]
            Dictionary of attributes that should be pickled.
        """
        state = self.__dict__.copy()

        # Don't pickle the S3Helper object, which is stateless and be re-created
        state.pop("s3_helper", None)

        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """
        This method is called when unpickling the object. It sets the attributes of the unpickled object to the attributes
        of the pickled object. If any sub-class of BaseTrainer has attributes that should be restored, then this method
        should be overridden if they are not python built-in types.

        Parameters
        ----------
        state: Dict[str, Any]
            Dictionary of attributes of the pickled object.
        """
        self.__dict__.update(state)

        # Re-create the S3Helper object
        self.s3_helper = S3Helper()

        return None

    def upload_trainer(self, obj_key: str) -> None:
        """
        This method uploads the trainer object to S3 using joblib, except for the S3Helper object.

        Parameters
        ----------
        obj_key: str
            Key of the object in S3.
        """
        s3_helper = S3Helper()
        s3_helper.upload_joblib(obj=self, obj_key=obj_key)

        self.logger.info(f"Uploaded trainer object to {obj_key}")

        return None

    def download_trainer(self, obj_key: str) -> None:
        """
        This method downloads the trainer object from S3 using joblib.

        Parameters
        ----------
        obj_key: str
            Key of the object in S3.
        """
        s3_helper = S3Helper()
        trainer = s3_helper.download_joblib(obj_key=obj_key)
        self.__dict__.update(trainer.__dict__)
        # No need to manually reattach s3_helper, __setstate__ will handle it
        self.logger.info(f"Downloaded trainer object from {obj_key}")

        return None

    @staticmethod
    def extract_prediction_intervals(
        pi: pd.DataFrame, level: float
    ) -> Dict[str, pd.DataFrame]:
        """
        Extract the prediction intervals from the results returned from a forecaster's `predict_interval()` method.
        The intervals returned by any sktime forecaster has multi-level column names, so we need to index into the
        columns to get the (lower, upper) columns. THe first level is the target names, the second level is the
        prediction interval levels, and the third level is the (lower, upper) columns.

        Parameters
        ----------
        pi: pd.DataFrame
            Prediction intervals DataFrame with multi-indexed columns.
        level: float
            The level of the prediction interval, e.g. 0.95 for a 95% prediction interval.

        Returns
        -------
        Dict[str, pd.DataFrame]
            Dictionary of prediction intervals.
        """
        pi = pi.copy()
        pi_result = {}
        for target in pi.columns.levels[0]:
            pi_result[target] = pi[target][level]

        return pi_result

    @staticmethod
    def _diagnostic_tests(
        residuals: pd.DataFrame, lags: int = None, auto_lag: bool = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Perform diagnostic tests on residuals.

        Parameters
        ----------
        residuals : pd.DataFrame
            Dataframe of residuals for each target.
        lags : int, optional
            Number of lags to use for Ljung-Box test, by default None.
        auto_lag : bool, optional
            Whether to automatically determine the number of lags to use for Ljung-Box test, by default None.

        Returns
        -------
        Dict[str, pd.DataFrame]
            Dictionary where keys are target names and values are DataFrames with test results.
        """
        if lags is None:
            auto_lag = True
        else:
            auto_lag = False

        results = {}
        for target in residuals.columns:
            series = residuals[target]

            # Shapiro-Wilk Test
            sw_stat, sw_p_value = shapiro(series)
            sw_result = [
                "Shapiro-Wilk",
                "Normal",
                sw_p_value,
                sw_p_value < 0.01,
                sw_p_value < 0.05,
                sw_p_value < 0.10,
            ]

            # Ljung-Box Test
            lb_stat, lb_p_value = acorr_ljungbox(
                series, lags=lags, return_df=True, auto_lag=auto_lag
            ).values[0]
            lb_result = [
                "Ljung-Box",
                "No Autocorrelation",
                lb_p_value,
                lb_p_value < 0.01,
                lb_p_value < 0.05,
                lb_p_value < 0.10,
            ]

            tests_results = pd.DataFrame(
                [sw_result, lb_result],
                columns=[
                    "Test",
                    "Null Hypothesis",
                    "P-Value",
                    "Reject at 1%",
                    "Reject at 5%",
                    "Reject at 10%",
                ],
            )
            results[target] = tests_results

        return results

    @staticmethod
    def plot_forecast(
        start_date: str,
        pi: Dict[str, pd.DataFrame],
        y_train: pd.DataFrame,
        y_pred: pd.DataFrame,
        y_test: pd.DataFrame = None,
        static: bool = True,
        title: str = None,
        height: int = 450,
        width: int = 1200,
    ) -> Union[None, Image]:
        """
        Plot the forecast.

        Parameters
        ----------
        start_date: str
            Start date of the forecast.
        pi: Dict[str, pd.DataFrame]
            Dictionary of prediction intervals.
        y_train: pd.DataFrame
            Training set targets.
        y_test: pd.DataFrame
            Testing set targets.
        y_pred: pd.DataFrame
            Predictions.
        static: bool, optional
            Whether to save the plot as a static image, by default True.
        title: str, optional
            Title of the plot, by default None.
        height: int, optional
            Height of the plot, by default 450.
        width: int, optional
            Width of the plot, by default 1200.

        Returns
        -------
        Union[None, Image]
            None if static is False, otherwise an Image object of the plot.
        """
        return ForecastVisualizer.plot_forecast(
            start_date=start_date,
            pi=pi,
            y_train=y_train,
            y_test=y_test,
            y_pred=y_pred,
            static=static,
            title=title,
            height=height,
            width=width,
        )

    @staticmethod
    def plot_forecast_comparison(
        start_date: str,
        pi_original: Dict[str, pd.DataFrame],
        pi_counterfactual: Dict[str, pd.DataFrame],
        y_train_original: pd.DataFrame,
        y_pred_original: pd.DataFrame,
        y_train_counterfactual: pd.DataFrame,
        y_pred_counterfactual: pd.DataFrame,
        static: bool = True,
        title: str = None,
        height: int = 450,
        width: int = 1200,
    ) -> Union[None, Image]:
        """
        Plot the forecast for both bus and rail boardings. This method combines
        the forecast plots for both the counterfactual and original data.

        Parameters
        ----------
        start_date: str
            Start date of the forecast.
        pi_original: Dict[str, pd.DataFrame]
            Prediction intervals for the original data.
        pi_counterfactual: Dict[str, pd.DataFrame]
            Prediction intervals for the counterfactual data.
        y_train_original: pd.DataFrame
            Training data for the original data.
        y_pred_original: pd.DataFrame
            Predictions for the original data.
        y_train_counterfactual: pd.DataFrame
            Training data for the counterfactual data.
        y_pred_counterfactual: pd.DataFrame
            Predictions for the counterfactual data.
        static: bool, optional
            Whether to save the plot as a static image, by default True.
        title: str, optional
            Title of the plot, by default None.
        height: int, optional
            Height of the plot, by default 450.
        width: int, optional
            Width of the plot, by default 1200.

        Returns
        -------
        Union[None, Image]
            None if static is False, otherwise an Image object of the plot.
        """
        return ForecastVisualizer.plot_forecast_comparison(
            start_date=start_date,
            pi_original=pi_original,
            pi_counterfactual=pi_counterfactual,
            y_train_original=y_train_original,
            y_pred_original=y_pred_original,
            y_train_counterfactual=y_train_counterfactual,
            y_pred_counterfactual=y_pred_counterfactual,
            static=static,
            title=title,
            height=height,
            width=width,
        )


class ForecastVisualizer(object):
    """
    This class contains methods for visualizing forecasts.
    """

    @staticmethod
    def _prepare_data_for_plotting(
        start_date: str, **data_objs: Union[pd.DataFrame, Dict[str, pd.DataFrame]]
    ) -> Dict[str, pd.DataFrame]:
        """
        This method takes the dataframes and prepares them for plotting. It does the following:

            - Converts the index to a datetime index if it is a PeriodIndex using `to_timestamp()`
            - Slices the training data `y_train` from the start date
            - Makes a copy of the (mutable) dataframes to avoid modifying the original dataframes

        The expected keys for the `data` dictionary are:

            - `y_train` (original or counterfactual)
            - `y_test` (optional)
            - `y_pred` (original or counterfactual)
            - `pi` (a dictionary of prediction intervals for each target for the original or counterfactual data)

        Parameters
        ----------
        start_date: str
            Start date for zooming in on the plot based on the time series index.
        **data_objs: Union[pd.DataFrame, Dict[str, pd.DataFrame]]
            Dataframes for plotting. The expected keys are `y_train`, `y_test`, `y_pred`, and `pi`.

        Returns
        -------
        Dict[str, pd.DataFrame]
            Dictionary of dataframes for plotting.
        """
        data_dict = {}
        for key, data in data_objs.items():
            if data is not None and isinstance(data, pd.DataFrame):
                data = data.copy()
                if isinstance(data.index, pd.PeriodIndex):
                    data.index = data.index.to_timestamp()
                if key == "y_train":
                    data = data.loc[start_date:]
            elif data is not None and isinstance(data, dict):
                data = {target: pi.copy() for target, pi in data.items()}
                for target, pi in data.items():
                    if isinstance(pi.index, pd.PeriodIndex):
                        pi.index = pi.index.to_timestamp()
            data_dict[key] = data
        return data_dict

    @staticmethod
    def _update_layout_and_return_figure(
        fig: Figure, title: str, static: bool, height: int, width: int
    ) -> Optional[Image]:
        """
        Update the layout of the provided figure and return as a static image if requested.

        Parameters
        ----------
        fig : Figure
            The Plotly figure to be updated.
        title : str
            The title to be set for the figure.
        static : bool
            Determines if the figure should be returned as a static image. If False, the figure is displayed interactively.
        height : int
            The height of the figure in pixels.
        width : int
            The width of the figure in pixels.

        Returns
        -------
        Image or Figure
            A static image of the figure if `static` is True, otherwise the figure is displayed interactively.
        """
        fig.update_layout(title_text=title, height=height, width=width)
        print(type(fig))
        if static:
            fig_bytes = fig.to_image(format="png")
            return Image(fig_bytes)
        else:
            return fig

    @staticmethod
    def plot_forecast(
        start_date: str,
        pi: Dict[str, pd.DataFrame],
        y_train: pd.DataFrame,
        y_pred: pd.DataFrame,
        y_test: pd.DataFrame = None,
        static: bool = True,
        title: str = None,
        height: int = 450,
        width: int = 1200,
    ) -> Union[None, Image]:
        data_dict = ForecastVisualizer._prepare_data_for_plotting(
            start_date=start_date, y_train=y_train, y_test=y_test, y_pred=y_pred, pi=pi
        )
        y_train, y_test, y_pred, pi = (
            data_dict["y_train"],
            data_dict["y_test"],
            data_dict["y_pred"],
            data_dict["pi"],
        )

        num_vars = y_train.shape[1]
        subplot_titles = tuple(y_train.columns)

        if title is None:
            title = "CTA Forecast"

        fig = make_subplots(rows=1, cols=num_vars, subplot_titles=subplot_titles)

        for i, var_name in enumerate(y_train.columns):
            showlegend = i == 0

            # Training Data
            fig.add_trace(
                go.Scatter(
                    x=y_train.index,
                    y=y_train[var_name],
                    mode="lines",
                    name="Train",
                    line=dict(color="blue"),
                    showlegend=showlegend,
                ),
                row=1,
                col=i + 1,
            )

            # Prediction Intervals
            fig.add_trace(
                go.Scatter(
                    x=pi[var_name].index,
                    y=pi[var_name]["lower"],
                    mode="lines",
                    name="Lower PI",
                    line=dict(width=0),
                    showlegend=showlegend,
                ),
                row=1,
                col=i + 1,
            )
            fig.add_trace(
                go.Scatter(
                    x=pi[var_name].index,
                    y=pi[var_name]["upper"],
                    mode="lines",
                    name="Upper PI",
                    line=dict(width=0),
                    fill="tonexty",
                    fillcolor="rgba(0,100,80,0.2)",
                    showlegend=showlegend,
                ),
                row=1,
                col=i + 1,
            )

            # Predictions
            fig.add_trace(
                go.Scatter(
                    x=y_pred.index,
                    y=y_pred[var_name],
                    mode="lines",
                    name="Forecast",
                    line=dict(color="orange"),
                    showlegend=showlegend,
                ),
                row=1,
                col=i + 1,
            )

            # Test Data (if provided)
            if y_test is not None and var_name in y_test:
                fig.add_trace(
                    go.Scatter(
                        x=y_test.index,
                        y=y_test[var_name],
                        mode="lines",
                        name="Test",
                        line=dict(color="green"),
                        showlegend=showlegend,
                    ),
                    row=1,
                    col=i + 1,
                )

            y_axis_title = "Bus Ridership" if var_name == "bus" else "Rail Boardings"
            fig.update_yaxes(title_text=y_axis_title, row=1, col=i + 1)

        return ForecastVisualizer._update_layout_and_return_figure(
            fig=fig, title=title, static=static, height=height, width=width
        )

    @staticmethod
    def plot_forecast_comparison(
        start_date: str,
        pi_original: Dict[str, pd.DataFrame],
        pi_counterfactual: Dict[str, pd.DataFrame],
        y_train_original: pd.DataFrame,
        y_pred_original: pd.DataFrame,
        y_train_counterfactual: pd.DataFrame,
        y_pred_counterfactual: pd.DataFrame,
        static: bool = True,
        title: str = None,
        height: int = 450,
        width: int = 1200,
    ) -> Union[None, Image]:
        data_dict = ForecastVisualizer._prepare_data_for_plotting(
            start_date=start_date,
            y_train_original=y_train_original,
            y_pred_original=y_pred_original,
            y_train_counterfactual=y_train_counterfactual,
            y_pred_counterfactual=y_pred_counterfactual,
            pi_original=pi_original,
            pi_counterfactual=pi_counterfactual,
        )
        y_train_original, y_pred_original = (
            data_dict["y_train_original"],
            data_dict["y_pred_original"],
        )
        y_train_counterfactual, y_pred_counterfactual = (
            data_dict["y_train_counterfactual"],
            data_dict["y_pred_counterfactual"],
        )
        pi_original, pi_counterfactual = (
            data_dict["pi_original"],
            data_dict["pi_counterfactual"],
        )

        num_vars = y_train_original.shape[1]
        subplot_titles = tuple(y_train_original.columns)

        if title is None:
            title = "CTA Forecast"

        fig = make_subplots(rows=1, cols=num_vars, subplot_titles=subplot_titles)

        for i, var_name in enumerate(y_train_original.columns):
            showlegend = i == 0

            # Training Data
            fig.add_trace(
                go.Scatter(
                    x=y_train_original.index,
                    y=y_train_original[var_name],
                    mode="lines",
                    name="Train (Original)",
                    line=dict(color="blue"),
                    showlegend=showlegend,
                ),
                row=1,
                col=i + 1,
            )
            fig.add_trace(
                go.Scatter(
                    x=y_train_counterfactual.index,
                    y=y_train_counterfactual[var_name],
                    mode="lines",
                    name="Train (Counterfactual)",
                    line=dict(color="pink"),
                    showlegend=showlegend,
                ),
                row=1,
                col=i + 1,
            )

            # Prediction Intervals
            fig.add_trace(
                go.Scatter(
                    x=pi_original[var_name].index,
                    y=pi_original[var_name]["lower"],
                    mode="lines",
                    name="Lower PI (Original)",
                    line=dict(width=0),
                    showlegend=showlegend,
                ),
                row=1,
                col=i + 1,
            )
            fig.add_trace(
                go.Scatter(
                    x=pi_original[var_name].index,
                    y=pi_original[var_name]["upper"],
                    mode="lines",
                    name="Upper PI (Original)",
                    line=dict(width=0),
                    fill="tonexty",
                    fillcolor="rgba(0,100,80,0.2)",
                    showlegend=showlegend,
                ),
                row=1,
                col=i + 1,
            )
            fig.add_trace(
                go.Scatter(
                    x=pi_counterfactual[var_name].index,
                    y=pi_counterfactual[var_name]["lower"],
                    mode="lines",
                    name="Lower PI (Counterfactual)",
                    line=dict(width=0),
                    showlegend=showlegend,
                ),
                row=1,
                col=i + 1,
            )
            fig.add_trace(
                go.Scatter(
                    x=pi_counterfactual[var_name].index,
                    y=pi_counterfactual[var_name]["upper"],
                    mode="lines",
                    name="Upper PI (Counterfactual)",
                    line=dict(width=0),
                    fill="tonexty",
                    fillcolor="rgba(0,100,80,0.2)",
                    showlegend=showlegend,
                ),
                row=1,
                col=i + 1,
            )

            # Predictions
            fig.add_trace(
                go.Scatter(
                    x=y_pred_original.index,
                    y=y_pred_original[var_name],
                    mode="lines",
                    name="Forecast (Original)",
                    line=dict(color="orange"),
                    showlegend=showlegend,
                ),
                row=1,
                col=i + 1,
            )
            fig.add_trace(
                go.Scatter(
                    x=y_pred_counterfactual.index,
                    y=y_pred_counterfactual[var_name],
                    mode="lines",
                    name="Forecast (Counterfactual)",
                    line=dict(color="orange"),
                    showlegend=showlegend,
                ),
                row=1,
                col=i + 1,
            )

            y_axis_title = "Bus Ridership" if var_name == "bus" else "Rail Boardings"
            fig.update_yaxes(title_text=y_axis_title, row=1, col=i + 1)

        return ForecastVisualizer._update_layout_and_return_figure(
            fig=fig, title=title, static=static, height=height, width=width
        )
