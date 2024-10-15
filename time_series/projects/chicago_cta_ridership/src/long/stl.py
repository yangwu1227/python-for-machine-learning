import warnings
from typing import Dict, List, Tuple, Union, Optional, Any
from IPython.display import Image

import pandas as pd
import numpy as np

from sktime.transformations.series.boxcox import LogTransformer
from sktime.transformations.compose import OptionalPassthrough
from sktime.transformations.series.detrend import Detrender
from sktime.transformations.series.detrend import Deseasonalizer
from sktime.forecasting.trend import STLForecaster
from sktime.forecasting.statsforecast import StatsForecastAutoETS

from sktime.forecasting.compose import TransformedTargetForecaster
from sktime.forecasting.compose import ColumnEnsembleForecaster
from sktime.forecasting.model_selection import ForecastingGridSearchCV

from src.custom_utils import S3Helper
from src.base_trainer import ForecastVisualizer
from src.long.long_trainer import LongTrainer


class STLTrainer(LongTrainer):
    """
    This class implements methods for forecasting using bagging STL forecaster with bootstrapped SLT transformation. The main steps are as follows:

    1. Apply STL transformation to the target, returning the trend, seasonal, and residual components
    2. Apply an STL forecaster to forecast the components separately
    3. Apply the inverse STL transformation to the forecasts of the components to obtain the final forecasts
    """

    def __init__(
        self,
        horizon: str,
        config_path: str,
        logger_name: str,
        config_name: str,
        s3_helper: S3Helper = None,
        data_type: str = "original",
    ) -> None:
        super().__init__(
            horizon=horizon,
            config_path=config_path,
            logger_name=logger_name,
            config_name=config_name,
            s3_helper=s3_helper,
        )
        self.best_forecaster = None
        self.grid_search = None
        self.y_pred = None
        self.y_forecast = None
        self.oos_fh = None
        self.data_type = data_type

    def _create_model(self) -> ColumnEnsembleForecaster:
        """
        Create an STL forecaster modeling pipeline. Details of the model pipeline
        are included in the notebook `notebooks/long_term_forecast.ipynb`.

        Returns
        -------
        ColumnEnsembleForecaster
            A forecaster containing an STL forecaster for each target.
        """
        forecasters = []
        for target in self.config["targets"]:
            target_pipeline = TransformedTargetForecaster(
                [
                    ("log_transform", LogTransformer()),
                    (
                        "detrend",
                        OptionalPassthrough(transformer=Detrender(), passthrough=False),
                    ),
                    (
                        "deseasonalize",
                        OptionalPassthrough(
                            transformer=Deseasonalizer(
                                sp=self.config[self.horizon]["m"]
                            ),
                            passthrough=False,
                        ),
                    ),
                    (
                        "slt",
                        STLForecaster(
                            sp=self.config[self.horizon]["m"],
                            robust=True,
                            forecaster_trend=StatsForecastAutoETS(
                                season_length=self.config[self.horizon]["m"]
                            ),
                            forecaster_seasonal=StatsForecastAutoETS(
                                season_length=self.config[self.horizon]["m"]
                            ),
                            forecaster_resid=StatsForecastAutoETS(
                                season_length=self.config[self.horizon]["m"]
                            ),
                        ),
                    ),
                ]
            )

            forecasters.append((target, target_pipeline, target))

        ensemble_forecaster = ColumnEnsembleForecaster(forecasters=forecasters)

        return ensemble_forecaster

    def cross_validate(self, verbose: int = 1, n_jobs: int = -1, refit: bool = True):
        self.setup_cross_validation()

        if self.model is None:
            self.logger.info("Creating model...")
            self.model = self._create_model()
        else:
            self.logger.info("Model already created, skipping creation...")

        # Hyperparameter tuning
        grid = {}
        for target in self.config["targets"]:
            grid[f"{target}__slt__forecaster_resid__damped"] = [True, False]
            grid[f"{target}__slt__forecaster_seasonal__damped"] = [True, False]
            grid[f"{target}__slt__forecaster_trend__damped"] = [True, False]
            grid[f"{target}__detrend__passthrough"] = [True, False]
            grid[f"{target}__deseasonalize__passthrough"] = [True, False]

        grid_search = ForecastingGridSearchCV(
            forecaster=self.model,
            cv=self.cv,
            strategy="refit",  # Refit on each train-val split (more computationally expensive)
            param_grid=grid,
            scoring=self.metric,
            n_jobs=n_jobs,
            refit=True,  # Refit on entire data with the best params
            verbose=verbose,
            error_score="raise",
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            grid_search.fit(y=self.y_train)

        self.logger.info(f"Best params: {grid_search.best_params_}")
        self.logger.info(f"Best score: {grid_search.best_score_}")

        self.grid_search = grid_search
        self.best_forecaster = grid_search.best_forecaster_.clone()

        return None

    def forecast(self) -> Dict[str, Union[pd.DataFrame, Dict[str, pd.DataFrame]]]:
        """
        Overwrite the base class method to make predictions on the test set. The STL forecaster does not support prediction intervals.
        """
        if self._attribute_is_none("grid_search"):
            raise ValueError(
                "The grid search attribute is None, please run `cross_validate` method first"
            )

        self.logger.info("Making predictions on test set...")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.y_pred = self.grid_search.predict(fh=self.test_fh)
        rmse = self.metric(y_true=self.y_test, y_pred=self.y_pred)
        metric_name = self.config[self.horizon]["metric"].upper()
        self.logger.info(f"Test {metric_name}: {rmse}")

        return {"y_train": self.y_train, "y_test": self.y_test, "y_pred": self.y_pred}

    def refit_and_forecast(
        self,
    ) -> Dict[str, Union[pd.DataFrame, Dict[str, pd.DataFrame]]]:
        """
        Overwrite the base class method to refit the model on the entire dataset and make out-of-sample forecasts. The STL forecaster does not support prediction intervals.
        """
        if self._attribute_is_none("best_forecaster"):
            raise ValueError(
                "Best forecaster attribute is None, please run `cross_validate` method first"
            )

        self.logger.info("Refitting model on entire data...")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.best_forecaster.fit(y=self.y_full)

        # Make out-of-sample forecasts
        self.logger.info("Making out-of-sample forecasts...")
        self.oos_fh = np.arange(1, self.config[self.horizon]["forecast_horizon"] + 1)
        if self.best_forecaster.is_fitted:
            self.y_forecast = self.best_forecaster.predict(fh=self.oos_fh)

        return {"y_train": self.y_full, "y_pred": self.y_forecast}

    def diagnostics(
        self, full_model: bool, lags: int = None, auto_lag: bool = None
    ) -> pd.DataFrame:
        if full_model:
            if self._attribute_is_none("best_forecaster"):
                raise ValueError(
                    "Best forecaster attribute is None, please run `cross_validate` method first"
                )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                residuals = self.best_forecaster.predict_residuals(y=self.y_full)
        else:
            if self._attribute_is_none("grid_search"):
                raise ValueError(
                    "The grid search attribute is None, please run `cross_validate` method first"
                )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                residuals = self.grid_search.best_forecaster_.predict_residuals(
                    y=self.y_train
                )

        return self.__class__._diagnostic_tests(
            residuals=residuals, lags=lags, auto_lag=auto_lag
        )

    @staticmethod
    def extract_prediction_intervals(
        self, pi: pd.DataFrame, level: float
    ) -> Dict[str, pd.DataFrame]:
        """
        Overwrite the base class method to extract prediction intervals from the STL forecaster. The STL forecaster does not support prediction intervals.
        """
        raise NotImplementedError(
            "The STL forecaster does not support prediction intervals"
        )

    @staticmethod
    def plot_forecast(
        start_date: str,
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
        # Create prediction intervals that are all zeros
        pi = {}
        for target in y_train.columns:
            pi[target] = pd.DataFrame(
                np.zeros((len(y_pred), 2)),
                index=y_pred.index,
                columns=["lower", "upper"],
            )

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
        # Create prediction intervals that are all zeros
        pi_original = {}
        pi_counterfactual = {}
        for target in y_train_original.columns:
            pi_original[target] = pd.DataFrame(
                np.zeros((len(y_pred_original), 2)),
                index=y_pred_original.index,
                columns=["lower", "upper"],
            )
            pi_counterfactual[target] = pd.DataFrame(
                np.zeros((len(y_pred_counterfactual), 2)),
                index=y_pred_counterfactual.index,
                columns=["lower", "upper"],
            )

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
