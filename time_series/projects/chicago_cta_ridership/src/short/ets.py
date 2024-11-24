# mypy: disable-error-code="union-attr"
import warnings
from typing import Dict, Optional, Union

import numpy as np
import pandas as pd
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.compose import (
    ColumnEnsembleForecaster,
    TransformedTargetForecaster,
)
from sktime.forecasting.model_selection import ForecastingGridSearchCV
from sktime.forecasting.statsforecast import StatsForecastAutoETS
from sktime.transformations.series.boxcox import LogTransformer

from src.base_trainer import BaseTrainer
from src.model_utils import S3Helper


class ETSTrainer(BaseTrainer):
    """
    This class implements methods for forecasting using ETS.
    """

    def __init__(
        self,
        horizon: str,
        config_path: str,
        logger_name: str,
        config_name: str,
        s3_helper: Optional[S3Helper] = None,
    ) -> None:
        super().__init__(
            horizon=horizon,
            config_path=config_path,
            logger_name=logger_name,
            config_name=config_name,
            s3_helper=s3_helper,
        )
        self.best_forecaster: Optional[ColumnEnsembleForecaster] = None
        self.grid_search: Optional[ForecastingGridSearchCV] = None
        self.y_pred: Optional[pd.DataFrame] = None
        self.y_forecast: Optional[pd.DataFrame] = None
        self.oos_fh: Optional[ForecastingHorizon] = None

    def _create_model(self) -> ColumnEnsembleForecaster:
        """
        Create the ets model pipeline with the following steps:

            1. Log transform the target
            2. Forecast using AutoETS to automatically select trend, seasonality, and error type with season length

        Returns
        -------
        ColumnEnsembleForecaster
            A forecaster containing ets model pipeline for each target.
        """
        forecasters = []
        for target in self.config["targets"]:
            target_pipeline = TransformedTargetForecaster(
                [
                    ("log_transform", LogTransformer()),
                    (
                        "ets",
                        StatsForecastAutoETS(
                            season_length=self.config[self.horizon]["m"],
                            model="ZZZ",  # Automatically select trend, seasonality, and error
                        ),
                    ),
                ]
            )

            forecasters.append((target, target_pipeline, target))

        # Ensemble the forecasters
        ensemble_forecaster = ColumnEnsembleForecaster(forecasters=forecasters)

        return ensemble_forecaster

    def cross_validate(self, verbose: int = 1, n_jobs: int = -1, refit: bool = True):
        self.setup_cross_validation()

        if self.model is None:
            self.logger.info("Creating model...")
            self.model: ColumnEnsembleForecaster = self._create_model()
        else:
            self.logger.info("Model already created, skipping creation...")

        # Hyperparameter grid (names can be obtained from `get_params()` method)
        grid = {
            f"{target}__ets__damped": [True, False] for target in self.config["targets"]
        }

        grid_search = ForecastingGridSearchCV(
            forecaster=self.model,
            cv=self.cv,
            strategy="refit",  # Refit on each train-val split (more computationally expensive)
            param_grid=grid,
            scoring=self.metric,
            n_jobs=n_jobs,
            refit=True,  # Refit on entire data with the best params
            verbose=verbose,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            grid_search.fit(y=self.y_train)

        self.logger.info(f"Best params: {grid_search.best_params_}")
        self.logger.info(f"Best score: {grid_search.best_score_}")

        self.grid_search = grid_search
        self.best_forecaster = grid_search.best_forecaster_.clone()

        return None

    def forecast(
        self, level: float = 0.95
    ) -> Dict[str, Union[pd.DataFrame, Dict[str, pd.DataFrame]]]:
        if self._attribute_is_none("grid_search"):
            raise ValueError(
                "The grid search attribute is None, please run `cross_validate` method first"
            )

        self.logger.info("Making predictions on test set...")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.y_pred = self.grid_search.predict(fh=self.test_fh)
            pi = self.__class__.extract_prediction_intervals(
                pi=self.grid_search.predict_interval(fh=self.test_fh, coverage=level),
                level=level,
            )
        rmse = self.metric(y_true=self.y_test, y_pred=self.y_pred)
        metric_name = self.config[self.horizon]["metric"].upper()
        self.logger.info(f"Test {metric_name}: {rmse}")

        return {
            "y_train": self.y_train,
            "y_test": self.y_test,
            "y_pred": self.y_pred,
            "pi": pi,
        }

    def refit_and_forecast(
        self, level: float = 0.95
    ) -> Dict[str, Union[pd.DataFrame, Dict[str, pd.DataFrame]]]:
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
        pi = self.__class__.extract_prediction_intervals(
            pi=self.best_forecaster.predict_interval(fh=self.oos_fh, coverage=level),
            level=level,
        )

        return {"y_train": self.y_full, "y_pred": self.y_forecast, "pi": pi}

    def diagnostics(
        self, full_model: bool, lags: Optional[int] = None, auto_lag: bool = False
    ) -> pd.DataFrame:
        if full_model:
            if self._attribute_is_none("best_forecaster"):
                raise ValueError(
                    "Best forecaster attribute is None, please run `cross_validate` method first"
                )
            residuals = self.best_forecaster.predict_residuals(y=self.y_full)
        else:
            if self._attribute_is_none("grid_search"):
                raise ValueError(
                    "The grid search attribute is None, please run `cross_validate` method first"
                )
            residuals = self.grid_search.best_forecaster_.predict_residuals(
                y=self.y_train
            )

        return self.__class__._diagnostic_tests(
            residuals=residuals, lags=lags, auto_lag=auto_lag
        )
