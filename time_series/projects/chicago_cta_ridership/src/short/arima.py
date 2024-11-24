# mypy: disable-error-code="union-attr"
import warnings
from typing import Dict, Union, Optional

import numpy as np
import pandas as pd
from sktime.forecasting.compose import (
    ColumnEnsembleForecaster,
    ForecastingPipeline,
    TransformedTargetForecaster,
)
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.model_selection import ForecastingGridSearchCV
from sktime.forecasting.statsforecast import StatsForecastAutoARIMA
from sktime.transformations.compose import OptionalPassthrough
from sktime.transformations.series.boxcox import LogTransformer
from sktime.transformations.series.date import DateTimeFeatures
from sktime.transformations.series.detrend import Deseasonalizer, Detrender
from src.base_trainer import BaseTrainer
from src.model_utils import S3Helper


class ArimaTrainer(BaseTrainer):
    """
    This class implements methods for forecasting using (S)ARIMA(X).
    """

    def __init__(
        self,
        horizon: str,
        config_path: str,
        logger_name: str,
        config_name: str,
        s3_helper: Optional[S3Helper] = None,
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
        Create the SARIMAX model pipeline with the following steps:

            1. Log transform the target
            2. Tunable step to detrend the target (on/off)
            3. Tunable step to deseasonalize the target (on/off)
            4. Tunable step to generate datetime features (on/off)
            5. Forecast using AutoARIMA to automatically select p, d, and q, and P, D, and Q, if needed

        Returns
        -------
        ColumnEnsembleForecaster
            A forecaster containing pipelines for each target.
        """
        forecasters = []
        for target in self.config["targets"]:
            target_pipeline = TransformedTargetForecaster(
                [
                    ("log_transform", LogTransformer()),
                    (
                        "detrend",
                        OptionalPassthrough(transformer=Detrender(), passthrough=True),
                    ),
                    (
                        "deseasonalize",
                        OptionalPassthrough(
                            transformer=Deseasonalizer(
                                sp=self.config[self.horizon]["m"]
                            ),
                            passthrough=True,
                        ),
                    ),
                    (
                        "auto_arima",
                        StatsForecastAutoARIMA(
                            sp=self.config[self.horizon]["m"],
                            n_jobs=-1,
                            stationary=False,  # Allow the I (integrate) in SARIMA
                            blambda=None,  # Since we log transform already
                        ),
                    ),
                ]
            )

            # Nest target pipeline inside the exogenous pipeline
            exogenous_pipeline = ForecastingPipeline(
                [
                    (
                        "datetime",
                        OptionalPassthrough(
                            transformer=DateTimeFeatures(
                                ts_freq=self.config[self.horizon]["freq"],
                                manual_selection=self.config[self.horizon][
                                    "date_features"
                                ],
                                keep_original_columns=True,
                            )
                        ),
                    ),
                    ("target", target_pipeline),
                ]
            )

            forecasters.append((target, exogenous_pipeline, target))

        ensemble_forecaster = ColumnEnsembleForecaster(forecasters=forecasters)

        return ensemble_forecaster

    def cross_validate(self, verbose: int = 1, n_jobs: int = -1, refit: bool = True):
        self.setup_cross_validation()

        if self.model is None:
            self.logger.info("Creating model...")
            self.model: ColumnEnsembleForecaster = self._create_model()
        else:
            self.logger.info("Model already created, skipping creation...")

        # Hyperparameter grid (there should be 2^3 = 8 combinations)
        grid = {}
        for target in self.config["targets"]:
            grid[f"{target}__target__detrend__passthrough"] = [True, False]
            grid[f"{target}__target__deseasonalize__passthrough"] = [True, False]
            grid[f"{target}__datetime__passthrough"] = [True, False]

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
            grid_search.fit(y=self.y_train, X=self.X_train)

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
        # Use X_test to generate date features for the test period in case they are needed (date_feature_passthrough = True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.y_pred = self.grid_search.predict(fh=self.test_fh, X=self.X_test)
            pi = self.__class__.extract_prediction_intervals(
                pi=self.grid_search.predict_interval(
                    fh=self.test_fh, X=self.X_test, coverage=level
                ),
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
            self.best_forecaster.fit(y=self.y_full, X=self.X_full)

        # Make out-of-sample forecasts
        self.logger.info("Making out-of-sample forecasts...")
        self.oos_fh = np.arange(1, self.config[self.horizon]["forecast_horizon"] + 1)

        # Generate a dummy times series for the next 'self.oos_fh' days with all 0s so we can generate date features
        oos_X = pd.Series(
            data=np.zeros(
                self.config[self.horizon]["forecast_horizon"]
            ),  # All 0 vector with length equal to the forecast horizon
            index=pd.date_range(
                start=self.y_full.index[
                    -1
                ],  # Start from the last date in the entire data set
                periods=self.oos_fh[
                    -1
                ],  # Move forward by the 'forecast horizon' of periods
                freq=self.config[self.horizon][
                    "freq"
                ],  # Use the same frequency as the data set
            ),
            name=self.config["predictor"],
        )

        if self.best_forecaster.is_fitted:
            self.y_forecast = self.best_forecaster.predict(fh=self.oos_fh, X=oos_X)
        pi = self.__class__.extract_prediction_intervals(
            pi=self.best_forecaster.predict_interval(
                fh=self.oos_fh, X=oos_X, coverage=level
            ),
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
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                residuals = self.best_forecaster.predict_residuals(
                    y=self.y_full, X=self.X_full
                )
        else:
            if self._attribute_is_none("grid_search"):
                raise ValueError(
                    "The grid search attribute is None, please run `cross_validate` method first"
                )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                residuals = self.grid_search.best_forecaster_.predict_residuals(
                    y=self.y_train, X=self.X_train
                )

        return self.__class__._diagnostic_tests(
            residuals=residuals, lags=lags, auto_lag=auto_lag
        )
