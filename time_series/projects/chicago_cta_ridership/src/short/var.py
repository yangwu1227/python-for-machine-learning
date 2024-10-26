import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sktime.forecasting.compose import (ForecastingPipeline,
                                        TransformedTargetForecaster)
from sktime.forecasting.model_selection import ForecastingGridSearchCV
from sktime.forecasting.var import VAR
from sktime.transformations.compose import OptionalPassthrough
from sktime.transformations.series.boxcox import LogTransformer
from sktime.transformations.series.date import DateTimeFeatures
from sktime.transformations.series.detrend import Deseasonalizer, Detrender
from sktime.transformations.series.difference import Differencer
from src.base_trainer import BaseTrainer
from src.custom_utils import S3Helper


class VARTrainer(BaseTrainer):
    """
    This class implements methods for forecasting using Vector Autoregression (VAR).
    """

    def __init__(
        self,
        horizon: str,
        config_path: str,
        logger_name: str,
        config_name: str,
        s3_helper: S3Helper = None,
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

    def _create_model(self) -> ForecastingPipeline:
        """
        Create the var model pipeline with the following steps:

            1. Log transform the target
            2. Apply a first a round of seasonal differencing, then a round of first (ordinary) differencing to remove seasonality and trend
            3. Generate date features to be used as exogenous variables
            4. Forecast using VAR, tuning the trend parameter e.g., constant, constant + trend, constant, etc.

        The order of the lags will be selected based on BIC for more parsimonious model, since AICc tends to overfit by choosing models with
        more parameters (fewer degrees of freedom).

        Returns
        -------
        ForecastingPipeline
            A forecasting pipeline with the steps described above.
        """
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
                        transformer=Deseasonalizer(sp=self.config[self.horizon]["m"]),
                        passthrough=True,
                    ),
                ),
                (
                    "differencing",
                    OptionalPassthrough(
                        Differencer(lags=[self.config[self.horizon]["m"], 1])
                    ),
                ),
                (
                    "var",
                    VAR(
                        maxlags=None,  # Defaults to 12 * (nobs/100)^{1/4}
                        trend=None,  # This will be tuned
                        freq=self.config[self.horizon]["freq"],
                        ic="bic",
                    ),
                ),
            ]
        )

        forecasting_pipeline = ForecastingPipeline(
            [
                (
                    "date_features",
                    OptionalPassthrough(
                        DateTimeFeatures(
                            ts_freq=self.config[self.horizon]["freq"],
                            manual_selection=self.config[self.horizon]["date_features"],
                            keep_original_columns=False,
                        )
                    ),
                ),
                ("target", target_pipeline),
            ]
        )

        return forecasting_pipeline

    def cross_validate(self, verbose: int = 1, n_jobs: int = -1, refit: bool = True):
        self.setup_cross_validation()

        if self.model is None:
            self.logger.info("Creating model...")
            self.model = self._create_model()
        else:
            self.logger.info("Model already created, skipping creation...")

        # Hyperparameter tuning
        grid = {
            "target__var__trend": ["n", "c", "ct", "ctt"],
            "target__differencing__passthrough": [True, False],
            "target__deseasonalize__passthrough": [True, False],
            "target__detrend__passthrough": [True, False],
            "date_features__passthrough": [True, False],
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
        # Use X_test to generate date features for the test period
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
        self, full_model: bool, lags: int = None, auto_lag: bool = None
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
