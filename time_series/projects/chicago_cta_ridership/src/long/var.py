# mypy: disable-error-code="union-attr"
import warnings
from typing import Dict, Union, Optional

import numpy as np
import pandas as pd
from sktime.forecasting.model_selection import ForecastingGridSearchCV
from sktime.forecasting.compose import TransformedTargetForecaster
from sktime.forecasting.model_selection import ForecastingGridSearchCV
from sktime.forecasting.var import VAR
from sktime.forecasting.base import ForecastingHorizon
from sktime.transformations.compose import OptionalPassthrough
from sktime.transformations.series.boxcox import LogTransformer
from sktime.transformations.series.detrend import Deseasonalizer, Detrender
from src.model_utils import S3Helper
from src.long.long_trainer import LongTrainer


class VARTrainer(LongTrainer):
    """
    This class implements methods for forecasting using vector autoregression (VAR) models.
    """

    def __init__(
        self,
        horizon: str,
        config_path: str,
        logger_name: str,
        config_name: str,
        s3_helper: Optional[S3Helper] = None,
        data_type: str = "original",
    ) -> None:
        super().__init__(
            horizon=horizon,
            config_path=config_path,
            logger_name=logger_name,
            config_name=config_name,
            s3_helper=s3_helper,
        )
        self.best_forecaster: Optional[TransformedTargetForecaster] = None
        self.grid_search: Optional[ForecastingGridSearchCV] = None
        self.y_pred: Optional[pd.DataFrame] = None
        self.y_forecast: Optional[pd.DataFrame] = None
        self.oos_fh: Optional[ForecastingHorizon] = None
        self.data_type: str = data_type

    def _create_model(self) -> TransformedTargetForecaster:
        """
        Create the var model pipeline with the following steps:

            1. Log transform the target
            2. Tunable steps to detrend and deseasonalize the target
            3. Forecast using VAR, tuning the trend parameter e.g., constant, constant + trend, constant, etc.

        The order of the lags will be selected based on BIC for more parsimonious model, since AICc tends to overfit by choosing models with
        more parameters (fewer degrees of freedom).

        Returns
        -------
        TransformedTargetForecaster
            The VAR model pipeline.
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

        return target_pipeline

    def cross_validate(self, verbose: int = 1, n_jobs: int = -1, refit: bool = True):
        self.setup_cross_validation()

        if self.model is None:
            self.logger.info("Creating model...")
            self.model: TransformedTargetForecaster = self._create_model()
        else:
            self.logger.info("Model already created, skipping creation...")

        # Hyperparameter tuning
        grid = {
            "var__trend": ["n", "c", "ct", "ctt"],
            "detrend__passthrough": [True, False],
            "deseasonalize__passthrough": [True, False],
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
