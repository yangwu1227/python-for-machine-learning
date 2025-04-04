from typing import Optional, Union

import numpy as np
import pandas as pd
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.compose import (
    ColumnEnsembleForecaster,
    TransformedTargetForecaster,
)
from sktime.forecasting.model_selection import ForecastingGridSearchCV
from sktime.split import SlidingWindowSplitter

from src.base_trainer import BaseTrainer
from src.model_utils import S3Helper


class LongTrainer(BaseTrainer):
    """
    This class subclasses the BaseTrainer class and overrides the setup_cross_validation method
    to set up the cross-validation parameters for long-term forecasting. The two differences are:

    1. We select either the 'original' or 'counterfactual' data depending on the `data_type` attribute.
    2. We use the pandas PeriodIndex to represent the monthly periods in the data.

    The `data_type` attribute is used to select either the 'original' or 'counterfactual' data.
    The `data_type` attribute is set to 'original' by default, but can be set to 'counterfactual'
    to select the counterfactual data.
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
        self.best_forecaster: Optional[
            Union[ColumnEnsembleForecaster, TransformedTargetForecaster]
        ] = None
        self.grid_search: Optional[ForecastingGridSearchCV] = None
        self.y_pred: Optional[pd.DataFrame] = None
        self.y_forecast: Optional[pd.DataFrame] = None
        self.oos_fh: Optional[ForecastingHorizon] = None
        self.data_type: str = data_type

    def setup_cross_validation(self) -> None:
        """
        Override the base class method to set up the cross-validation parameters for long-term forecasting.
        """
        if self._attribute_is_none("y_train") and self._attribute_is_none("y_test"):
            self.logger.info("Ingesting data...")
            data_dict = self.load_and_process_data()

            # Select either the original or counterfactual data
            data_cols = [
                col for col in data_dict["train"].columns if self.data_type in col
            ]

            # Rename the columns to remove the data type prefix, e.g. 'original' or 'counterfactual'
            self.y_train = data_dict["train"][data_cols].rename(
                columns={
                    col: col.replace(f"{self.data_type}_", "") for col in data_cols
                }
            )
            self.y_test = data_dict["test"][data_cols].rename(
                columns={
                    col: col.replace(f"{self.data_type}_", "") for col in data_cols
                }
            )
            self.y_full = pd.concat([self.y_train, self.y_test], axis=0)

            # Set index to PeriodIndex
            freq = self.config[self.horizon]["freq"]
            self.y_train.index = pd.PeriodIndex(self.y_train.index, freq=freq)
            self.y_test.index = pd.PeriodIndex(self.y_test.index, freq=freq)
            self.y_full.index = pd.PeriodIndex(self.y_full.index, freq=freq)
        else:
            self.logger.info("Data already ingested, skipping ingestion...")

        if self._attribute_is_none("cv"):
            self.logger.info("Creating cross-validation splitter...")
            self.test_fh = ForecastingHorizon(self.y_test.index, is_relative=False)
            self.cv = SlidingWindowSplitter(
                fh=np.arange(1, len(self.test_fh) + 1),
                window_length=self.config[self.horizon]["window_length"],
                step_length=self.config[self.horizon]["step_length"],
            )
        else:
            self.logger.info(
                "Cross-validation splitter already created, skipping creation..."
            )
