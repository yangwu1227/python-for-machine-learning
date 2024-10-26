import argparse
import logging
import os
import warnings
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import pandas as pd
from custom_utils import S3Helper, SetUp
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.compose import (BaggingForecaster,
                                        ColumnEnsembleForecaster,
                                        TransformedTargetForecaster)
from sktime.forecasting.statsforecast import StatsForecastAutoETS
from sktime.transformations.bootstrap import STLBootstrapTransformer
from sktime.transformations.series.boxcox import LogTransformer


class Processor(object):
    """
    Processor class to process data for forecasting tasks.
    """

    def __init__(
        self, logger: logging.Logger, config: Dict[str, Any], s3_helper: S3Helper = None
    ):
        """
        Instantiate a Processor object.

        Parameters
        ----------
        logger : logging.Logger
            Logger object to log messages.
        config : Dict[str, Any]
            Dictionary containing configuration parameters.
        s3_helper : S3Helper, optional
            S3Helper object to read/write data from/to S3 bucket, by default None and a new S3Helper object will be created.

        Returns
        -------
        None
        """
        self.logger = logger
        if s3_helper is None:
            self.s3_helper = S3Helper()
        else:
            self.s3_helper = s3_helper
        self.config = config

    def _ingest_data(self, input_key: str) -> pd.DataFrame:
        """
        Ingest raw data from s3 bucket. Conduct basic data preprocessing.

        Parameters
        ----------
        input_key : str
            S3 object key to read data from.

        Returns
        -------
        pd.DataFrame
            Dataframe containing cleaned raw data.
        """
        data = self.s3_helper.read_parquet(obj_key=input_key)

        # Convert service date to datetime
        data[self.config["date_col"]] = pd.to_datetime(data[self.config["date_col"]])
        # Sort by service date
        data.sort_values(by=self.config["date_col"], inplace=True)
        # Remove duplicates
        data.drop_duplicates(keep="first", inplace=True)
        # Drop 'total_rides' column
        data.drop(columns=["total_rides"], inplace=True)
        # Map string column 'day_type' to integer
        data[self.config["predictor"]] = (
            data[self.config["predictor"]]
            .map(self.config["day_type_map"])
            .astype(pd.Int8Dtype())
        )

        return data

    def _resample(self, data: pd.DataFrame, freq: str) -> pd.DataFrame:
        """
        Resample data to the given frequency.

        Parameters
        ----------
        data : pd.DataFrame
            Dataframe containing cleaned raw data.
        freq : str
            Frequency to resample data to.

        Returns
        -------
        pd.DataFrame
            Dataframe containing resampled data.
        """
        data.copy()
        # Drop 'day_type' column
        data.drop(columns=[self.config["predictor"]], inplace=True)
        data = (
            data.resample(rule=freq, on=self.config["date_col"])
            .median()
            .astype(np.int32)
        )
        # Reset index as a column so downstream modeling tasks can use the date column
        data.reset_index(inplace=True)

        return data

    def _create_covid_model(self) -> ColumnEnsembleForecaster:
        """
        Create the bagging ets model pipeline with the following steps:

                1. Log transform the target
                2. Apply bagging forecaster (AutoETS) to automatically select trend, seasonality, and error type with season length

        Returns
        -------
        ColumnEnsembleForecaster
            A forecaster containing ets model pipeline for each target.
        """
        # Bagging ets model pipeline
        bagging_ets = BaggingForecaster(
            bootstrap_transformer=STLBootstrapTransformer(
                n_series=10,  # Generate 10 bootstrapped samples
                sp=self.config["long"]["m"],  # Seasonal periodicity
                sampling_replacement=True,
                robust=True,
            ),
            forecaster=StatsForecastAutoETS(
                season_length=self.config["long"]["m"], model="ZZZ"
            ),
        )

        forecasters = []
        for target in self.config["targets"]:
            # Create pipeline
            target_pipeline = TransformedTargetForecaster(
                [("log_transform", LogTransformer()), ("bagging_ets", bagging_ets)]
            )

            forecasters.append((target, target_pipeline, target))

        # Ensemble the forecasters
        ensemble_forecaster = ColumnEnsembleForecaster(forecasters=forecasters)

        return ensemble_forecaster

    def _forecast_covid(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Use a bagging (Bootstrap Aggregation) ETS model trained on
        historical data prior to January 2020 to forecast all values
        after January 2020. The pipeline is simple and consists of
        a log transformation of the data prior to training the model.

        Parameters
        ----------
        data : pd.DataFrame
            Dataframe containing cleaned raw data, resampled to monthly frequency.

        Returns
        -------
        pd.DataFrame
            Dataframe containing forecasted values and all historical values prior to January 2020.
        """
        data = data.copy()
        # Set date column as the index
        data[self.config["date_col"]] = pd.to_datetime(data[self.config["date_col"]])
        data.sort_values(by=self.config["date_col"], inplace=True)
        data.set_index(self.config["date_col"], inplace=True)

        # Extract targets
        y = data[self.config["targets"]]

        # Split into train and test period (after January 2020)
        y_train = y.loc[y.index < self.config["covid_start"]]
        y_test = y.loc[y.index >= self.config["covid_start"]]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = self._create_covid_model()
            model.fit(y=y_train)
            fh = ForecastingHorizon(y_test.index, is_relative=False)
            y_pred = model.predict(fh=fh)

        # Concatenate train and test predictions
        forecast_data = pd.concat([y_train, y_pred], axis=0)

        # Convert targets to integers using floor function
        forecast_data = forecast_data.apply(np.floor).astype(np.int32).reset_index()

        return forecast_data.rename(
            columns={"index": "service_date"}
        )  # Reset index as a column so downstream data splitting tasks can use the date column

    def short_horizon(self) -> None:
        """
        Ingest raw data from s3, filters the data based on
        the given start and end dates, and splits the data into
        train and test sets based on the given horizon. The
        train and test sets are then written to s3.
        """
        self.logger.info("Processing data for short-horizon forecasting")

        data = self._ingest_data(self.config["input_key"])

        # Configuration parameters
        start_date = pd.to_datetime(self.config["short"]["start_date"])
        end_date = pd.to_datetime(self.config["short"]["end_date"])
        forecast_horizon = self.config["short"]["forecast_horizon"]

        # Filter data based on start and end dates
        self.logger.info(
            f'Filtering data from {start_date.strftime("%Y-%m-%d")} to {end_date.strftime("%Y-%m-%d")}'
        )
        data_2023 = data.loc[
            (data[self.config["date_col"]] >= start_date)
            & (data[self.config["date_col"]] <= end_date)
        ]

        # Forecasting horizon
        self.logger.info(f"Forecasting horizon: {forecast_horizon}")
        test_period_indices = pd.Index(
            data_2023[self.config["date_col"]][-forecast_horizon:]
        )

        # Split into train and test
        self.logger.info("Splitting data into train and test sets")
        train_start, train_end = (
            min(data_2023[self.config["date_col"]]).strftime("%Y-%m-%d"),
            (test_period_indices[0] - pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        )
        test_start, test_end = (
            test_period_indices[0].strftime("%Y-%m-%d"),
            max(data_2023[self.config["date_col"]]).strftime("%Y-%m-%d"),
        )

        self.logger.info(f"Train period: {train_start} to {train_end}")
        self.logger.info(f"Test period: {test_start} to {test_end}")
        train = data_2023.loc[
            ~data_2023[self.config["date_col"]].isin(test_period_indices)
        ]
        test = data_2023.loc[
            data_2023[self.config["date_col"]].isin(test_period_indices)
        ]

        # Write train and test to s3
        for data, name in zip([train, test], ["train", "test"]):
            self.s3_helper.to_parquet(
                data=data,
                obj_key=os.path.join(
                    self.config["short"]["output_key"], f"{name}.parquet"
                ),
            )
            self.logger.info(
                f"Successfully written {name} data to s3 for short-horizon forecasting"
            )

    def long_horizon(self) -> None:
        """
        Ingest raw data from s3, train a bagging ETS model on data prior to Covid-19,
        and forecast all values since January 2020. This is done for long horizon forecasting tasks.
        Both the original and forecasted data are written to s3.
        """
        data = self._ingest_data(self.config["input_key"])
        # Resample to monthly frequency for long horizon
        monthly_data = self._resample(data=data, freq=self.config["long"]["freq"])

        # Counterfactual data
        self.logger.info("Forecasting counterfactual data for Covid-19")
        counterfactual_data = self._forecast_covid(data=monthly_data)

        # Forecast horizon
        self.logger.info(
            f'Forecasting horizon: {self.config["long"]["forecast_horizon"]} months ahead'
        )
        forecast_horizon = self.config["long"]["forecast_horizon"]
        # Get test period indices, the date indices are same for both original and counterfactual data, and so we only need to get it once
        test_period_indices = pd.Index(
            monthly_data[self.config["date_col"]][-forecast_horizon:]
        )

        train_start, train_end = (
            self.config["long"]["start_date"],
            (test_period_indices[0] - pd.DateOffset(months=1)).strftime("%Y-%m-%d"),
        )
        test_start, test_end = (
            test_period_indices[0].strftime("%Y-%m-%d"),
            self.config["long"]["end_date"],
        )
        self.logger.info(f"Train period: {train_start} to {train_end}")
        self.logger.info(f"Test period: {test_start} to {test_end}")

        data_names = ["original", "counterfactual"]
        data_splits = ["train", "test"]

        # Attach 'original' and 'counterfactual' prefix to the data column names
        monthly_data.columns = [
            "original_" + col if col != self.config["date_col"] else col
            for col in monthly_data.columns
        ]
        counterfactual_data.columns = [
            "counterfactual_" + col if col != self.config["date_col"] else col
            for col in counterfactual_data.columns
        ]
        # Combine original and counterfactual data
        combined_data = pd.merge(
            left=monthly_data,
            right=counterfactual_data,
            how="inner",
            on=self.config["date_col"],
        )

        self.logger.info(
            "Splitting data into train and test sets for long-horizon forecasting"
        )

        # For long horizon (36 months), we start on 2013-01-31 and end on 2023-06-30
        data_dict = {
            "train": combined_data.loc[
                (combined_data[self.config["date_col"]] >= pd.to_datetime(train_start))
                & (combined_data[self.config["date_col"]] <= pd.to_datetime(train_end))
            ],
            "test": combined_data.loc[
                (combined_data[self.config["date_col"]] >= pd.to_datetime(test_start))
                & (combined_data[self.config["date_col"]] <= pd.to_datetime(test_end))
            ],
        }
        self.logger.info(
            f"Successfully split data into train and test sets for long-horizon forecasting"
        )

        # Save data to s3
        for data_split in data_splits:
            self.s3_helper.to_parquet(
                data=data_dict[data_split],
                obj_key=os.path.join(
                    self.config["long"]["output_key"], f"{data_split}.parquet"
                ),
            )
            self.logger.info(
                f"Successfully written {data_split} data to s3 for long-horizon forecasting"
            )

        return None


def main():
    parser = argparse.ArgumentParser(description="Process data for forecasting tasks")
    parser.add_argument(
        "--config",
        type=str,
        default="config",
        help="This is the path to the configuration file, which defaults to the config directory in src",
    )
    parser.add_argument(
        "--process_task_type",
        type=str,
        choices=["short", "long"],
        help="Forecasting horizon determines the type of processing task",
    )
    args, _ = parser.parse_known_args()

    logger, config = SetUp(
        logger_name=f"processing_{args.process_task_type}",
        config_name="processing_config",
        config_path=args.config,
    ).setup()

    s3_helper = S3Helper(s3_bucket=config["s3_bucket"], s3_key=config["s3_key"])

    processor = Processor(logger=logger, config=config, s3_helper=s3_helper)

    if args.process_task_type == "short":
        processor.short_horizon()
    elif args.process_task_type == "long":
        processor.long_horizon()
    else:
        raise ValueError("Invalid process_task_type, must be one of short or long")

    return 0


if __name__ == "__main__":
    main()
