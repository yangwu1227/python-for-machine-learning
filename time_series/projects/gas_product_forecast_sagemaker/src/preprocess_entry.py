import argparse
import logging
import os
import warnings
from typing import Any, Dict, cast

import numpy as np
import pandas as pd
from hydra import compose, core, initialize
from model_utils import get_logger
from omegaconf import OmegaConf
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.compose import TransformedTargetForecaster
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.trend import STLForecaster
from sktime.transformations.series.boxcox import LogTransformer

logger = get_logger(__name__)

# ------------------------ STL with naive forecasting ------------------------ #


def forecast(
    y_train: pd.Series,
    y_test: pd.Series,
    config: Dict[str, Any],
    logger: logging.Logger,
) -> np.ndarray:
    """
    Function that forecasts Covid-19 data using STL forecasting. The goal is to fit a model that performs well in-sample so we
    can use this model to forecast Covid-19 data (in a counterfactual scenario) with confidence. The following pipeline is
    employed:

    1. Log transform the target variable
    2. Fit an STL forecasting model using the naive method to forecast the trend, seasonality, and residual components separately

    Parameters
    ----------
    y_train : pd.Series
        A series that contains the training data.
    y_test : pd.Series
        A series that contains the test (i.e. Covid-19 data) data.
    config : Dict[str, Any]
        A dictionary that contains the configuration parameters.
    logger : logging.Logger
        A logger that logs messages.

    Returns
    -------
    np.ndarray
        A numpy array that contains the forecasted Covid-19 data.
    """
    forecasters = {}
    components = ["trend", "seasonal", "resid"]
    for name in components:
        forecasters[name] = NaiveForecaster(sp=int(config["m"]), strategy="mean")

    # Pipeline for target
    target_pipeline = TransformedTargetForecaster(
        [
            ("log_transform", LogTransformer()),
            (
                "stl",
                STLForecaster(
                    sp=int(config["m"]),
                    robust=True,
                    forecaster_trend=forecasters["trend"],
                    forecaster_seasonal=forecasters["seasonal"],
                    forecaster_resid=forecasters["resid"],
                ),
            ),
        ]
    )

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        target_pipeline.fit(y_train)

    # Forecast Covid-19 data
    y_pred_covid = target_pipeline.predict(
        fh=ForecastingHorizon(y_test.index, is_relative=False)
    )

    return y_pred_covid


# ------------------------------ Main function ------------------------------ #


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test_mode",
        action="store_true",
        help="Reduce data size and do not save the data to disk for uploading if in test mode",
    )
    args, _ = parser.parse_known_args()

    core.global_hydra.GlobalHydra.instance().clear()
    initialize(version_base="1.2", config_path="config", job_name="preprocess")
    config: Dict[str, Any] = cast(
        Dict[str, Any],
        OmegaConf.to_container(compose(config_name="main"), resolve=True),
    )

    logger.info("Loading raw data...")
    data = pd.read_csv(
        os.path.join(config["preprocess_input"], "gas_data.csv"), index_col=0
    )
    data.index = pd.to_datetime(data.index)
    data.index.freq = config["freq"]

    # Only use more recent data for forecasting covid-19 data since earlier data (going back to the 90s) may be less relevant, but this can be changed in the config file
    data = data.loc[data.index >= config["preprocess_counterfactual_start_date"]]

    logger.info("Splitting data into train and test sets...")
    y_train, y_test = temporal_train_test_split(
        data, test_size=config["forecast_horizon"]
    )
    logger.info(
        f'Train set period: {y_train.index.min().strftime("%Y-%m-%d")} to {y_train.index.max().strftime("%Y-%m-%d")}'
    )
    logger.info(
        f'Test set period: {y_test.index.min().strftime("%Y-%m-%d")} to {y_test.index.max().strftime("%Y-%m-%d")}'
    )

    logger.info("Forecasting Covid-19 data...")
    y_test_covid = y_train.loc[y_train["covid_forecast"], "gas_product"]
    y_train_covid = y_train.loc[y_train.index < y_test_covid.index.min(), "gas_product"]

    if args.test_mode:
        logger.info("Running in test mode...")
        # In local test mode, take last 100 data points for training and first 10 data points for testing
        y_train_covid = y_train_covid.iloc[-100:]
        y_test_covid = y_test_covid.iloc[:10]

    logger.info(
        f'Covid Train set period: {y_train_covid.index.min().strftime("%Y-%m-%d")} to {y_train_covid.index.max().strftime("%Y-%m-%d")}'
    )
    logger.info(
        f'Covid Test set period: {y_test_covid.index.min().strftime("%Y-%m-%d")} to {y_test_covid.index.max().strftime("%Y-%m-%d")}'
    )
    logger.info(f"Covid-19 Train set size: {len(y_train_covid)}")
    logger.info(f"Covid-19 Test set size: {len(y_test_covid)}")
    y_pred_covid = forecast(y_train_covid, y_test_covid, config, logger)

    # Substitue the forecasted Covid-19 data for the actual Covid-19 data in y_train, composed of three segments
    gas_product_forecast = pd.concat(
        [
            # Values before the first Covid-19 data point
            y_train.loc[y_train.index < y_test_covid.index.min(), "gas_product"],
            # Forecast Covid-19 data
            y_pred_covid,
            # Values after the last Covid-19 data point
            y_train.loc[y_train.index > y_test_covid.index.max(), "gas_product"],
        ],
        axis=0,
    ).astype(np.int16)

    gas_product_forecast.name = "gas_product_forecast"
    y_train = pd.concat([y_train, gas_product_forecast], axis=1)

    logger.info("Saving data...")
    if not args.test_mode:
        for data, name in zip([y_train, y_test], ["train", "test"]):
            data.drop(columns=["covid_forecast"]).to_csv(
                os.path.join(config["preprocess_output"], f"{name}/{name}.csv"),
                index=True,
            )
            logger.info(
                f'Saved {name} data to {os.path.join(config["preprocess_output"], f"{name}/{name}.csv")}'
            )
    logger.info("Preprocessing job completed successfully!")

    return 0


if __name__ == "__main__":
    main()
