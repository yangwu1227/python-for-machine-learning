import argparse
import logging
import os
import warnings
from typing import Any, Dict, Optional, Tuple, Union, cast

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from hydra import compose, core, initialize
from matplotlib.ticker import MaxNLocator
from model_utils import get_logger
from omegaconf import OmegaConf
from scipy.stats import shapiro
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.compose import TransformedTargetForecaster
from sktime.forecasting.statsforecast import StatsForecastAutoARIMA
from sktime.performance_metrics.forecasting import MeanSquaredError
from sktime.split import SlidingWindowSplitter
from sktime.transformations.series.boxcox import LogTransformer
from sktime.transformations.series.detrend import Deseasonalizer, Detrender
from sktime.transformations.series.fourier import FourierFeatures
from statsmodels.stats.diagnostic import acorr_ljungbox

# ---------------------------------- Trainer --------------------------------- #


class TSTrainer(object):
    """
    Trainer class that implements methods to build, train, and evaluate a time series model
    for forecasting the next `h` weeks of gasoline product data.
    """

    def __init__(
        self,
        hyperparameters: Dict[str, Any],
        config: Dict[str, Any],
        y_train: Union[pd.DataFrame, pd.Series],
        y_test: Union[pd.DataFrame, pd.Series],
        logger: logging.Logger,
    ) -> None:
        """
        Constructor method.

        Parameters
        ----------
        hyperparameters : Dict[str, Any]
            A dictionary that contains the hyperparameters for the model.
        config : Dict[str, Any]
            A dictionary that contains the configuration parameters.
        y_train : Union[pd.DataFrame, pd.Series]
            A dataframe or Series that contains the training data.
        y_test : Union[pd.DataFrame, pd.Series]
            A dataframe or Series that contains the test data.
        logger : logging.Logger
            A logger that logs messages.
        """
        self.hyperparameters = hyperparameters
        self.config = config
        self.y_train = y_train
        self.logger = logger

    def _check_data(self, data: Union[pd.DataFrame, pd.Series], data_name: str) -> None:
        """
        Check that input data is a dataframe or Series, has a DatetimeIndex, and has a frequency.

        Parameters
        ----------
        data : Union[pd.DataFrame, pd.Series]
            A dataframe or Series that contains the data.
        data_name : str
            The name of the data (e.g. 'y_train', 'y_test').

        Raises
        ------
        TypeError
            If `data` is not a dataframe or Series.
        ValueError
            If `data` does not have a DatetimeIndex or a frequency.
        """
        if not (isinstance(data, pd.DataFrame) or isinstance(data, pd.Series)):
            raise TypeError(f"{data_name} must be a dataframe or series")
        if not isinstance(data.index, pd.PeriodIndex):
            raise ValueError(f"{data_name} must have a PeriodIndex")
        if data.index.freq is None:
            raise ValueError(f"{data_name} must have a frequency")

    def _check_is_fitted(
        self,
        model_obj: Dict[str, Union[TransformedTargetForecaster, FourierFeatures]],
        should_be_fitted: bool = True,
    ) -> bool:
        """
        Check that the model is fitted.

        Parameters
        ----------
        model_obj : Dict[str, Union[TransformedTargetForecaster, FourierFeatures]]
            A dictionary that contains the model pipeline and the Fourier features transformer.
        should_be_fitted : bool, optional
            Whether the model should be fitted or not, by default True.

        Returns
        -------
        bool
            True if the object is fitted, False otherwise.
        """
        components_to_check = {
            "target_pipeline": "Model pipeline",
            "fourier_transformer": "Fourier features transformer",
        }

        for component, component_name in components_to_check.items():
            is_fitted = model_obj[component].is_fitted
            if should_be_fitted and not is_fitted:
                raise ValueError(f"{component_name} is not fitted yet")
            elif not should_be_fitted and is_fitted:
                raise ValueError(f"{component_name} is already fitted")

        return True

    @property
    def y_train(self) -> pd.DataFrame:
        return self._y_train

    @y_train.setter
    def y_train(self, y_train: pd.DataFrame) -> None:
        self._check_data(y_train, "y_train")
        self._y_train = y_train.copy()

    def _create_model(
        self,
    ) -> Dict[str, Union[TransformedTargetForecaster, FourierFeatures]]:
        """
        Create a harmonic regression model and a Fourier features transformer.

        Returns
        -------
        Dict[str, Union[TransformedTargetForecaster, FourierFeatures]]
            A dictionary that contains the untrained harmonic regression model pipeline and the unfitted Fourier features transformer.
        """
        target_pipeline = TransformedTargetForecaster(
            [
                ("log_transform", LogTransformer()),
                (
                    "auto_arima",
                    StatsForecastAutoARIMA(
                        sp=self.config["m"],
                        seasonal=False,
                        information_criterion="aicc",
                        n_jobs=None,
                    ),
                ),
            ]
        )

        # If preprocess_detrend is true but preprocess_deseasonalize is false
        if (
            self.hyperparameters["preprocess_detrend"]
            and not self.hyperparameters["preprocess_deseasonalize"]
        ):
            # Optionally, insert a detrender transformer after log transform (step 0)
            target_pipeline.steps.insert(1, ("detrend", Detrender(model="additive")))
        # If preprocess_detrend is false but preprocess_deseasonalize is true
        elif (
            not self.hyperparameters["preprocess_detrend"]
            and self.hyperparameters["preprocess_deseasonalize"]
        ):
            # Optionally, insert a deseasonalizer transformer after log transform (step 0)
            target_pipeline.steps.insert(
                1,
                (
                    "deseasonalize",
                    Deseasonalizer(sp=int(self.config["m"]), model="additive"),
                ),
            )
        elif (
            self.hyperparameters["preprocess_detrend"]
            and self.hyperparameters["preprocess_deseasonalize"]
        ):
            # Insert both preprocess_detrend and preprocess_deseasonalize transformers after log transform (step 0)
            target_pipeline.steps.insert(1, ("detrend", Detrender(model="additive")))
            target_pipeline.steps.insert(
                2,
                (
                    "deseasonalize",
                    Deseasonalizer(sp=int(self.config["m"]), model="additive"),
                ),
            )
        else:
            pass

        fourier_transformer = FourierFeatures(
            sp_list=[self.config["m"]],
            fourier_terms_list=[self.hyperparameters["preprocess_fourier_k"]],
            freq=self.config["freq"],
        )

        return {
            "target_pipeline": target_pipeline,
            "fourier_transformer": fourier_transformer,
        }

    def _train_model(
        self,
        y_train: Union[pd.DataFrame, pd.Series],
        model_obj: Dict[str, Union[TransformedTargetForecaster, FourierFeatures]],
    ) -> Dict[str, Union[TransformedTargetForecaster, FourierFeatures]]:
        """
        Train the harmonic regression model and return the trained model along with the the
        fitted Fourier features transformer. The latter is needed to transform the test data
        for prediction. The Fourier features transformer is fitted on the training data and
        then used to transform the test data to avoid data leakage; interpolation is used to
        generate the Fourier features for out-of-sample predictions.

        Parameters
        ----------
        y_train : Union[pd.DataFrame, pd.Series]
            A dataframe or Series that contains the training data.
        model_obj : Dict[str, Union[TransformedTargetForecaster, FourierFeatures]]
            A dictionary that contains the untrained harmonic regression model pipeline and the unfitted Fourier features transformer.

        Returns
        -------
        Dict[str, Union[TransformedTargetForecaster, FourierFeatures]]
            A dictionary that contains the trained harmonic regression model pipeline and the fitted Fourier features transformer.
        """
        self._check_is_fitted(model_obj, should_be_fitted=False)

        X_train = model_obj["fourier_transformer"].fit_transform(y_train)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model_obj["target_pipeline"].fit(y_train, X=X_train)

        return model_obj

    def _evaluate_model(
        self,
        y_val: Union[pd.DataFrame, pd.Series],
        model_obj: Dict[str, Union[TransformedTargetForecaster, FourierFeatures]],
    ) -> float:
        """
        Evaluate the model on the validation data, returning the mean squared error (MSE).

        Parameters
        ----------
        y_val : Union[pd.DataFrame, pd.Series]
            A dataframe or Series that contains the validation data.
        model_obj : Dict[str, Union[TransformedTargetForecaster, FourierFeatures]]
            A dictionary that contains the trained harmonic regression model pipeline and the fitted Fourier features transformer.

        Returns
        -------
        float
            The mean squared error.
        """
        self._check_is_fitted(model_obj, should_be_fitted=True)

        fh = ForecastingHorizon(y_val.index, is_relative=False)
        y_pred = model_obj["target_pipeline"].predict(
            fh=fh, X=model_obj["fourier_transformer"].transform(y_val)
        )
        mse = MeanSquaredError()

        return mse(y_true=y_val, y_pred=y_pred)

    def cross_validate(self) -> None:
        """
        Time series cross-validation with sliding window splits. The number of train-val
        splits depends on a few factors.

            - The number of training examples, n
            - The size of the sliding window, w
            - The size of the forecasting horizon, h
            - The size of the step length, s

        Given n, w, and h, the number of train-val splits is given by:

            ((n - w - h) // s) + 1

        Where // is the floor division operator.
        """
        if self.hyperparameters["test_mode"]:
            window_size = self.config["test_window_size"]
        else:
            window_size = self.config["cv_window_size"]

        cv = SlidingWindowSplitter(
            fh=list(range(1, self.config["forecast_horizon"])),
            window_length=window_size,
            step_length=self.config["step_length"],
        )

        mse_scores = {}
        for fold, (train_indices, val_indices) in enumerate(cv.split(self.y_train)):
            fold_y_train, fold_y_val = (
                self.y_train.iloc[train_indices],
                self.y_train.iloc[val_indices],
            )

            self.logger.info(
                f"Training set size for fold {fold + 1}: {fold_y_train.shape[0]}"
            )
            self.logger.info(
                f"Validation set size for fold {fold + 1}: {fold_y_val.shape[0]}"
            )

            untrained_model_obj = self._create_model()
            trained_model_obj = self._train_model(
                y_train=fold_y_train, model_obj=untrained_model_obj
            )
            mse = self._evaluate_model(y_val=fold_y_val, model_obj=trained_model_obj)

            self.logger.info(f"MSE for fold {fold + 1}: {round(mse, 4)}")
            mse_scores[fold] = mse

        self.logger.info(
            f"Mean MSE across all splits: {round(np.mean(list(mse_scores.values())), 4)}"
        )

        return None

    def refit(
        self, data: Union[pd.DataFrame, pd.Series]
    ) -> Dict[str, Union[TransformedTargetForecaster, FourierFeatures, pd.Series]]:
        """
        Refit the model on the entire training data plus the test set. This is done after cross-validation
        to get the best hyperparameters. The refitted model is used to forecast the next `h` weeks of gasoline
        product data.

        Parameters
        ----------
        data : Union[pd.DataFrame, pd.Series]
            A dataframe or Series that contains the data for refitting the model.

        Returns
        -------
        Dict[str, Union[TransformedTargetForecaster, FourierFeatures, pd.Series]]
            A dictionary that contains

                - the refitted harmonic regression model pipelinethe
                - the fitted Fourier features transformer
                - the entire data set to be used to generate the Fourier features for out-of-sample predictions
        """
        untrained_model_obj = self._create_model()
        trained_model_obj = self._train_model(
            y_train=data, model_obj=untrained_model_obj
        )

        return {
            "target_pipeline": trained_model_obj["target_pipeline"],
            "fourier_transformer": trained_model_obj["fourier_transformer"],
            "data": data,
        }

    @staticmethod
    def generate_fourier_features(
        fourier_transformer: FourierFeatures,
        fh: int,
        y_full: Union[pd.DataFrame, pd.Series] = None,
        y_test: Union[pd.DataFrame, pd.Series] = None,
    ) -> pd.DataFrame:
        """
        Generate Fourier features for the test data, if `y_test` is not None. If `y_test` is None, then the Fourier features
        will be interpolated for out-of-sample forecasting. Only one of `y_test` or `y_full` should be provided.

        Parameters
        ----------
        fourier_transformer : FourierFeatures
            A fitted Fourier features transformer.
        fh : int
            The forecasting horizon for out-of-sample forecasting.
        y_full : Union[pd.DataFrame, pd.Series], optional
            A dataframe or Series that contains the entire data set, by default None.
        y_test : Union[pd.DataFrame, pd.Series], optional
            A dataframe or Series that contains the test data, by default None.

        Returns
        -------
        pd.DataFrame
            A dataframe that contains the Fourier features.
        """
        if fh <= 0 or not isinstance(fh, int):
            raise ValueError("Forecasting horizon (fh) must be a positive integer")

        # Only one of y_test or y_full should be provided
        if y_test is not None and y_full is not None:
            raise ValueError("Only one of y_test or y_full should be provided")

        if y_test is not None:
            # The index of y_test should be a PeriodIndex, which is what the Fourier features transformer expects
            X_test = fourier_transformer.transform(y_test)
            return X_test
        elif y_full is not None:
            y_full = y_full.copy()
            # Convert y_train index from period index to datetime index, since we need to get the max date of the data
            y_full.index = y_full.index.to_timestamp(freq=y_full.index.freq)
            max_date = y_full.index.max()
            # Create a dummy series with fh steps ahead of the last in-sample date
            dummy_series = pd.Series(
                data=0,
                index=pd.period_range(
                    start=max_date, periods=fh, freq=y_full.index.freq
                ),
            )
            # Convert back to period index, which is what the Fourier features transformer expects
            dummy_series.index = pd.PeriodIndex(
                data=dummy_series.index, freq=y_full.index.freq
            )
            # Generate Fourier features for the dummy series
            X_oos = fourier_transformer.transform(dummy_series)
            return X_oos

    @staticmethod
    def plot_sliding_cv_windows(
        w: int,
        s: int,
        h: int,
        start_date: str,
        end_date: str,
        freq: str,
        title: str = "CV Splits",
        figsize: Tuple[int, int] = (12, 8),
    ) -> None:
        """
        Plot the sliding windows for time series cross-validation.

        Parameters
        ----------
        w : int
            The sliding window size.
        s : int
            The step size.
        h : int
            The forecast horizon.
        start_date : str
            The start date of the time series.
        end_date : str
            The end date of the time series.
        freq : str
            The frequency of the time series.
        title : str, optional
            The plot title, by default 'CV Splits'
        figsize : Tuple[int, int]
            The figure size.
        """
        if w <= 0 or s <= 0 or h <= 0:
            raise ValueError(
                "Sliding window size, step size, and forecast horizon must be positive"
            )

        # Create the sliding window splitter
        cv = SlidingWindowSplitter(window_length=w, step_length=s, fh=range(0, h))
        y = pd.Series(
            np.zeros(len(pd.date_range(start_date, end_date, freq=freq))),
            index=pd.date_range(start_date, end_date, freq=freq),
        )

        fig, ax = plt.subplots(1, figsize=figsize)
        train_color, test_color = "#1f77b4", "#ff7f0e"

        # Plot each training and test window
        for i, (train, test) in enumerate(cv.split(y)):
            ax.plot(np.arange(len(y)), np.ones(len(y)) * i, marker="o", c="lightgray")
            ax.plot(
                train,
                np.ones(len(train)) * i,
                marker="o",
                c=train_color,
                label="Training Window" if i == 0 else "",
            )
            ax.plot(
                test,
                np.ones(len(test)) * i,
                marker="o",
                c=test_color,
                label="Test Window" if i == 0 else "",
            )

        ax.invert_yaxis()
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        xtickslocs = [tick for tick in ax.get_xticks() if tick in np.arange(len(y))]
        ax.set(
            title=title,
            ylabel="Window number",
            xlabel="Time",
            xticks=xtickslocs,
            xticklabels=y.iloc[xtickslocs].index,
        )
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[:2], labels[:2])  # To avoid repeating labels

        plt.show()

    @staticmethod
    def plot_forecast(
        target_pipeline: TransformedTargetForecaster,
        fourier_transformer: FourierFeatures,
        y_train: Union[pd.DataFrame, pd.Series],
        y_test: Union[pd.DataFrame, pd.Series] = None,
        conf: float = 0.95,
        forecast_horizon: int = 26,
        start_date: str = "2023-01-01",
    ) -> None:
        """
        Plot the forecasted gasoline product data. If `y_test` is not None, then the forecasted values are plotted against the
        actual values. Otherwise, only the forecasted values are plotted. In both cases, the training data is also plotted along
        with the prediction intervals.

        Parameters
        ----------
        target_pipeline : TransformedTargetForecaster
            A trained harmonic regression model pipeline.
        fourier_transformer : FourierFeatures
            A fitted Fourier features transformer.
        y_train : Union[pd.DataFrame, pd.Series]
            A dataframe or Series that contains the training data.
        y_test : Union[pd.DataFrame, pd.Series], optional
            A dataframe or Series that contains the test data, by default None.
        conf : float, optional
            The confidence level for the prediction intervals, by default 0.95.
        forecast_horizon : int, optional
            The number of weeks to forecast, by default 26. If `y_test` is not None, then this parameter is ignored.
        start_date : str, optional
            The start date for plotting the forecast, by default '2023-01-01'. This helps to zoom in on the forecasted values.
        """
        # Forecasting test data
        if y_test is not None:
            fh = ForecastingHorizon(y_test.index, is_relative=False)
            X_test = TSTrainer.generate_fourier_features(
                fourier_transformer=fourier_transformer,
                fh=y_test.shape[0],
                y_test=y_test,
            )
            y_pred = target_pipeline.predict(fh=fh, X=X_test)
            # The prediction interval returned by sktime has multi-level column names, so we need to index into the columns to get (lower, upper)
            pi = target_pipeline.predict_interval(fh=fh, X=X_test, coverage=conf)[0][
                conf
            ]
        # Forecasting out-of-sample
        else:
            X_oos = TSTrainer.generate_fourier_features(
                fourier_transformer=fourier_transformer,
                fh=forecast_horizon,
                y_full=y_train,
            )
            y_pred = target_pipeline.predict(
                fh=list(range(1, forecast_horizon + 1)), X=X_oos
            )
            pi = target_pipeline.predict_interval(
                fh=list(range(1, forecast_horizon + 1)), X=X_oos, coverage=conf
            )[0][conf]

        # Convert period index to datetime index for plotting
        y_train = y_train.copy()
        y_train.index = y_train.index.to_timestamp(freq=y_train.index.freq)
        y_pred.index = y_pred.index.to_timestamp(freq=y_pred.index.freq)

        # Use the start date to zoom in on the forecasted values
        y_train = y_train.loc[pd.to_datetime(start_date) :]

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(y_train, label="Train", color="#1f77b4")  # Blue
        ax.plot(y_pred, label="Forecast", color="#ff7f0e")  # Orange
        if y_test is not None:
            plt.plot(y_test, label="Test", color="#2ca02c")  # Green
        # Add prediction interval pi['lower'] and pi['upper'] as shaded regions
        ax.fill_between(
            pi.index,
            pi["lower"],
            pi["upper"],
            alpha=0.2,
            color="grey",
            label=f"{conf * 100}% prediction interval",
        )
        ax.legend()
        ax.set_title("Gasoline Product Forecast")
        ax.set_xlabel("Date")
        ax.set_ylabel("Gasoline Product (Thousand Barrels Per Day)")
        plt.show()

    @staticmethod
    def diagnostics(
        target_pipeline: TransformedTargetForecaster,
        fourier_transformer: FourierFeatures,
        fh: int,
        y_full: Union[pd.DataFrame, pd.Series],
        lags: Optional[int] = None,
        auto_lag: Optional[bool] = None,
    ) -> pd.DataFrame:
        """
        Perform diagnostics tests on the model. The tests performed are:

            - Shapiro-Wilk test for normality of residuals
            - Ljung-Box test for autocorrelation of residuals

        Parameters
        ----------
        target_pipeline : TransformedTargetForecaster
            A trained harmonic regression model pipeline.
        fourier_transformer : FourierFeatures
            A fitted Fourier features transformer.
        fh : int
            The forecasting horizon for out-of-sample forecasting.
        lags : int, optional
            The number of lags to use for the Ljung-Box test, by default None.
        auto_lag : bool, optional
            Whether to automatically select the number of lags for the Ljung-Box test, by default None.

        Returns
        -------
        pd.DataFrame
            A dataframe that contains the results of the diagnostics tests.
        """
        X_oos = TSTrainer.generate_fourier_features(
            fourier_transformer=fourier_transformer, fh=fh, y_full=y_full
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            residuals = target_pipeline.predict_residuals(y=y_full, X=X_oos)

        # Shapiro-Wilk Test
        sw_stat, sw_p_value = shapiro(residuals)
        sw_result = [
            "Shapiro-Wilk",
            "Normality",
            sw_p_value,
            sw_p_value < 0.01,
            sw_p_value < 0.05,
            sw_p_value < 0.10,
        ]

        # Ljung-Box Test
        lb_stat, lb_p_value = acorr_ljungbox(
            residuals, lags=lags, return_df=True, auto_lag=auto_lag
        ).values[0]
        lb_result = [
            "Ljung-Box",
            "No Autocorrelation",
            lb_p_value,
            lb_p_value < 0.01,
            lb_p_value < 0.05,
            lb_p_value < 0.10,
        ]

        return pd.DataFrame(
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


def main() -> int:
    # ---------------------------------- Set up ---------------------------------- #

    logger = get_logger(__name__)

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--train", type=str, default=os.environ["SM_CHANNEL_TRAIN"])
    parser.add_argument("--test", type=str, default=os.environ["SM_CHANNEL_TEST"])
    parser.add_argument("--preprocess_detrend", type=int)
    parser.add_argument("--preprocess_deseasonalize", type=int)
    parser.add_argument("--preprocess_fourier_k", type=int)
    parser.add_argument("--use_counterfactual_data", type=int)
    parser.add_argument("--test_mode", type=int)
    args, _ = parser.parse_known_args()

    core.global_hydra.GlobalHydra.instance().clear()
    initialize(version_base="1.2", config_path="config", job_name="train")
    config: Dict[str, Any] = cast(
        Dict[str, Any],
        OmegaConf.to_container(compose(config_name="main"), resolve=True),
    )

    # ----------------------------- Cross-validation ----------------------------- #

    if args.test_mode:
        logger.info("Running in local test mode...")
    else:
        logger.info("Running in SageMaker mode...")

    logger.info("Loading training and test data...")
    data = {}
    for channel, path in zip(["train", "test"], [args.train, args.test]):
        data[channel] = pd.read_csv(os.path.join(path, f"{channel}.csv"), index_col=0)
        data[channel].index = pd.PeriodIndex(
            data=data[channel].index, freq=config["freq"]
        )
    y_train = data["train"]
    y_test = data["test"]

    if args.use_counterfactual_data:
        logger.info("Using counterfactual data...")
        y_train = y_train["gas_product_forecast"]
        y_test = y_test["gas_product"]
        y_test.name = "gas_product_forecast"
    else:
        logger.info("Using original data...")
        y_train = y_train["gas_product"]
        y_test = y_test["gas_product"]
    logger.info(f"Number of training examples: {y_train.shape[0]}")
    logger.info(f"Number of test examples: {y_test.shape[0]}")

    ts_trainer = TSTrainer(
        hyperparameters={
            "preprocess_detrend": args.preprocess_detrend,
            "preprocess_deseasonalize": args.preprocess_deseasonalize,
            "preprocess_fourier_k": args.preprocess_fourier_k,
            "test_mode": args.test_mode,
        },
        config=config,
        y_train=y_train,
        y_test=y_test,
        logger=logger,
    )

    ts_trainer.cross_validate()

    # ------------------- Final training and model persistence ------------------- #

    models = {}
    # Evaluate the model performance by training on the entire training set and forecasting the test set
    logger.info("Training on the entire training set and forecasting the test set...")
    models["model_train"] = ts_trainer.refit(data=y_train)
    # One last round of training on the entire data (train and test)
    logger.info(
        "Training on the entire data set (train + test) for forecasting out-of-sample..."
    )
    models["model_full"] = ts_trainer.refit(data=pd.concat([y_train, y_test], axis=0))

    logger.info(
        "Saving both the model trained just on the training set and the model trained on the entire data set..."
    )
    for model_name, model_obj in models.items():
        joblib.dump(
            model_obj["target_pipeline"],
            os.path.join(args.model_dir, f"{model_name}_target_pipeline.joblib"),
        )
        joblib.dump(
            model_obj["fourier_transformer"],
            os.path.join(args.model_dir, f"{model_name}_fourier_transformer.joblib"),
        )
        # Convert period index to datetime index for saving the data
        model_obj["data"].index = model_obj["data"].index.to_timestamp(
            freq=config["freq"]
        )
        model_obj["data"].to_csv(
            os.path.join(args.model_dir, f"{model_name}_data.csv"), index=True
        )

    return 0


if __name__ == "__main__":
    main()
