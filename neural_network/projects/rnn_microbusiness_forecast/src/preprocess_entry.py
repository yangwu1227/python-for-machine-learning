import os
import sys
import argparse
import logging
import glob
from typing import List, Tuple, Dict, Union, Optional

import polars as pl
import numpy as np

from hydra import compose, initialize, core
from omegaconf import OmegaConf

# ------------------------------- Preprocessing ------------------------------ #


def preprocess(train_data: pl.DataFrame) -> pl.DataFrame:
    """
    This function applies preprocessing to the training data.

    Parameters
    ----------
    train_data : pl.DataFrame
        The training data to be preprocessed.

    Returns
    -------
    pl.DataFrame
        The preprocessed training data.
    """
    train_data = train_data.with_columns(
        # Convert 'first_day_of_month' column to datetime
        pl.col("first_day_of_month").str.to_date().alias("first_day_of_month")
    ).with_columns(
        # Extract year from 'first_day_of_month' column
        pl.col("first_day_of_month").dt.year().alias("year")
    )

    return train_data


# ---------------------------- Density adjustment ---------------------------- #


def density_adjustment(train_data: pl.DataFrame, census_data_dir: str) -> pl.DataFrame:
    """
    Microbusiness density is defined as microbusinesses per 100 people over the age of 18.
    The training data contains microbusiness density for years 2019 - 2022. The census
    data contains population estimates for years 2017 - 2021. In order to adjust the
    microbusiness density for each year in the training data, we need to multiply the
    microbusiness density for that year by the ratio of the population estimates for that
    year (minus 2 years) and year 2021 (most recent in the census). For example, to adjust
    the microbusiness density for 2019 (earliest in the trianing data), we need to multiply
    the microbusiness densities for all counties for 2019 by the ratio of the population
    estimates for (2019 - 2 = 2017) and 2021. This function returns a new DataFrame with a
    new 'microbusiness_density' column that has been adjusted for each year.

    Parameters
    ----------
    train_data : pl.DataFrame
        The training data to be adjusted.
    census_data_dir : str
        The directory containing the census data files.

    Returns
    -------
    pl.DataFrame
        The adjusted training data with a new 'microbusiness_density' column.
    """
    # -------------------------------- First loop -------------------------------- #

    # First loop to create dictionary mappings cfips to population by year
    queries = []
    census_years = []
    cols = ["GEO_ID", "S0101_C01_026E"]
    for file_path in glob.glob(os.path.join(census_data_dir, "*Data.csv")):
        # Skip the first row after the header since it is a second header
        query = pl.scan_csv(file_path, skip_rows_after_header=1).select(cols)

        # Split the GEO_ID column into two columns (the second one is cfips)
        query = (
            query.with_columns(
                [
                    pl.col("GEO_ID")
                    .str.split_exact(by="US", n=1, inclusive=False)
                    .struct.rename_fields(["prefix", "cfips"])
                    .alias("GEO_ID")
                ]
            )
            .unnest("GEO_ID")
            .with_columns(pl.col("cfips").cast(pl.Int32))
            .drop("prefix")
        )

        queries.append(query)

        # Extract year from file name e.g. 'ACSST5Y2019.S0101-Data.csv'
        census_year = int(os.path.basename(file_path).split(".", maxsplit=1)[0][-4:])
        census_years.append(census_year)

    # Execute the query plans in parallel returning a list of DataFrames
    census_data_frames = pl.collect_all(queries)

    # Index for 2021
    index_2021 = census_years.index(2021)
    census_data_2021 = census_data_frames[index_2021]
    # Mapping cfips to population for 2021
    cfips_pop_map_2021 = dict(census_data_2021.iter_rows())

    # -------------------------------- Second loop ------------------------------- #

    # Second loop to adjust microbusiness density for each year (2019 - 2022)
    train_data = train_data.with_columns(
        # Create a new column for 2021 population estimates
        pl.col("cfips").map_dict(cfips_pop_map_2021).alias("est_pop_2021")
    )
    # Census years go from 2017 - 2021, so add 2 to get the correct training data year
    for census_data, census_year in zip(census_data_frames, census_years):
        # Adjust microbusiness density for each training data year
        cfips_pop_map = dict(census_data.iter_rows())
        train_data_year = census_year + 2

        # For each training data year (2019 - 2022), multiply microbusiness density by (train_data_year - 2)_est_pop / est_pop_2021
        train_data = train_data.with_columns(
            pl.col("cfips").map_dict(cfips_pop_map).alias("est_pop")
        ).with_columns(
            pl.when(pl.col("year") == train_data_year)
            .then(
                pl.col("microbusiness_density")
                * (pl.col("est_pop") / pl.col("est_pop_2021"))
            )
            .otherwise(pl.col("microbusiness_density"))
        )

    # Drop columns that are no longer needed
    train_data = train_data.drop(["year", "est_pop"])

    return train_data


# ------------------------- Large and small counties ------------------------- #


def split_counties(
    train_data: pl.DataFrame, threshold: int = 50
) -> Dict[str, List[int]]:
    """
    This function splits the data into two groups: counties whose 2021 population estimates are larger than
    the 10th quantile value of all 2021 population estimates (across all counties) and those whose
    population estimates are smaller than the 10th quantile value. This is because we want to train on counties
    with larger populations, which have larger variances in terms their year-over-year microbusiness density changes.
    In addition, EDA shows that the bottom 10% of counties have virtually stable densities over time. Finally, we
    exclude outlier counties that have max microbusiness density values that are greater than 50.

    Parameters
    ----------
    train_data : pl.DataFrame
        The training data to be adjusted.
    threshold : int, optional
        The threshold for microbusiness density, by default 50.

    Returns
    -------
    Dict[str, List[int]]
        A dictionary mapping 'large_counties' and 'small_counties' to lists of cfips.
    """
    # Max density for each cfips
    train_data = (
        train_data.groupby("cfips")
        .agg(pl.col("microbusiness_density").max().alias("max_density_by_cfips"))
        .join(train_data, on="cfips", how="left")
    )

    # Find the cut-off quantile value for small counties (above which we consider a county large)
    quantile_val_10 = train_data.select(pl.col("est_pop_2021").quantile(0.1))[0, 0]

    # Filter to keep only large counties
    large_counties_cfips = (
        train_data.filter(
            (pl.col("est_pop_2021") >= quantile_val_10)
            & (pl.col("max_density_by_cfips") < threshold)
        )
        .select(pl.col("cfips"))
        .unique()["cfips"]
        .to_list()
    )

    small_counties_cfips = list(
        set(train_data["cfips"].to_list()) - set(large_counties_cfips)
    )

    return {
        "large_counties": large_counties_cfips,
        "small_counties": small_counties_cfips,
    }


def convert_to_numpy(
    train_data: pl.DataFrame,
    cfips: Dict[str, List[int]],
    num_predictions: int = 5,
    series_len: int = 18,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    This function converts the training data to a numpy array for training. Each county has 41 months of historical data
    from August 2019 thru Dec 2022. By default, we will train our models with Feb 2020 thru Dec 2022 only, which is 35 months.
    For each county, we will break this single time series of 35 months into 18 time series of 18 months each. For example,
    the first time series window will be Feb 2020 through and including July 2021, which adds upt to 18 months. The model will
    train on the first 13 months of this training example and predict the last 5 months. If these values seem arbitrary, they
    are based on one of the winning solutions on Kaggle. We can, in fact, change these values as we see fit. The `num_predictions`
    and `series_len` are set to 5 and 18 in the configuration yaml file and can be adjusted if so desired. Finally, we also return
    the original data before time series creation for prediction purposes; we take the last (series_len - num_predictions) months
    of the original data for each county and save it to disk for prediction.

    Parameters
    ----------
    train_data : pl.DataFrame
        The training data to be adjusted.
    cfips : Dict[str, List[int]]
        A dictionary mapping 'large_counties' and 'small_counties' to lists of cfips.
    num_predictions : int, optional
        The number of months to predict, by default 5.
    series_len : int, optional
        The length of each time series for training, by default 18.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        A tuple of numpy arrays representing the training data, the target, and the original data before time series creation.
    """
    num_counties = len(cfips["large_counties"])
    # Training window is computed to ensure that (num_predictions) + (num_training_months) = (series_len)
    # The formula for (num_training_months) = (training_window) − (series_len) − (num_predictions − 1)
    # Substituting second equation into the first, (num_predictions) + [(training_window) − (series_len) − (num_predictions − 1)] = (series_len)
    # Solving for (training_window), (num_predictions) + (training_window) − (series_len) − (num_predictions) + 1 = (series_len)
    # Then, (training_window) = (series_len) - (num_predictions) + (series_len) + (num_predictions) - 1
    # Finally, (training_window) = 2 * (series_len) - 1
    training_window = 2 * series_len - 1
    # Convert to numpy array (each row is a unique county with 41 columns representing months from Aug 2019 thru Dec 2022)
    large_counties = (
        train_data.filter(pl.col("cfips").is_in(cfips["large_counties"]))
        .sort("cfips")
        .select(pl.col("microbusiness_density"))
        .to_numpy()
        .reshape(num_counties, 41)[:, -1 * training_window :]
    )  # Keep only the last 'training_window' months

    # With `num_counties` counties and `series_len` time series per each county (rows = num_counties * series_len)
    # Since python is zero-indexed, we subtract 4 from (training_window - series_len) to leave num_predictions months
    X_train = np.zeros(
        (
            num_counties * series_len,
            training_window - series_len - (num_predictions - 1),
        )
    )
    # The number of months to predict (columns = num_predictions)
    y_train = np.zeros((num_counties * series_len, num_predictions))

    # Iterate over each unique large counties
    for j in range(num_counties):
        # For each unique large county, create `series_len` time series each of length `series_len`
        for k in range(series_len):
            # This indexes the number of training examples, which should total to `num_counties * series_len`
            # The index j = 0, 1, ..., num_counties - 1 increments the counties
            # The index k = 0, 1, ..., series_len - 1 increments the time series within each county index j

            # ------------------------------- First example ------------------------------ #

            # For county j = 0, the first training example has index i = 0 * series_len + 0 = 0
            # For county j = 0, the second training example has index i = 0 * series_len + 1 = 1
            # ...
            # For county j = 0, the `series_len` training example has index i = 0 * series_len + (series_len - 1) = series_len - 1

            # ------------------------------ Second example ------------------------------ #

            # For county j = 10, the first training example has index i = 10 * series_len + 0 = 10
            # For county j = 10, the second training example has index i = 10 * series_len + 1 = 11
            # ...
            # For county j = 10, the `series_len` training example has index i = 10 * series_len + (series_len - 1)

            # ------------------------------- Final example ------------------------------ #

            # For county j = num_counties - 1, the first training example has index i = (num_counties - 1) * series_len + 0 = num_counties * series_len - series_len
            # For county j = num_counties - 1, the second training example has index i = (num_counties - 1) * series_len + 1 = num_counties * series_len - series_len + 1
            # ...
            # For county j = num_counties - 1, the `series_len` training example has index i = (num_counties - 1) * series_len + (series_len - 1) = num_counties * series_len - 1
            i = j * series_len + k

            # Slide the window of length `series_len - num_predictions` over the 35 months of data for each county j
            X_train[i,] = large_counties[
                j, k : k + (training_window - series_len - (num_predictions - 1))
            ]
            # The target should be the next 5 months after the window of length `series_len - num_predictions` above
            y_train[i,] = large_counties[
                j,
                k + (training_window - series_len - (num_predictions - 1)) : k
                + (training_window - series_len + 1),
            ]

    return X_train, y_train, large_counties[:, -(series_len - num_predictions) :]


# ----------------------------- Convert to ratios ---------------------------- #


def compute_ratios(
    X: np.ndarray, y: np.ndarray, num_predictions: int = 5, series_len: int = 18
) -> Tuple[np.ndarray, np.ndarray]:
    """
    This function converts the training data to ratios. The training data is a numpy array of shape
    `(num_counties * series_len, training_window - series_len - (num_predictions - 1))`. The target is
    a numpy array of shape `(num_counties * series_len, num_predictions)`. The function converts (raw
    microbusiness densities) to ratios by dividing each month's microbusiness density by the previous
    month's microbusiness density. The goal is to train the model on the month-to-month growth ratio
    instead of the raw densities. Because the ratios are computed month-to-month, we lose one month of
    data for each county in the training data. For example, if we have `x` months of microbusiness density
    data for a county, then we will have `x - 1` months of ratios after the conversion. On the other hand,
    we will still have `x` months of target data since the first ratio is computed by dividing the first
    month of the target density by the last month of the training density. For example, if we have 18 months
    of microbusiness density data for a county (13 months of training data and 5 months of target data),
    then we will have 12 months of ratios for the training data and 5 months of ratios for the target data:
    `(first month of target / last month of training data), (second month of target / first month of target), ...,
    (last month of target / second to last month of target)`.


    Parameters
    ----------
    X : np.ndarray
        The training data.
    y : np.ndarray
        The target.
    training_window : int, optional
        Number of months to include in the training set for each county, by default 35.
    num_predictions : int, optional
        The number of months to predict, by default 5.
    series_len : int, optional
        The length of each time series for training, by default 18.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A tuple of numpy arrays representing the training data and the target.
    """
    training_window = 2 * series_len - 1
    X_out = X.copy()
    y_out = y.copy()

    for k in range(training_window - series_len - num_predictions):
        # Take column k + 1 and divide by column k and place the result in column k + 1 of the output array
        X_out[:, k + 1] = X[:, k + 1] / X[:, k]

    y_out[:, 0] = y[:, 0] / X[:, -1]
    for k in range(num_predictions - 1):
        y_out[:, k + 1] = y[:, k + 1] / y[:, k]

    # The first column of X_train_out is kept since there is no previous month to divide by
    X_out = X_out[:, 1:]

    # Add the last month of raw microbusiness density to data since we need it for prediction
    X_out = np.concatenate([X[:, -1].reshape(-1, 1), X_out], axis=1)

    return X_out, y_out


# ------------------------------ Main function ------------------------------- #


def main():
    # ---------------------------------- Set up ---------------------------------- #

    core.global_hydra.GlobalHydra.instance().clear()
    initialize(version_base="1.2", config_path="config", job_name="processing_job")
    config = OmegaConf.to_container(compose(config_name="main"), resolve=True)

    logger = logging.getLogger(__name__)
    logger.addHandler(logging.StreamHandler(sys.stdout))
    logger.setLevel(logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test_mode",
        action="store_true",
        help="Do not save the data to disk for uploading if in test mode",
    )
    args, _ = parser.parse_known_args()

    # ------------------------------- Preprocessing ------------------------------ #

    logger.info("Ingesting training data...")
    train_data = pl.read_csv(os.path.join(config["processing_job_input"], "train.csv"))

    logger.info("Preprocessing training data...")
    train_data = preprocess(train_data)

    logger.info("Adjusting microbusiness density with new census data...")
    train_data = density_adjustment(
        train_data, os.path.join(config["processing_job_input"], "census-data")
    )

    logger.info("Splitting counties into large and small groups...")
    large_small_counties = split_counties(train_data)

    logger.info("Converting training data to numpy arrays...")
    X_train_densities, y_train_densities, large_counties = convert_to_numpy(
        train_data,
        large_small_counties,
        num_predictions=config["gru"]["num_predictions"],
        series_len=config["gru"]["series_len"],
    )

    logger.info("Converting training data to ratios...")
    X_train, y_train = compute_ratios(
        X_train_densities,
        y_train_densities,
        num_predictions=config["gru"]["num_predictions"],
        series_len=config["gru"]["series_len"],
    )

    logger.info("Saving training data to output directory...")
    train_data = pl.DataFrame(
        data=np.concatenate([X_train, y_train], axis=1),
        schema=["baseline_raw_density"]
        + [f"x{i}" for i in range(X_train.shape[1] - 1)]
        + [f"y{i}" for i in range(y_train.shape[1])],
    )
    train_data = train_data.with_columns(
        pl.Series(
            name="cfips",
            values=np.repeat(
                sorted(large_small_counties["large_counties"]),
                config["gru"]["series_len"],
            ),
        )
    )

    num_of_months = config["gru"]["series_len"] - config["gru"]["num_predictions"]
    logger.info(
        f"Saving last {num_of_months} months of raw densities to disk for predictions..."
    )

    # Only save if not in test mode
    if not args.test_mode:
        train_data.write_csv(os.path.join(config["processing_job_output"], "train.csv"))

        np.save(
            os.path.join(
                config["processing_job_output"], "densities_train_forecast.npy"
            ),
            large_counties,
        )

        np.save(
            os.path.join(config["processing_job_output"], "densities_target.npy"),
            y_train_densities,
        )

    logger.info("Preprocessing completed successfully!")

    return 0


if __name__ == "__main__":
    main()
