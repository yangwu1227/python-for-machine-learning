import csv
import logging
import multiprocessing.pool as mpp
import sys
from typing import List, TextIO, Tuple

import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger.setLevel(logging.INFO)


def generate_random_pairs(num_pairs: int) -> List[Tuple[float, float]]:
    """
    Generate a list of `num_pairs` random long/lat pairs.

    Parameters
    ----------
    num_pairs : int
        The number of long/lat pairs to generate.

    Returns
    -------
    List[Tuple[float, float]]
        A list of (longitude, latitude) tuples.

    Examples
    --------
    >>> pairs = generate_random_pairs(5)
    """
    # Bounding box of the United States.
    us_bounds = {
        "min_lon": -124.848974,
        "max_lon": -66.885444,
        "min_lat": 24.396308,
        "max_lat": 49.384358,
    }

    # Generate random longitudes and latitudes within the bounding box of the United States.
    longitudes = np.random.uniform(
        us_bounds["min_lon"], us_bounds["max_lon"], size=num_pairs
    )
    latitudes = np.random.uniform(
        us_bounds["min_lat"], us_bounds["max_lat"], size=num_pairs
    )

    # Combine longitudes and latitudes into a list of tuples of (longitude, latitude)
    pairs = list(zip(longitudes, latitudes))

    logger.info(f"Generated {num_pairs} random long/lat pairs")

    return pairs


def write_batch_to_csv(file: TextIO, pairs: List[Tuple[float, float]]) -> None:
    """
    Write a batch of long/lat pairs to a CSV file.

    Parameters
    ----------
    file : TextIO
        A file-like object open for writing in text mode.
    pairs : List[Tuple[float, float]]
        A list of long/lat pairs represented as tuples of floats.
    """
    writer = csv.writer(file)
    # Headers are assumed to be written once before calling this function
    writer.writerows(pairs)


def write_pairs_to_file_threaded(
    file_path: str,
    pairs_per_batch: int,
    num_threads: int,
    num_pairs: int,
    logger: logging.Logger,
) -> None:
    """
    Generate `num_pairs` random long/lat pairs and write them to a CSV file in batches using multiple threads.

    Parameters
    ----------
    file_path : str
        The path to the output CSV file.
    pairs_per_batch : int
        The number of long/lat pairs to write to each batch.
    num_threads : int
        The number of threads to use for writing the CSV file.
    num_pairs : int
        The total number of long/lat pairs to generate and write to the file.
    logger : logging.Logger
        A logger object to log messages to.
    """
    total_batches = num_pairs // pairs_per_batch
    if num_pairs % pairs_per_batch != 0:
        # Add one more batch to account for any remainder
        total_batches += 1
    with open(file_path, mode="w", newline="") as file:
        # Write the headers once
        writer = csv.writer(file)
        writer.writerow(["longitude", "latitude"])
        # Use a thread pool to manage a fixed number of threads
        with mpp.ThreadPool(num_threads) as pool:
            logger.info(
                f"Starting generation of {num_pairs} random long/lat pairs in {total_batches} batches"
            )
            # Generate the long/lat pairs in parallel using the thread pool
            batches: List[List[Tuple[float, float]]] = pool.map(
                generate_random_pairs,
                [pairs_per_batch] * total_batches,
            )
            logger.info(f"Finished generation of {num_pairs} random long/lat pairs")

        for batch_num, batch in enumerate(batches, start=1):
            logger.info(
                f"Writing batch {batch_num}/{total_batches} with {len(batch)} pairs to file"
            )
            write_batch_to_csv(file, batch)

        logger.info(f"Wrote {num_pairs} long/lat pairs to file")


def main() -> int:
    write_pairs_to_file_threaded(
        "random_long_lat.csv",
        pairs_per_batch=100_000,
        num_threads=8,
        num_pairs=1_000_000,
        logger=logger,
    )
    return 0


if __name__ == "__main__":
    main()
