import json
import os
import random
import subprocess
import sys
from multiprocessing import Pool
from typing import Dict, Tuple

import numpy as np
import s3fs
from hydra import compose, core, initialize
from omegaconf import OmegaConf

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Nopep8
import matplotlib.pyplot as plt
from tensorflow.keras.utils import image_dataset_from_directory

# ------------------------- Downsample a single class ------------------------ #


def downsample_class(args: Tuple[str, int, int]) -> None:
    """
    Downsample a single class by deleting extra samples from the class directory.

    Parameters
    ----------
    args : tuple
        A tuple containing:
        - The directory of the class (str).
        - The maximum number of samples allowed (int).
        - The random seed to use for shuffling the samples (int).

    Returns
    -------
    None
    """
    class_dir, max_samples, random_seed = args

    # Initialize a random number generator with the given seed
    rng = random.Random(random_seed)

    # List all class image file (absolute) paths
    image_files = [
        os.path.join(class_dir, image_file) for image_file in os.listdir(class_dir)
    ]

    # Shuffle the image file paths 3 times
    for _ in range(3):
        rng.shuffle(image_files)

    # Count the number of samples for this class
    class_count = len(image_files)

    # Delete the first `class_count - max_samples` files
    num_files_to_delete = class_count - max_samples
    for image_file in image_files[:num_files_to_delete]:
        os.remove(image_file)

    return None


# ---------------------- Function to downsample datasets --------------------- #


def downsample(directory: str, max_samples: int, random_seed: int) -> None:
    """
    This funtion takes a directory containing training, validatoin, or
    test data. It will first count the number of files in the subdirectories,
    which represent the number of samples for the classes. Next, for classes
    with more than `max_samples` samples, it will randomly remove files until
    the number of samples is equal to `max_samples`. The train, val, and test
    directory should be structured as follows:

    ```
    main_directory/
    ...class_a/
    ......a_image_1.jpg
    ......a_image_2.jpg
    ...class_b/
    ......b_image_1.jpg
    ......b_image_2.jpg
    ```

    Parameters
    ----------
    directory : str
        The directory containing the train, validation, and test data.
    max_sample : int
        The maximum number of samples allowed for each class.
    random_seed : int
        The random seed to use for reproducibility.

    Returns
    -------
    None
    """
    # List all class subdirectories and sort by class label
    class_dirs = dict(
        sorted(
            {
                int(int_label): os.path.join(directory, int_label)
                for int_label in os.listdir(directory)
            }.items()
        )
    )

    # Classes to delete samples from
    class_to_delete = {
        int_label: path
        for int_label, path in class_dirs.items()
        if len(os.listdir(path)) > max_samples
    }

    if len(class_to_delete) != 0:
        with Pool() as p:
            p.map(
                downsample_class,
                [
                    (class_dir, max_samples, random_seed)
                    for int_label, class_dir in class_to_delete.items()
                ],
            )


# ------------------------------- Count samples ------------------------------ #


def count_samples(directory: str) -> Dict[str, int]:
    """
    Count the number of samples for each class in a directory. Again, the directory must be structured as follows:

    ```
    main_directory/
    ...class_a/
    ......a_image_1.jpg
    ......a_image_2.jpg
    ...class_b/
    ......b_image_1.jpg
    ......b_image_2.jpg
    ```

    Parameters
    ----------
    directory : str
        The directory containing the train, validation, and test data.

    Returns
    -------
    Dict[str, int]
        A dictionary mapping class labels to the number of samples.
    """
    # List all subdirectories (classes)
    classes = os.listdir(directory)
    # Count the number of files (samples) in each subdirectory
    sample_counts = {
        cls: len(os.listdir(os.path.join(directory, cls))) for cls in classes
    }
    return sample_counts


# ------------------- Function for computing class weights ------------------- #


def calculate_class_weights(class_counts: Dict[str, int]) -> Dict[int, float]:
    """
    Calculate class weights based on class counts.

    Parameters
    ----------
    class_counts : Dict[str, int]
        A dictionary where keys are class labels and values are the number of samples per class.

    Returns
    -------
    class_weights : Dict[int, float]
        A dictionary where keys are class labels and values are the corresponding class weights.
    """

    total_samples = sum(class_counts.values())
    class_weights = {
        int(class_label): total_samples / count
        for class_label, count in class_counts.items()
    }
    sorted_class_weights = dict(sorted(class_weights.items()))

    return sorted_class_weights


# ------------------------------- Main program ------------------------------- #


def main() -> int:
    # ---------------------------------- Set up ---------------------------------- #

    # Get logger
    logger = get_logger(name="data_ingest")

    # Hyra
    core.global_hydra.GlobalHydra.instance().clear()
    initialize(version_base="1.2", config_path="config", job_name="data_ingest")
    config = OmegaConf.to_container(compose(config_name="main"), resolve=True)

    # --------------------------- Download raw data zip -------------------------- #

    logger.info("Downloading and unzipping raw data zip file from s3...")

    # Create a folder in the parent directory of this python script to store the raw data
    raw_data_dir = os.path.join(os.path.dirname(__file__), "data")
    if not os.path.exists(raw_data_dir):
        os.makedirs(raw_data_dir)

    subprocess.run(
        f'aws s3 cp s3://{config["s3_bucket"]}/{config["s3_key"]}/raw-data/data.zip {raw_data_dir}/data.zip',
        shell=True,
    )
    subprocess.run(f"unzip -q {raw_data_dir}/data.zip -d {raw_data_dir}", shell=True)

    # -------------------------------- Downsample -------------------------------- #

    logger.info("Downsampling data...")

    # Dictionary of directories to downsample
    directories = {
        directory: os.path.join(raw_data_dir, directory)
        for directory in ["train", "val", "test"]
    }

    # Downsample each directory
    for dir_key in directories:
        downsample(directories[dir_key], config["max_samples"], config["random_seed"])

    # --------------------------------- Load data -------------------------------- #

    class_percentages = {}
    total_counts = {}
    fs = s3fs.S3FileSystem()

    for dir_key in directories:
        # Load data as tensorflow dataset
        dataset = image_dataset_from_directory(
            directories[dir_key],
            labels="inferred",
            label_mode="categorical",
            shuffle=config["shuffle"],
            batch_size=1,
            image_size=(config["image_size"], config["image_size"]),
            seed=config["random_seed"],
        )

        # Count the samples
        class_counts = count_samples(directories[dir_key])
        # Store total counts for each directory (train, val, test)
        total_counts[dir_key] = sum(class_counts.values())
        percentages = {
            cls: (count / total_counts[dir_key]) * 100
            for cls, count in class_counts.items()
        }
        class_percentages[dir_key] = dict(sorted(percentages.items()))

        # Save to s3 directory
        logger.info(f"Saving {dir_key} data to s3...")

        dataset.save(
            f's3://{config["s3_bucket"]}/{config["s3_key"]}/input-data/{dir_key}'
        )
        # Also save the class percentages to s3
        with fs.open(
            f's3://{config["s3_bucket"]}/{config["s3_key"]}/input-data/{dir_key}_weights.json',
            "w",
        ) as f:
            json.dump(calculate_class_weights(class_counts), f)

    # ------------------------- Plot class distributions ------------------------- #

    logger.info("Plotting class distributions...")

    fig, ax = plt.subplots(figsize=(20, 10))
    bar_width = 0.25

    # Compute the indices for the X-axis, spread them out a bit more to avoid bars being jammed together
    indices = np.arange(len(class_percentages["train"])) * 1.5

    train_bar = ax.bar(
        indices,
        class_percentages["train"].values(),
        width=bar_width,
        label=f'Train (Total: {total_counts["train"]})',
    )
    val_bar = ax.bar(
        [i + bar_width for i in indices],
        class_percentages["val"].values(),
        width=bar_width,
        label=f'Validation (Total: {total_counts["val"]})',
    )
    test_bar = ax.bar(
        [i + 2 * bar_width for i in indices],
        class_percentages["test"].values(),
        width=bar_width,
        label=f'Test (Total: {total_counts["test"]})',
    )

    # Show class labels on the X-axis
    ax.set_xticks(indices + bar_width)
    ax.set_xticklabels(class_percentages["train"].keys(), rotation="vertical")

    ax.set_xlabel("Class")
    ax.set_ylabel("Percentage of total (%)")
    ax.set_title("Class distributions")
    ax.legend()

    plt.tight_layout()
    plt.show()

    # ------------------------------- Clean up ---------------------------------- #

    logger.info("Cleaning up by removing raw data zip file and unzipped data...")

    subprocess.run(f"rm -rf {raw_data_dir}", shell=True)

    return 0


if __name__ == "__main__":
    from custom_utils import get_logger

    sys.exit(main())
