import json
import os
import shutil
import subprocess
from typing import Any, Dict, cast

import s3fs

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # NoPep8
import tensorflow as tf
from hydra import compose, core, initialize
from model_utils import setup_logger
from omegaconf import OmegaConf


def main() -> int:
    # ---------------------------------- Set up ---------------------------------- #

    core.global_hydra.GlobalHydra.instance().clear()
    initialize(version_base="1.2", config_path="config", job_name="ingest_data")
    config: Dict[str, Any] = cast(
        Dict[str, Any],
        OmegaConf.to_container(compose(config_name="main"), resolve=True),
    )

    logger = setup_logger("ingest_data")

    # --------------------------- Download zip from s3 --------------------------- #

    logger.info("Downloading zip from s3...")

    # Create a folder in the parent directory of the directory of this python script to store the raw data
    raw_data_dir = os.path.join(
        os.path.dirname(  # Get the parent directory of the current directory
            os.path.dirname(  # Get the parent directory of the current script
                os.path.abspath(__file__)  # Get the absolute path of the current script
            )
        ),
        "data",
    )
    if not os.path.exists(raw_data_dir):
        os.makedirs(raw_data_dir)

    # Download the zip file from s3
    subprocess.run(
        f"aws s3 cp s3://{config['s3_bucket']}/{config['s3_key']}/raw-data/data.zip {raw_data_dir}/data.zip",
        shell=True,
    )
    # Unzip the file
    subprocess.run(f"unzip -q {raw_data_dir}/data.zip -d {raw_data_dir}", shell=True)

    # --------------------------- Upload data to s3 --------------------------- #

    logger.info("Load, compute class_weights, and upload to s3...")

    fs = s3fs.S3FileSystem()
    for dir_key in ["train", "val", "test"]:
        local_dir_key = os.path.join(raw_data_dir, dir_key)
        s3_dir_key = (
            f"s3://{config['s3_bucket']}/{config['s3_key']}/input-data/{dir_key}"
        )

        # Read in and save dataset to s3
        dataset = tf.keras.utils.image_dataset_from_directory(
            directory=local_dir_key,
            labels="inferred",
            label_mode="categorical",
            batch_size=None,
            image_size=(config["image_size"], config["image_size"]),
            shuffle=True,
            seed=config["random_seed"],
            interpolation="bicubic",
        )
        dataset.save(s3_dir_key)

        # Compute class percentages
        total_counts = 0
        class_counts = {}
        for class_name in os.listdir(local_dir_key):
            class_dir = os.path.join(local_dir_key, class_name)
            class_counts[class_name] = len(os.listdir(class_dir))
            total_counts += class_counts[class_name]

        # Log class distributions
        class_dist = {
            class_name: round((class_counts[class_name] / total_counts) * 100, 6)
            for class_name in class_counts
        }
        formatted_class_dist = " | ".join(
            f"{key}: {value:.2f} %" for key, value in class_dist.items()
        )
        logger.info(
            f"Class distribution for {dir_key} with total count of {total_counts}: {formatted_class_dist}"
        )

        # Generate class weights
        class_weights = {
            class_name: total_counts / class_counts[class_name]
            for class_name in class_counts
        }
        # Sort by class name and convert class names to indices
        class_weights_sorted = {
            i: class_weights[class_name]
            for i, class_name in enumerate(sorted(class_weights))
        }
        with fs.open(f"{s3_dir_key}_weights.json", "w") as f:
            json.dump(class_weights_sorted, f)

    logger.info("Finished uploading data and weights to s3...")

    # --------------------------------- Clean-up --------------------------------- #

    shutil.rmtree(raw_data_dir)

    logger.info("Finished cleaning up...")

    return 0


if __name__ == "__main__":
    main()
