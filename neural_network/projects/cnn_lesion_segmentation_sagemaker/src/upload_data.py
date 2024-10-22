import argparse
import logging
import os
import random
import subprocess
import sys
from concurrent import futures
from queue import Queue
from typing import Tuple

import numpy as np
import sagemaker
from custom_utils import get_logger
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import img_to_array, load_img


def ingest_image(path: str, image_size: Tuple[int, int]) -> np.ndarray:
    """
    Ingest an image from a path and return a numpy array of the image.

    Parameters
    ----------
    path : str
        Path to the image.
    image_size : Tuple[int, int]
        Size of the image.

    Returns
    -------
    np.ndarray
        Numpy array of the image.
    """
    image = load_img(path=path, color_mode="grayscale", target_size=image_size)
    image = img_to_array(image)
    return image


def ingest_mask(path: str, image_size: Tuple[int, int]) -> np.ndarray:
    """
    Ingest a mask from a path and return a numpy array of the mask.

    Parameters
    ----------
    path : str
        Path to the mask.
    image_size : Tuple[int, int]
        Size of the mask.

    Returns
    -------
    np.ndarray
        Numpy array of the mask.
    """
    mask = load_img(path=path, color_mode="grayscale", target_size=image_size)
    mask = img_to_array(mask)
    return mask


def ingest_images_and_masks(
    path_queue: Queue,
    image_paths: list,
    mask_paths: str,
    images: np.ndarray,
    masks: np.ndarray,
    image_size: Tuple[int, int],
) -> None:
    """
    Function for concurrent ingestion of images and masks.

    Parameters
    ----------
    path_queue : Queue
        Thread-safe queue of image and mask paths.
    image_paths : list
        List of image paths in the desired order.
    mask_paths : list
        List of mask paths in the desired order.
    images : np.ndarray
        Numpy array to store the images.
    masks : np.ndarray
        Numpy array to store the masks.
    image_size : Tuple[int, int]
        Size of the images and masks.
    """
    while not path_queue.empty():
        image_path, mask_path = path_queue.get()
        image_index = image_paths.index(image_path)
        mask_index = mask_paths.index(mask_path)
        images[image_index] = ingest_image(image_path, image_size)
        masks[mask_index] = ingest_mask(mask_path, image_size)
        path_queue.task_done()


if __name__ == "__main__":
    random_seed = 1227
    image_size = (256, 256)
    num_channels = 1
    s3_bucket = "yang-ml-sagemaker"
    s3_key = "lesion-segmentation"

    logger = get_logger(__name__)

    # ------------------------- Download zip file from s3 ------------------------ #

    logger.info("Downloading zip file from s3...")

    # Create a folder in the parent directory of the directory of this python script to store the raw data
    raw_data_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"
    )
    if not os.path.exists(raw_data_dir):
        os.makedirs(raw_data_dir)

    subprocess.run(
        f"aws s3 cp s3://{s3_bucket}/{s3_key}/raw-data/data.zip {raw_data_dir}/lesion-segmentation.zip",
        shell=True,
    )
    subprocess.run(
        f"unzip -q {raw_data_dir}/lesion-segmentation.zip -d {raw_data_dir}", shell=True
    )

    # Ensure that the images paths and mask paths are sorted
    image_dir = os.path.join(raw_data_dir, "frames")
    image_paths = sorted(
        [os.path.join(image_dir, image_path) for image_path in os.listdir(image_dir)]
    )
    mask_dir = os.path.join(raw_data_dir, "masks")
    mask_paths = sorted(
        [os.path.join(mask_dir, mask_path) for mask_path in os.listdir(mask_dir)]
    )
    num_images = len(image_paths)

    # Combine the image paths and mask paths into pairs
    image_mask_pairs = list(zip(image_paths, mask_paths))
    # Shuffle pairs in place a few times
    for i in range(5):
        random.Random(random_seed).shuffle(image_mask_pairs)
    # Separate the pairs back into separate lists
    image_paths, mask_paths = zip(*image_mask_pairs)

    # -------------------- Ingest images and masks in parallel ------------------- #

    logger.info("Ingesting images and masks in parallel...")

    # Instantiate empty arrays
    images = np.zeros((num_images,) + image_size + (num_channels,), dtype="float32")
    masks = np.zeros((num_images,) + image_size + (num_channels,), dtype="float32")

    # Create a thread-safe queue to store the paths
    path_queue = Queue()

    # Enqueue image and mask paths in the desired order
    for image_path, mask_path in zip(image_paths, mask_paths):
        path_queue.put((image_path, mask_path))

    # Number of worker threads
    num_threads = min(os.cpu_count(), num_images)

    with futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        # Submit ingestion tasks for execution
        future_tasks = [
            executor.submit(
                ingest_images_and_masks,
                path_queue,
                image_paths,
                mask_paths,
                images,
                masks,
                image_size,
            )
            for _ in range(num_threads)
        ]

        # Wait for all tasks to complete
        futures.wait(future_tasks)

    # -------------------------- Train, val, test split -------------------------- #

    logger.info("Splitting data into train, val, and test sets...")

    train_images, test_images, train_masks, test_masks = train_test_split(
        images, masks, test_size=0.2, random_state=random_seed
    )

    train_images, val_images, train_masks, val_masks = train_test_split(
        train_images, train_masks, test_size=0.2, random_state=random_seed
    )

    # Fix non-binary masks
    train_masks = np.where(
        np.logical_or(train_masks == 0, train_masks == 255), train_masks, 255
    )
    val_masks = np.where(
        np.logical_or(val_masks == 0, val_masks == 255), val_masks, 255
    )
    test_masks = np.where(
        np.logical_or(test_masks == 0, test_masks == 255), test_masks, 255
    )

    # Scale masks to [0, 1] (images will be scaled after data augmentation)
    train_masks /= 255.0
    val_masks /= 255.0
    test_masks /= 255.0

    logger.info(f"Training set has shape {train_images.shape}")
    logger.info(f"Validation set has shape {val_images.shape}")
    logger.info(f"Test set has shape {test_images.shape}")
    logger.info("Saving train, val, and test sets locally...")

    data_save_paths = {
        "train": os.path.join(raw_data_dir, "train"),
        "val": os.path.join(raw_data_dir, "val"),
        "test": os.path.join(raw_data_dir, "test"),
    }

    for key, path in data_save_paths.items():
        if not os.path.exists(path):
            os.makedirs(path)

    for key, path in data_save_paths.items():
        np.save(file=os.path.join(path, f"{key}_images.npy"), arr=eval(f"{key}_images"))
        np.save(file=os.path.join(path, f"{key}_masks.npy"), arr=eval(f"{key}_masks"))

    # ------------------------------- Upload to s3 ------------------------------- #

    logger.info("Uploading data to s3...")

    sm_session = sagemaker.Session(default_bucket=s3_bucket)
    s3_uploader = sagemaker.s3.S3Uploader()

    for key in data_save_paths:
        upload_uri = s3_uploader.upload(
            local_path=data_save_paths[key],
            desired_s3_uri=f"s3://{s3_bucket}/{s3_key}/input-data/{key}",
            sagemaker_session=sm_session,
        )

    logger.info("Finished uploading data to s3!")

    # --------------------------------- Clean-up --------------------------------- #

    subprocess.run(f"rm -rf {raw_data_dir}", shell=True)

    del sm_session, s3_uploader
