import json
import os
import subprocess
from random import Random
from shutil import copyfile

import sagemaker
from model_utils import setup_logger


def main() -> int:
    random_seed = 1227
    s3_bucket = "yang-ml-sagemaker"
    s3_key = "lesion-segmentation"

    logger = setup_logger(__name__)

    # ------------------------- Download zip file from s3 ------------------------ #

    logger.info("Downloading zip file from s3...")

    # Create a folder in the parent directory of the directory of this python script to store the raw data
    raw_data_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"
    )
    if not os.path.exists(raw_data_dir):
        os.makedirs(raw_data_dir)

    subprocess.run(
        f"aws s3 cp s3://{s3_bucket}/{s3_key}/raw-data/data.zip {raw_data_dir}/data.zip",
        shell=True,
    )
    subprocess.run(f"unzip -q {raw_data_dir}/data.zip -d {raw_data_dir}", shell=True)

    # ------ Set up the directories for the train, validation, and test sets ----- #

    logger.info("Setting up directories...")

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
        Random(random_seed).shuffle(image_mask_pairs)
    # Separate the pairs back into separate lists
    image_paths, mask_paths = map(list, zip(*image_mask_pairs))

    dir_dict = {}
    dir_name_sets = [
        "train",
        "train_annotation",
        "validation",
        "validation_annotation",
        "test",
        "test_annotation",
    ]
    for set_name in dir_name_sets:
        dir_name = os.path.join(raw_data_dir, set_name)
        os.makedirs(dir_name, exist_ok=True)
        dir_dict[set_name] = dir_name
        logger.info(f"Successully create directory: {dir_name}")

    # Combine the image paths and mask paths into pairs
    image_mask_pairs = list(zip(image_paths, mask_paths))
    # Shuffle pairs in place a few times
    for i in range(5):
        Random(random_seed).shuffle(image_mask_pairs)
    # Separate the pairs back into separate lists
    image_paths, mask_paths = map(list, zip(*image_mask_pairs))

    # Compute number of files for each set
    num_files = len(image_paths)
    num_train = int(num_files * 0.7)  # 70% for training
    num_val = int(num_files * 0.20)  # 15% for validation
    num_test = num_files - num_train - num_val  # Remaining for testing

    # --------------- Copy the files to the appropriate directories -------------- #

    logger.info("Copying images and masks to appropriate directories...")

    for i, (image_path, mask_path) in enumerate(zip(image_paths, mask_paths)):
        # The first batch of files will be used as train
        if i < num_train:
            image_dir = dir_dict["train"]
            mask_dir = dir_dict["train_annotation"]
        # The next batch will be used as val
        elif i < num_train + num_val:
            image_dir = dir_dict["validation"]
            mask_dir = dir_dict["validation_annotation"]
        # The remaining will be test
        else:
            image_dir = dir_dict["test"]
            mask_dir = dir_dict["test_annotation"]

        dest_image = os.path.join(image_dir, os.path.basename(image_path))
        dest_mask = os.path.join(mask_dir, os.path.basename(mask_path))
        copyfile(image_path, dest_image)
        copyfile(mask_path, dest_mask)

    label_map = {"scale": 1}
    label_file_names = [
        os.path.join(raw_data_dir, path)
        for path in ["train_label_map.json", "validation_label_map.json"]
    ]

    for file_name in label_file_names:
        with open(file_name, "w") as file:
            json.dump(label_map, file)

    # ------------------------------- Upload to s3 ------------------------------- #

    logger.info("Uploading data to s3...")

    sm_session = sagemaker.Session(default_bucket=s3_bucket)
    s3_uploader = sagemaker.s3.S3Uploader()

    for file_name in label_file_names:
        upload_uri = s3_uploader.upload(
            local_path=file_name,
            desired_s3_uri=f"s3://{s3_bucket}/{s3_key}/input-data/label_map",
            sagemaker_session=sm_session,
        )

    for key in dir_name_sets:
        upload_uri = s3_uploader.upload(
            local_path=dir_dict[key],
            desired_s3_uri=f"s3://{s3_bucket}/{s3_key}/input-data/{key}",
            sagemaker_session=sm_session,
        )

    logger.info("Finished uploading data to s3!")

    # --------------------------------- Clean-up --------------------------------- #

    subprocess.run(f"rm -rf {raw_data_dir}", shell=True)

    del sm_session, s3_uploader

    return 0


if __name__ == "__main__":
    main()
