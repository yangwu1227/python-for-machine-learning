import os 
import sys
import re
import subprocess
import shutil

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import sagemaker
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

from hydra import compose, initialize
from custom_utils import get_logger

if __name__ == '__main__':

    # --------------------------- Compose configuration -------------------------- #

    initialize(version_base='1.2', config_path='config', job_name='ingest_data')
    config = compose(config_name='main')

    logger = get_logger(__name__)

    # --------------------------- Download zip from s3 --------------------------- #

    logger.info('Downloading zip from s3...')

    # Create a folder in the parent directory of the directory of this python script to store the raw data
    raw_data_dir = os.path.join(
        os.path.dirname(  # Get the parent directory of the current directory
            os.path.dirname(  # Get the parent directory of the current script
                os.path.abspath(__file__)  # Get the absolute path of the current script
            )
        ),
        'data' 
    )
    if not os.path.exists(raw_data_dir):
        os.makedirs(raw_data_dir)

    file_names = ['train.zip', 'validation.zip']
    for file_name in file_names:
        # Download the zip file from s3
        subprocess.run(
            f'aws s3 cp s3://{config.s3_bucket}/{config.s3_key}/raw-data/{file_name} {raw_data_dir}/{file_name}',
            shell=True
        )
        # Unzip the file
        subprocess.run(
            f'unzip -q {raw_data_dir}/{file_name} -d {raw_data_dir}',
            shell=True
        )

    # ------------------------------ Train and test ------------------------------ #

    logger.info('Loading train and test data...')

    train_ds = tf.keras.utils.image_dataset_from_directory(
        directory=os.path.join(raw_data_dir, 'train'),
        labels='inferred',
        label_mode='int',
        batch_size=None,
        image_size=tuple(config.image_size),
        shuffle=True,
        seed=config.random_seed,
        interpolation='bilinear'
    )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        directory=os.path.join(raw_data_dir, 'validation'),
        labels='inferred',
        label_mode='int',
        batch_size=None,
        image_size=tuple(config.image_size),
        shuffle=True,
        seed=config.random_seed,
        interpolation='bilinear'
    )

    # Get number of rows in the shuffle dataset
    logger.info(f'Number of examples in train_ds: {len(train_ds)}')
    logger.info(f'Number of examples in test_ds: {len(test_ds)}')

    # ----------------------------- Convert to numpy ----------------------------- #

    # Initialize empty lists to store the data
    X_train, y_train = [], []
    X_val, y_val = [], []
    X_test, y_test = [], []

    # Iterate over the batches
    for X, y in train_ds:
        X_train.append(X.numpy())
        y_train.append(y.numpy())

    for X, y in test_ds:
        X_test.append(X.numpy())
        y_test.append(y.numpy())

    # Convert the lists to numpy arrays
    X_train, y_train= np.array(X_train), np.array(y_train)
    X_test, y_test = np.array(X_test), np.array(y_test)

    # ------------------------ Train and validation split ------------------------ #

    logger.info('Splitting train and validation data...')

    # Split the train data into train and validation data
    sss = StratifiedShuffleSplit(n_splits=1, test_size=config.validation_size, random_state=config.random_seed)
    for train_index, val_index in sss.split(X_train, y_train):
        X_train, X_val = X_train[train_index], X_train[val_index]
        y_train, y_val = y_train[train_index], y_train[val_index]

    logger.info(f'Training data shape: {X_train.shape}')
    logger.info(f'Validation data shape: {X_val.shape}')
    logger.info(f'Test data shape: {X_test.shape}')

    # Format class distribution for better readability in the log
    for data, data_name in zip([y_train, y_val, y_test], ['training', 'validation', 'test']):
        class_dist = (np.unique(y_train, return_counts=True)[1] / len(y_train)) * 100
        list_of_formated_strings = [f'{label}: {percentage:.2f}%' for label, percentage in zip(np.unique(data), class_dist)]
        formated_class_dist = ' | '.join(list_of_formated_strings)
        logger.info(f'Class distribution in the {data_name} set: {formated_class_dist}')

    # ------------------------------- Upload to s3 ------------------------------- #

    logger.info('Uploading data to s3...')

    sm_session = sagemaker.Session(default_bucket=config.s3_bucket)
    s3_uploader = sagemaker.s3.S3Uploader()

    # Create directory for uploading datasets to s3
    data_save_paths = {}
    input_data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'input-data')
    for data_set in ['train', 'val', 'test']:
        data_save_paths[data_set] = f'{input_data_dir}/{data_set}'
        os.makedirs(data_save_paths[data_set])

    # Save numpy arrays to disk
    for key, list_of_arrays in zip(data_save_paths, [[X_train, y_train], [X_val, y_val], [X_test, y_test]]):
        np.save(os.path.join(data_save_paths[key], f'X_{key}.npy'), list_of_arrays[0])
        np.save(os.path.join(data_save_paths[key], f'y_{key}.npy'), list_of_arrays[1])

    for key in data_save_paths:
        s3_uploader.upload(
            local_path=data_save_paths[key],
            desired_s3_uri=f's3://{config.s3_bucket}/{config.s3_key}/input-data/{key}',
            sagemaker_session=sm_session
        )

    logger.info('Finished uploading data to s3...')

    # --------------------------------- Clean-up --------------------------------- #

    for data_dir in [raw_data_dir, input_data_dir]:
        if os.path.exists(data_dir):
            shutil.rmtree(data_dir)
    del sm_session, s3_uploader

    logger.info('Finished cleaning up...')