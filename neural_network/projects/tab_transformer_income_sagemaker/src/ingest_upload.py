import sys

import pandas as pd 
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

from hydra import compose, initialize, core
from omegaconf import OmegaConf

from custom_utils import get_logger

if __name__ == '__main__':

    # ---------------------------------- Set up ---------------------------------- #

    logger = get_logger('data_ingest')
    
    core.global_hydra.GlobalHydra.instance().clear()
    initialize(version_base='1.2', config_path='config', job_name='data_ingest')
    config = OmegaConf.to_container(compose(config_name='main'), resolve=True)

    # ---------------------------------- Load data ---------------------------------- #

    logger.info('Loading data...')

    train = pd.read_csv(config['train_data_url'], names=config['csv_header'])
    test = pd.read_csv(config['test_data_url'], names=config['csv_header'])

    # --------------------------- Ad-hoc data cleaning --------------------------- #

    logger.info('Remove first row of test data since it is invalid...')
    test.dropna(axis=0, inplace=True)

    logger.info('Remove training dots in the target column of test set...')
    test[config['target']] = test[config['target']].apply(lambda x: x.replace('.', ''))

    # ----------------------------- Validation split ----------------------------- #

    logger.info('Splitting train data into train and validation sets...')
    sss = StratifiedShuffleSplit(n_splits=1, test_size=config['validation_size'], random_state=config['random_seed'])
    for train_indices, val_indices in sss.split(train, train[config['target']]):
        train_data = train.iloc[train_indices, :]
        val_data = train.iloc[val_indices, :]

    logger.info(f'Train data shape: {train_data.shape}')
    logger.info(f'Validation data shape: {val_data.shape}')
    logger.info(f'Test data shape: {test.shape}')

    train_target_dist = dict(train_data[config['target']].value_counts() / train_data.shape[0])
    val_target_dist = dict(val_data[config['target']].value_counts() / val_data.shape[0])
    test_target_dist = dict(test[config['target']].value_counts() / test.shape[0])

    logger.info(f'Train target distribution: {train_target_dist}')
    logger.info(f'Validation target distribution: {val_target_dist}')
    logger.info(f'Test target distribution: {test_target_dist}')
    
    # ------------------------------- Upload to s3 ------------------------------- #

    logger.info('Uploading data to s3...')

    s3_bucket = config['s3_bucket']
    s3_key = config['s3_key']
    s3_base_path = f's3://{s3_bucket}/{s3_key}/input-data'

    train_data.to_csv(
        f'{s3_base_path}/train/train.csv',
        index=config['index'],
        header=config['header']
    )
    val_data.to_csv(
        f'{s3_base_path}/val/val.csv',
        index=config['index'],
        header=config['header']
    )
    test.to_csv(
        f'{s3_base_path}/test/test.csv',
        index=config['index'],
        header=config['header']
    )

    logger.info('Successfully uploaded data to s3!')

    sys.exit(0)