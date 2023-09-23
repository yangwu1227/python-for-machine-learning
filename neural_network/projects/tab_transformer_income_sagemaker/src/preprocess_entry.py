import sys
import os
import logging
import argparse

import polars as pl
from sklearn.model_selection import StratifiedShuffleSplit

from hydra import compose, initialize, core
from omegaconf import OmegaConf

def main():

    # ---------------------------------- Set up ---------------------------------- #

    logger = logging.getLogger(__name__)
    logger.addHandler(logging.StreamHandler(sys.stdout))
    logger.setLevel(logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument('--test_mode', action='store_true', help='Do not save the data to disk for uploading if in test mode')
    args, _ = parser.parse_known_args()
    
    core.global_hydra.GlobalHydra.instance().clear()
    initialize(version_base='1.2', config_path='config', job_name='preprocess')
    config = OmegaConf.to_container(compose(config_name='main'), resolve=True)

    # ---------------------------------- Load data ---------------------------------- #

    logger.info('Loading data...')

    train = pl.read_csv(config['train_data_url'], new_columns=config['csv_header'])
    # First row of the test data is invalid, so we skip it
    test = pl.read_csv(config['test_data_url'], new_columns=config['csv_header'], skip_rows=1)

    # --------------------------- Ad-hoc data cleaning --------------------------- #

    logger.info('Remove trailing dots in the target column of the test set...')
    test = test.with_columns(pl.col(config['target']).str.replace_all('\.', ''))

    # ----------------------------- Validation split ----------------------------- #

    logger.info('Splitting train data into train and validation sets...')
    sss = StratifiedShuffleSplit(n_splits=1, test_size=config['validation_size'], random_state=config['random_seed'])
    for train_indices, val_indices in sss.split(train, train[config['target']]):
        train_data = train[train_indices, :]
        val_data = train[val_indices, :]

    logger.info(f'Train data shape: {train_data.shape}')
    logger.info(f'Validation data shape: {val_data.shape}')
    logger.info(f'Test data shape: {test.shape}')

    # ------------------------------ Target distribution ------------------------------ #

    names = ['train', 'val', 'test']
    datasets = [train_data, val_data, test]

    for name, data in zip(names, datasets):
        # Princt class distribution
        class_dist = dict(data[config['target']].value_counts() \
                                                .sort(by=pl.col('counts')) \
                                                .select(
                                                    pl.col(config['target']),
                                                    (pl.col('counts') / pl.col('counts').sum() * 100).round(4)
                                                ) \
                                                .iter_rows())
        logger.info(f'{name.capitalize()} target distribution: {class_dist}')

        # Only save if not in test mode
        if not args.test_mode:
            data.write_csv(os.path.join(config['processing_job_output'], f'{name}/{name}.csv'), has_header=config['header'])
        
    logger.info('Preprocessing completed successfully!')

    sys.exit(0)

if __name__ == '__main__':

    main()