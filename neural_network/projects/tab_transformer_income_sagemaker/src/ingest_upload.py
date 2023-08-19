import sys
import s3fs

import polars as pl
from sklearn.model_selection import StratifiedShuffleSplit

from hydra import compose, initialize, core
from omegaconf import OmegaConf

from custom_utils import get_logger

def main():

    # ---------------------------------- Set up ---------------------------------- #

    logger = get_logger('data_ingest')
    
    core.global_hydra.GlobalHydra.instance().clear()
    initialize(version_base='1.2', config_path='config', job_name='data_ingest')
    config = OmegaConf.to_container(compose(config_name='main'), resolve=True)

    # ---------------------------------- Load data ---------------------------------- #

    logger.info('Loading data...')

    train = pl.read_csv(config['train_data_url'], new_columns=config['csv_header'])
    # First row of the test data is invalid, so we skip it
    test = pl.read_csv(config['test_data_url'], new_columns=config['csv_header'], skip_rows=1)

    # --------------------------- Ad-hoc data cleaning --------------------------- #

    logger.info('Remove training dots in the target column of test set...')
    test = test.with_columns(pl.col(config['target']).str.replace_all('\.', ''))

    # logger.info('Remove all leading and trailing whitespaces in all columns...')
    # # Get names of all categorical features
    # strip_feat = list(config['tf_keras']['cat_feat_vocab'].keys())
    # strip_feat.append(config['target'])
    # train = train.with_columns(pl.col(strip_feat).str.strip())
    # test = test.with_columns(pl.col(strip_feat).str.strip())

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
    s3_base_path = f's3://{config["s3_bucket"]}/{config["s3_key"]}/input-data'
    s3_fs = s3fs.S3FileSystem(anon=False)

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

        # Upload to s3
        logger.info(f'Uploading {name} data to s3...')
        with s3_fs.open(f'{s3_base_path}/{name}/{name}.csv', 'wb') as f:
            data.write_csv(
                f,
                has_header=config['header']
            )

    logger.info('Successfully uploaded data to s3!')

    sys.exit(0)

if __name__ == '__main__':

    main()