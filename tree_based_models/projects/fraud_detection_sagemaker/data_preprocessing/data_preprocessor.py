import argparse
import logging
import os
from itertools import combinations
from typing import Tuple, Dict

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

# ---------------------------------------------------------------------------- #
#                   Function to parse command line arguments                   #
# ---------------------------------------------------------------------------- #


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns
    -------
    argparse.Namespace
        ArgumentParser parses arguments through the parse_args() method. This will inspect the command line, convert each argument to
        the appropriate type and then invoke the appropriate action. In most cases, this means a simple Namespace object will be built
        up from attributes parsed out of the command line.

        E.g. parser.parse_args(['--sum', '7', '-1', '42']) ---> Namespace(accumulate=<built-in function sum>, integers=[7, -1, 42])
    """
    # Instantiate an object for parsing command line strings into Python objects
    parser = argparse.ArgumentParser()

    # The add_argument() method attaches individual argument specifications to the parser
    # The 'help' the help message for an argument
    # The 'default' specifies the default value for an argument when an argument is not provided
    parser.add_argument('--data-dir', type=str,
                        default='/opt/ml/processing/input')
    parser.add_argument('--output-dir', type=str,
                        default='/opt/ml/processing/output')
    parser.add_argument('--transactions', type=str,
                        default='transaction.csv', help='name of file with transactions')
    parser.add_argument('--identity', type=str, default='identity.csv',
                        help='name of file with identity info')
    parser.add_argument('--cat-cols-xgboost', type=str, default='',
                        help='comma separated categorical cols that can be used as features for xgboost in transactions')
    # Default value of 0.7 means that 70% of the data will be used for training, 20% for hpo, and 10% for testing
    parser.add_argument('--train-data-ratio', type=float, default=0.7,
                        help='fraction of data to use in training set')
    parser.add_argument('--valid-data-ratio', type=float, default=0.2,
                        help='fraction of data to use in validation set')
    return parser.parse_args()

# ---------------------------------------------------------------------------- #
#                              Function to logging                             #
# ---------------------------------------------------------------------------- #


def get_logger(name: str) -> logging.Logger:
    """
    Parameters
    ----------
    name : str
        A string that specifies the name of the logger.

    Returns
    -------
    logging.Logger
        A logger with the specified name.
    """
    logger = logging.getLogger(name)  # Return a logger with the specified name
    log_format = '%(asctime)s %(levelname)s %(name)s: %(message)s'
    # The logging.basicConfig() function does basic configuration for the logging system
    # The 'format' argument is a format string for the log message
    # The 'level' argument is the root logger level, telling how important a given log message is; in this project, we set it to INFO
    # The 'info' level is used for reporting events that occur during 'normal' operation of a program
    logging.basicConfig(format=log_format, level=logging.INFO)
    logger.setLevel(logging.INFO)
    return logger
# ---------------------------------------------------------------------------- #
#                    Function to load, split, and save data                    #
# ---------------------------------------------------------------------------- #


def load_data(data_dir: str, transaction_data: str, identity_data: str, train_data_ratio: str, valid_data_ratio: str, output_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Function to load data from the data directory.

    Parameters
    ----------
    data_dir : str
        Path to the data directory.
    transaction_data : str
        Path to the transaction data.
    identity_data : str
        Path to the identity data.
    train_data_ratio : str

    valid_data_ratio : str
        _description_
    output_dir : str
        _description_

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]
        Returns a tuple of dataframes and numpy arrays--- transaction_df, identity_df, valid_ids, test_ids.
    """
    # --------------------------------- Read data -------------------------------- #

    # Read in transaction and identity data
    transaction_df = pd.read_csv(os.path.join(data_dir, transaction_data))
    logging.info(f"Shape of transaction data is {transaction_df.shape}")
    tagged_transactions = len(transaction_df) - \
        transaction_df.isFraud.isnull().sum()
    logging.info(f"# Tagged transactions: {tagged_transactions}")

    identity_df = pd.read_csv(os.path.join(data_dir, identity_data))
    logging.info(f"Shape of identity data is {identity_df.shape}")

    # ----------------------- Train, validation, test split ---------------------- #

    # Extract out transactions for train, validation, and test data
    logging.info(
        f"Training, validation, and test data fraction are {train_data_ratio}, {valid_data_ratio}, and {1-train_data_ratio-valid_data_ratio}, respectively")

    # Ensure that the sum of training and validation ratio is less than 1
    assert train_data_ratio + \
        valid_data_ratio < 1, "The sum of training and validation ratio is found more than or equal to 1."

    # Stratified splitter to ensure that the ratio of fraud and non-fraud transactions are the same in train, validation, and test data
    stratified_splitter_train_test = StratifiedShuffleSplit(
        n_splits=1, test_size=1-train_data_ratio-valid_data_ratio, train_size=train_data_ratio, random_state=42)

    for train_indices, test_indices in stratified_splitter_train_test.split(transaction_df, transaction_df.isFraud):

        # The remaining data (20 %) is used for validation
        valid_indices = np.setdiff1d(
            transaction_df.index.to_numpy(),
            np.concatenate((train_indices, test_indices), axis=None)
        )

    # Ensure that the train, validation, and test data are disjoint
    assert np.array_equal(np.sort(np.concatenate((train_indices, valid_indices, test_indices), axis=None)),
                          transaction_df.index.to_numpy()), "The train, validation, and test data are not disjoint."
    # Obtain the transaction ids for train, validation, and test data
    train_ids = transaction_df.TransactionID.values[train_indices]
    valid_ids = transaction_df.TransactionID.values[valid_indices]
    test_ids = transaction_df.TransactionID.values[test_indices]

    # Ensure that the ratio of fraud and non-fraud transactions are the same in train, validation, and test data
    def get_fraud_frac(series): return 100 * sum(series) / len(series)

    logging.info("Percentage of fraud transactions for train data: {}".format(
        get_fraud_frac(transaction_df.loc[transaction_df.TransactionID.isin(train_ids), 'isFraud'])))

    logging.info("Percentage of fraud transactions for validation data: {}".format(
        get_fraud_frac(transaction_df.loc[transaction_df.TransactionID.isin(valid_ids), 'isFraud'])))

    logging.info("Percentage of fraud transactions for test data: {}".format(
        get_fraud_frac(transaction_df.loc[transaction_df.TransactionID.isin(test_ids), 'isFraud'])))

    logging.info("Percentage of fraud transactions for all data: {}".format(
        get_fraud_frac(transaction_df.isFraud)))

    # -------------- Write train, validation, and test data to file -------------- #

    with open(os.path.join(output_dir, 'validation.csv'), 'w') as f:
        f.writelines(map(lambda x: str(x) + "\n", valid_ids))

    logging.info("Wrote validation data to file: {}".format(
        os.path.join(output_dir, 'validation.csv')))

    with open(os.path.join(output_dir, 'test.csv'), 'w') as f:
        f.writelines(map(lambda x: str(x) + "\n", test_ids))

    logging.info("Wrote test data to file: {}".format(
        os.path.join(output_dir, 'test.csv')))

    return transaction_df, identity_df, valid_ids, test_ids

# ---------------------------------------------------------------------------- #
#                          Function to preprocess data                         #
# ---------------------------------------------------------------------------- #


def get_features_and_labels(transactions_df: pd.DataFrame, cat_cols_xgboost: str, output_dir: str) -> None:
    """
    Function to get features and labels from the transaction data.

    Parameters
    ----------
    transactions_df : pd.DataFrame
    cat_cols_xgboost : str
        Categorical columns for XGBoost.
    output_dir : str
        Output directory for feature DataFrame and label DataFrame containing 'TransactionID' and 'isFraud' columns.
    """

    # --------------------------- Features for XGBoost --------------------------- #

    logging.info("Processing feature columns for XGBoost.")
    cat_cols_xgb = cat_cols_xgboost.split(",")
    logging.info(
        "Categorical feature columns for XGBoost: {}".format(cat_cols_xgb))
    logging.info("Numerical feature column for XGBoost: 'TransactionAmt'")

    # Feature matrix for XGBoost
    features_xgb = pd.get_dummies(
        transactions_df[['TransactionID'] + cat_cols_xgb], columns=cat_cols_xgb).fillna(0)
    features_xgb['TransactionAmt'] = transactions_df['TransactionAmt'].apply(np.log10)
    # Write xgboost feature matrix to disk
    features_xgb.to_csv(os.path.join(
        output_dir, 'features_xgboost.csv'), index=False, header=False)
    logging.info("Wrote features to file: {}".format(
        os.path.join(output_dir, 'features_xgboost.csv')))

    # ---------------------------------- Labels ---------------------------------- #

    transactions_df[['TransactionID', 'isFraud']].to_csv(
        os.path.join(output_dir, 'tags.csv'), index=False)
    logging.info("Wrote labels to file: {}".format(
        os.path.join(output_dir, 'tags.csv')))


if __name__ == '__main__':

    logging = get_logger(__name__)

    args = parse_args()

    transactions, identity, _, _ = load_data(args.data_dir,
                                             args.transactions,
                                             args.identity,
                                             args.train_data_ratio,
                                             args.valid_data_ratio,
                                             args.output_dir)

    # Preprocessing
    get_features_and_labels(transactions, args.cat_cols_xgboost, args.output_dir)
