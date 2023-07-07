import boto3
import pickle
from typing import Any, Tuple
import logging
from re import sub

import pandas as pd
import numpy as np

from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedShuffleSplit

import xgboost as xgb

# ----- Class for uploading and downloading Python objects to and from S3 ---- #

class S3Pickle:
    """
    A class for uploading and downloading Python objects to and from S3.
    """
    def __init__(self, s3_client=None):
        """
        Constructor for S3Pickle class.

        Parameters
        ----------
        s3_client : _type_, optional
            A boto3 S3 client. The default is None.
        """
        if s3_client is None: 
            self.s3_client = boto3.client('s3')
        else:
            self.s3_client = s3_client
        
    def upload_pickle(self, obj: Any, bucket_name: str, key_name: str) -> None:
        """
        Upload a Python object to S3 as a pickle byte string.

        Parameters
        ----------
        obj : Any
            A Python object.
        bucket_name : str
            S3 bucket name.
        key_name : str
            S3 key name.
        """
        # Serialize the object to a pickle byte string
        pickle_byte_string = pickle.dumps(obj)
        
        # Upload the pickle byte string to S3
        self.s3_client.put_object(Body=pickle_byte_string, Bucket=bucket_name, Key=key_name)
        
        return None
        
    def download_pickle(self, bucket_name: str, key_name: str) -> Any:
        """
        Download a Python object from S3 as a pickle byte string.
        
        Parameters
        ----------
        bucket_name : str
            S3 bucket name.
        key_name : str
            S3 key name.
        """
        # Download the pickle byte string from S3
        response = self.s3_client.get_object(Bucket=bucket_name, Key=key_name)
        pickle_byte_string = response['Body'].read()
        
        # Deserialize the pickle byte string to a Python object
        obj = pickle.loads(pickle_byte_string)
        
        return obj
    
# ----------------------------------- Data ----------------------------------- #

def load_data(data_s3_url: str, logger: logging.Logger = None) -> Tuple[pd.DataFrame, np.ndarray]:
    """
     Load data from S3 bucket and return X and y.

    Parameters
    ----------
    data_s3_url : str
        S3 url of data.
    logger : logging.Logger
        Logger object.

    Returns
    -------
    Tuple[pd.DataFrame, np.ndarray]
        Feature matrix and target array.
    """
    data = pd.read_csv(
        data_s3_url,
        index_col=0
    )
    
    # Drop ID column and 'churn category' column (not useful for prediction)
    data.drop(['Customer ID', 'Churn Category'], axis=1, inplace=True)
    
    # Change column names to lower case and relace white spaces with underscore
    data.columns = [sub('\s', '_', col.lower()) for col in data.columns]
    
    X, y = data.drop(['churn_value'], axis=1), data.churn_value.values
    
    if logger is not None:
        logger.info('Data Loaded')
        logger.info(f'The shape of training set: {(X.shape, y.shape)}')
    
    return X, y

# ----------------------- Custom metric for evaluation ----------------------- #

def weighted_ap_score(predt: np.ndarray, data: np.ndarray) -> Tuple[str, float]:
    y_true = data
    y_score = predt
    weighted_ap_score = average_precision_score(y_true=y_true, y_score=y_score, average='weighted', pos_label=1)
    return 'avgAP', weighted_ap_score

# ------------------------ Stratified train/test split ----------------------- #

def stratified_split(X_train: pd.DataFrame, y_train: np.ndarray) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]:
    """
    Split the training set into train and validation sets, stratifying on the target variable.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features.
    y_train : np.ndarray
        Training target.

    Returns
    -------
    Tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]
        X_train, y_train, X_val, y_val.
    """
    ssf = StratifiedShuffleSplit(n_splits=1, test_size=0.2)
    
    for train_index, val_index in ssf.split(X_train, y_train):
        X_train, X_val = X_train.iloc[train_index], X_train.iloc[val_index]
        y_train, y_val = y_train[train_index], y_train[val_index]
    
    return X_train, y_train, X_val, y_val