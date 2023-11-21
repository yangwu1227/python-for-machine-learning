from typing import Tuple, List
import argparse
import logging
import sys
import os
import io
import boto3

import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Nopep8
import tensorflow as tf

import bokeh
import bokeh.io
from bokeh.plotting import figure, show
from bokeh.models import HoverTool
from sagemaker.analytics import HyperparameterTuningJobAnalytics

# ------------------------- Class for data ingestion ------------------------- #

class NumpyS3(object):
    
    def __init__(self, bucket: str, key: str, client = None):
        if client is None:
            self.client = boto3.client('s3')
        else:
            self.client = client
        self.bucket = bucket
        self.key = key
        
    def __repr__(self) -> str:
        return f'S3(bucket = {self.bucket}, key = {self.key})'
    
    def download(self, object_key: str) -> np.ndarray:
        """
        Load numpy array from s3 bucket into memory.

        Parameters
        ----------
        object_key : str
            Key of the object in s3.

        Returns
        -------
        np.ndarray
            Data as numpy array.
        """
        obj = self.client.get_object(Bucket=self.bucket, Key=f'{self.key}/{object_key}')
        return np.load(io.BytesIO(obj['Body'].read()))
    
    def upload(self, object_key: str, array: np.ndarray) -> None:
        """
        Upload numpy array to s3 bucket as npy file.

        Parameters
        ----------
        object_key : str
            Key of the object in s3.
        array : np.ndarray
            Data as numpy array.
        """
        np.save(file='temp.npy', arr=array)
        self.client.upload_file(Filename='temp.npy', Bucket=self.bucket, Key=f'{self.key}/{object_key}')
        
        os.remove('temp.npy')
        
        return None
    
    def __del__(self):
        """
        The destructor of the class.
        """
        del self.client
        
        
# -------------------------- Class for loading data -------------------------- #

class DataHandler(NumpyS3):
    
    def __init__(self, bucket: str, key: str, s3_uri: str = None, client = None):
        super().__init__(bucket, key, client)
        
    def load_data(self, mode: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load data from s3 bucket.
        
        Parameters
        ----------
        mode : str
            Mode of the data to load. One of 'train', 'val', 'test'.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Tuple of numpy arrays--- X and y.
        """
        
        if mode == 'train':
            X_train = self.download(object_key='train-data/X_train.npy').reshape(-1, 28, 28, 1)
            y_train = self.download(object_key='train-data/y_train.npy')
            return X_train, y_train
        elif mode == 'val':
            X_val = self.download(object_key='val-data/X_val.npy').reshape(-1, 28, 28, 1)
            y_val = self.download(object_key='val-data/y_val.npy')
            return X_val, y_val
        elif mode == 'test':
            X_test = self.download(object_key='raw-data/test.npy').reshape(-1, 28, 28, 1)
            return X_test
            
        return None
    
    def __repr__(self) -> str:
        return f'DataHandler(bucket = {self.bucket}, key = {self.key})'
    
    def __del__(self):
        """
        The destructor of the class.
        """
        return super().__del__()
    
# ---------------------- Class for plotting HPO results ---------------------- #

class HoverHelper:
    def __init__(self, tuning_analytics: HyperparameterTuningJobAnalytics):
        self.tuner = tuning_analytics

    def hovertool(self) -> HoverTool:
        """
        Create a hovertool for the plot.

        Returns
        -------
        HoverTool
            A hovertool for the plot.
        """
        tooltips = [
            ("FinalObjectiveValue", "@FinalObjectiveValue"),
            ("TrainingJobName", "@TrainingJobName"),
        ]
        for k in self.tuner.tuning_ranges.keys():
            tooltips.append((k, "@{%s}" % k))
        ht = HoverTool(tooltips=tooltips)
        return ht

    def tools(self, standard_tools="pan,crosshair,wheel_zoom,zoom_in,zoom_out,undo,reset") -> List:
        """
        Return a list of tools for the plot.

        Parameters
        ----------
        standard_tools : str, optional
            A list of tools, by default "pan,crosshair,wheel_zoom,zoom_in,zoom_out,undo,reset"

        Returns
        -------
        List
            A list of tools for the plot.
        """
        return [self.hovertool(), standard_tools]

# --------------------- Parse argument from command line --------------------- #

def parser() -> argparse.Namespace:
    """
    Function that parses arguments from command line.

    Returns
    -------
    argparse.Namespace
        Namespace with arguments.
    """
    parser = argparse.ArgumentParser()

    # Data sources from s3 (not used since we use our own custom class for data ingestion)
    parser.add_argument('--s3_bucket', type=str, default='yang-ml-sagemaker')
    parser.add_argument('--s3_key', type=str, default='mnist')
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--val', type=str, default=os.environ['SM_CHANNEL_VAL'])

    # Model directory: we will use the default set by SageMaker, /opt/ml/model
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    
    # Architecture hyperparameters
    for i in range(1, 6):
        parser.add_argument(f'--filter_dim_{i}', type=int)
    for i in range(1, 3):
        parser.add_argument(f'--dense_units_{i}', type=int)
    parser.add_argument('--conv2d_regularizer_decay', type=float)
    parser.add_argument('--dense_regularizer_decay', type=float)
    parser.add_argument('--kernel_size', type=int)
    parser.add_argument('--dropout_rate', type=float)
    parser.add_argument('--batch_norm_momentum', type=float)
    # Optimizer and training hyperparameters
    parser.add_argument('--learning_rate', type=float)
    parser.add_argument('--clipnorm', type=float)
    parser.add_argument('epochs', type=int)
    parser.add_argument('batch_size', type=int, default=64)
    
    args, _ = parser.parse_known_args()

    return args

# ---------------------------------- Logger ---------------------------------- #

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
    formatter = logging.Formatter(log_format)
    # No matter how many processes we spawn, we only want one StreamHandler attached to the logger
    if not any(isinstance(handler, logging.StreamHandler) for handler in logger.handlers):
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.setLevel(logging.INFO)
    
    return logger