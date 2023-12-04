import os
import joblib
import io
import logging
import sys
from typing import Dict, Optional, Any, Tuple, List, Union

import pandas as pd
import numpy as np
import boto3

from hydra import compose, initialize, core
from omegaconf import OmegaConf

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import seaborn as sns
from matplotlib.ticker import MaxNLocator
from sktime.split import SlidingWindowSplitter
from sktime.forecasting.compose import ColumnEnsembleForecaster

# ---------------------------------- Set up ---------------------------------- #

class SetUp(object):
    """ 
    A utility class to set up logger and read configuration for each entry point script.
    """
    def __init__(self, logger_name: str, config_name: str, config_path: str):
        """ 
        Constructor for the SetUp class.
        
        Parameters
        ----------
        logger_name : str
            The name of the logger.
        config_name : str
            The name for the configuration job.
        config_path : str
            The path to the directory containing the YAML configuration file.
        """
        self.logger_name = logger_name
        self.config_name = config_name
        self.config_path = config_path
        
    def _get_logger(self, name: str) -> logging.Logger:
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

    def _get_config(self, job_name: str, config_path: str = 'config') -> Dict[str, Any]:
        """
        Read configuration from a YAML file.
        
        Parameters
        ----------
        job_name : str
            The name of the job (e.g. 'preprocess', 'train', 'predict').
        config_path : str, optional
            The path to the directory containing the YAML configuration file.
            
        Returns
        -------
        Dict[str, Any]
            A dictionary containing configuration parameters.
        """
        core.global_hydra.GlobalHydra.instance().clear()
        initialize(version_base='1.2', config_path=config_path, job_name=job_name)
        config = OmegaConf.to_container(compose(config_name='main'), resolve=True)
            
        return config
    
    def setup(self) -> Tuple[logging.Logger, Dict[str, Any]]:
        """
        Set up logger and read configuration.
        
        Returns
        -------
        Tuple[logging.Logger, Dict[str, Any]]
            A tuple containing a logger and a dictionary containing configuration parameters.
        """
        logger = self._get_logger(self.logger_name)
        config = self._get_config(self.config_name, self.config_path)
        
        return logger, config

# ------------------------------ S3 helper class ----------------------------- #

class S3Helper(object):
    """
    A utility class to read from and write to files on S3.
    """
    def __init__(self, s3_bucket: Optional[str] = 'yang-ml-sagemaker', s3_key: Optional[str] = 'chicago_cta_ridership', credentials: Optional[Dict[str, str]] = None):
        """
        Constructor for the S3Helper class.
        
        Parameters
        ----------
        credentials : Dict[str, str], optional
            A dictionary containing AWS credentials with keys 'key', 'secret', and 'token'.
            Defaults to None, in which case credentials are taken from environment variables.
        s3_bucket : str, optional
            The S3 bucket name. Defaults to 'yang-ml-sagemaker'.
        s3_key : str, optional
            The base S3 key (path) where the Parquet files are located or will be written to.
            Defaults to 'chicago_cta_ridership'.
        """
        if credentials is None:
            try:
                self.access_key = os.environ['AWS_ACCESS_KEY_ID']
                self.secret_key = os.environ['AWS_SECRET_ACCESS_KEY']
                self.session_token = os.environ['AWS_SESSION_TOKEN']
            except KeyError:
                raise ValueError('Please ensure AWS credentials are provided or set as environment variables.')
        else:
            self.access_key = credentials['key']
            self.secret_key = credentials['secret']
            self.session_token = credentials['token']

        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key,
            aws_session_token=self.session_token
        )
        self.s3_bucket = s3_bucket
        self.s3_key = s3_key
        
    @property
    def s3_bucket(self) -> str:
        return self._s3_bucket
    
    @s3_bucket.setter
    def s3_bucket(self, s3_bucket: str) -> None:
        if not isinstance(s3_bucket, str):
            raise TypeError('s3_bucket must be a string')
        self._s3_bucket = s3_bucket
        
    @property
    def s3_key(self) -> str:
        return self._s3_key
    
    @s3_key.setter
    def s3_key(self, s3_key: str) -> None:
        if not isinstance(s3_key, str):
            raise TypeError('s3_key must be a string')
        self._s3_key = s3_key 
    
    def read_parquet(self, obj_key: str) -> pd.DataFrame:
        """
        Read a Parquet file from S3.
        
        Parameters
        ----------
        obj_key : str
            The specific object key under the base key to read from.
        
        Returns
        -------
        pd.DataFrame
            The DataFrame read from the Parquet file.
        """ 
        s3_path = f"{self.s3_key}/{obj_key}"
        response = self.s3_client.get_object(Bucket=self.s3_bucket, Key=s3_path)
        
        return pd.read_parquet(io.BytesIO(response['Body'].read()))
    
    def to_parquet(self, data: pd.DataFrame, obj_key: str) -> None:
        """
        Write a DataFrame to a Parquet file on S3.
        
        Parameters
        ----------
        data : pd.DataFrame
            The DataFrame to be written to the Parquet file.
        obj_key : str
            The specific object key under the base key to write to.
        """
        s3_path = f"{self.s3_key}/{obj_key}"
        buffer = io.BytesIO()
        data.to_parquet(buffer)
        buffer.seek(0)
        self.s3_client.put_object(Bucket=self.s3_bucket, Key=s3_path, Body=buffer.getvalue())
        
    def upload_joblib(self, obj: Any, obj_key: str) -> None:
        """
        Upload a joblib model to S3.
        
        Parameters
        ----------
        obj : Any
            The object to be uploaded.
        obj_key : str
            The specific object key under the base key to write to.
        """
        s3_path = f'{self.s3_key}/{obj_key}'
        buffer = io.BytesIO()
        joblib.dump(obj, buffer)
        buffer.seek(0)
        self.s3_client.put_object(Bucket=self.s3_bucket, Key=s3_path, Body=buffer.getvalue())
        
    def download_joblib(self, obj_key: str) -> Any:
        """
        Download a joblib model from S3.
        
        Parameters
        ----------
        obj_key : str
            The specific object key under the base key to read from.
            
        Returns
        ------- 
        Any
            The object downloaded from S3, which, in this project, is the deserialized trainer instance.
        """
        s3_path = f'{self.s3_key}/{obj_key}'
        response = self.s3_client.get_object(Bucket=self.s3_bucket, Key=s3_path)
        buffer = io.BytesIO(response['Body'].read())
        obj = joblib.load(buffer)
        
        return obj
    
    def _add_to_tree(self, base: Dict[str, Any], parts: List[str], obj_info: Optional[Dict[str, Any]] = None) -> None:
        """
        Recursively adds entries to the directory tree, only adding `obj_info` to the leaf nodes.

        Parameters
        ----------
        base : Dict[str, Any]
            The current level in the directory tree.
        parts : List[str]
            The remaining parts of the key to add.
        obj_info : Optional[Dict[str, Any]], optional
            The information about the object to add (for leaf nodes only).

        Returns
        -------
        None
        """
        # If this is the last part, i.e., this is a file object
        if len(parts) == 1:
            # Add the obj_info if present
            if obj_info is not None:
                base[parts[0]] = obj_info
            else:
                # Ensure that there is a dictionary available to represent the current level of the tree
                # This will be further populated with subdirectories and files in the recursive calls
                base.setdefault(parts[0], {})
        # If 'parts' has more than one element, i.e., this is a subdirectory
        else:
            # If the first element of 'parts' is not in the current level of the tree, add it
            if parts[0] not in base:
                base[parts[0]] = {}
            # Recurse into the subdirectory
            self._add_to_tree(base[parts[0]], parts[1:], obj_info)
            
    def _parse_s3_response_to_tree(self, response: Dict) -> Dict:
        """
        Parses the S3 `list_objects_v2` response to a nested dictionary structure representing
        the S3 bucket's folder and file hierarchy.

        Parameters
        ----------
        response : Dict
            The response from the S3 list_objects_v2 call.

        Returns
        -------
        Dict
            A nested dictionary representing the folder and file structure of the S3 bucket.
        """
        directory_tree = {}

        # Add files to the directory tree with their metadata
        for obj in response.get('Contents', []):
            key = obj['Key']
            parts = key.split('/')
            # Skip this entry as it's a folder and not a file
            if parts[-1] == '':
                continue 
            obj_info = {
                'Size': obj['Size'],
                'LastModified': obj['LastModified'].isoformat()
            }
            self._add_to_tree(directory_tree, parts, obj_info)

        # Add common prefixes as subdirectories, but without any metadata
        for prefix in response.get('CommonPrefixes', []):
            subdirectory = prefix['Prefix']
            parts = subdirectory.rstrip('/').split('/')
            self._add_to_tree(directory_tree, parts)

        return directory_tree
    
    def list_objects(self) -> Dict:
        """
        List project s3 bucket objects as a nested dictionary.
        
        Returns 
        -------
        Dict
            A nested dictionary representing the folder and file structure of the S3 bucket. 
        """
        response = self.s3_client.list_objects_v2(Bucket=self.s3_bucket, Prefix=self.s3_key)
        
        return self._parse_s3_response_to_tree(response)
        
class CVHelper(object):
    """
    Utility class for cross-validation.
    """
    @staticmethod
    def calculate_window_size(num_splits: int, n: int, h: int, s: int) -> int:
        """
        Calculate the maximum window size that allows for at least the desired number of splits.
        
        Parameters
        ----------
        num_splits : int
            The number of splits to be created from the dataset.
        n : int
            The number of observations in the dataset.
        h : int
            The forecast horizon.
        s : int
            The step size.

        Returns
        -------
        int
            The sliding window size. 
        """
        if num_splits < 2:
            raise ValueError('Number of splits must be at least 2')
        
        if n <= 0:
            raise ValueError('Total number of data points (n) must be positive')
        
        if h <= 0 or s <= 0:
            raise ValueError('Forecasting horizon (h) and step length (s) must be positive')
        
        # Calculate the maximum window size that allows for at least the desired number of splits
        w = n - h - (s * num_splits - 2 * s)
        
        if w <= 0:
            raise ValueError('Calculated window size is non-positive. Please adjust your parameters')
        
        return w

    @staticmethod
    def calculate_num_splits(n: int, h: int, s: int, w: int) -> int:
        """
        Calculate the number of splits that can be created from the dataset.
        
        Parameters
        ----------
        n : int
            The number of observations in the dataset.
        h : int
            The forecast horizon.
        s : int
            The step size.
        w : int
            The sliding window size.

        Returns
        -------
        int
            The number of splits that can be created from the dataset.
        """
        if n <= 0:
            raise ValueError('Total number of data points (n) must be positive')
        
        if h <= 0 or s <= 0:
            raise ValueError('Forecasting horizon (h) and step length (s) must be positive')
        
        if w <= 0:
            raise ValueError('Sliding window size (w) must be positive')
        
        # Calculate the number of splits that can be created from the dataset
        num_splits = (n - h - w) // s + 2
        
        if num_splits < 2:
            raise ValueError('Calculated number of splits is less than 2. Please adjust your parameters')
        
        return num_splits
    
    @staticmethod
    def plot_windows(cv: SlidingWindowSplitter, y: pd.Series, title: str = 'CV Splits', ax: Axes = None) -> Axes:
        """
        Plot training and test windows for each split in a time series cross-validation.

        Parameters
        ----------
        cv : SlidingWindowSplitter
            An instance of SlidingWindowSplitter from sktime for time series cross-validation.
        y : pd.Series
            The time series data.
        title : str, optional
            The plot title, by default 'CV Plots'
        ax : Axes, optional
            Matplotlib Axes on which to plot, by default None

        Returns
        -------
        Axes
            The axes on which the plot is drawn.
        """
        if ax is None:
            fig, ax = plt.subplots(1, figsize=plt.figaspect(0.25))
        else:
            fig = ax.figure
        
        train_windows = []
        test_windows = []
        for train, test in cv.split(y):
            train_windows.append(train)
            test_windows.append(test)
        
        def get_y(length: int, split: int) -> np.ndarray:
            return np.ones(length) * split

        n_splits = len(train_windows)
        n_timepoints = len(y)
        len_test = len(test_windows[0])

        train_color, test_color = sns.color_palette("colorblind")[:2]

        for i in range(n_splits):
            train = train_windows[i]
            test = test_windows[i]

            ax.plot(
                np.arange(n_timepoints), get_y(n_timepoints, i), marker="o", c="lightgray"
            )
            ax.plot(
                train,
                get_y(len(train), i),
                marker="o",
                c=train_color,
                label="Window",
            )
            ax.plot(
                test,
                get_y(len_test, i),
                marker="o",
                c=test_color,
                label="Forecasting horizon",
            )
            
        ax.invert_yaxis()
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        xtickslocs = [tick for tick in ax.get_xticks() if tick in np.arange(n_timepoints)]
        ax.set(
            title=title,
            ylabel="Window number",
            xlabel="Time",
            xticks=xtickslocs,
            xticklabels=y.iloc[xtickslocs].index,
        )
        
        # remove duplicate labels/handles
        handles, labels = ((leg[:2]) for leg in ax.get_legend_handles_labels())
        ax.legend(handles, labels)
        
        return ax

    @staticmethod
    def plot_cv_windows(w: int, s: int, h: int, start_date: str, end_date: str, freq: str) -> None:
        """
        Plot the sliding windows.
        
        Parameters
        ----------
        w : int
            The sliding window size.
        s : int
            The step size.
        h : int
            The forecast horizon.
        start_date : str
            The start date of the time series.
        end_date : str
            The end date of the time series.   
        freq : str
            The frequency of the time series.
        """
        if w <= 0:
            raise ValueError('Sliding window size (w) must be positive')
        
        if s <= 0:
            raise ValueError('Step size (s) must be positive')
        
        if h <= 0:
            raise ValueError('Forecast horizon (h) must be positive')
        
        cv = SlidingWindowSplitter(window_length=w, step_length=s, fh=range(1, h + 1))
        
        # Create a dummy series with index as the date range (no hour, minute, second)
        y_train = pd.Series(np.zeros(len(pd.date_range(start_date, end_date, freq=freq))), index=pd.date_range(start_date, end_date, freq=freq))
    
        CVHelper.plot_windows(cv, y_train, ax=None)
        plt.show()