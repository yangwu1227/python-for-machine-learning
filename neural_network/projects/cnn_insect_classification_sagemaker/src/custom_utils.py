import os
import argparse
from typing import Tuple, Union, List, Dict, Any, Optional, Callable
import logging
import sys
import json

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Nopep8
import tensorflow as tf
from tensorflow.keras.utils import image_dataset_from_directory

from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import HtmlFormatter
import IPython
from pygments.style import Style
from pygments.token import Keyword, Name, Comment, String, Error, \
     Number, Operator, Generic

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
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(log_format))
    logger.addHandler(handler)

    logger.setLevel(logging.INFO)
    
    return logger

# --------------------- Parse argument from command line --------------------- #

def parser() -> argparse.ArgumentParser:
    """
    Function that parses arguments from the command line.

    Returns
    -------
    argparse.ArgumentParser
        An ArgumentParser object that contains the arguments passed from command line.
    """
    parser = argparse.ArgumentParser()

    # Data, model, and output directories 
    parser.add_argument('--model_dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--val', type=str, default=os.environ['SM_CHANNEL_VAL'])
    parser.add_argument('--training_env', type=str, default=json.loads(os.environ['SM_TRAINING_ENV']))

    parser.add_argument('--test_mode', type=int, default=0)

    return parser

# ------ Function decorator for adding additional command line arguments ----- #

def add_additional_args(parser_func: Callable, additional_args: Dict[str, type]) -> Callable:
    """
    Function decorator that adds additional command line arguments to the parser.
    This allows for adding additional arguments without having to change the base
    parser.

    Parameters
    ----------
    parser_func : Callable
        The parser function to add arguments to.
    additional_args : Dict[str, type]
        A dictionary where the keys are the names of the arguments and the values
        are the types of the arguments, e.g. {'arg1': str, 'arg2': int}.

    Returns
    -------
    Callable
        A parser function that returns the ArgumentParser object with the additional arguments added to it.
    """
    def wrapper():
        # Call the original parser function to get the parser object
        parser = parser_func()

        for arg_name, arg_type in additional_args.items():
            parser.add_argument(f'--{arg_name}', type=arg_type)

        args, _ = parser.parse_known_args()

        return args

    return wrapper

# ------------------- Parametrized data augmentation layer ------------------- #

class AugmentationModel(object):
    """
    A class for creating a parametrized data augmentation layers, which is essentially a sequetial model. This class can be extended to include more data augmentation layers, which the user can specify using 'layer_name' and **kwargs pairs.
    """
    def __init__(self, aug_params):
        """
        Instantiate the augmentation model. The augmentation parameters are passed as a dictionary.
        The format of the dictionary is must be 'layer_name': {'param1': value1, 'param2': value2, ...}.
        For example, for random flip, the dictionary is {'RandomFlip': {'mode': 'horizontal'}}

        Parameters
        ----------
        aug_params : Dict[str, Dict[str, Any]]
            The augmentation parameters.
        """
        # Base model is a sequential model
        self.base_model = tf.keras.Sequential()
        # Augmentation layer: parameters are passed as a dictionary
        self.aug_params = aug_params

    @property
    def aug_params(self):
        return self._aug_params

    # Validate aug_params input
    @aug_params.setter
    def aug_params(self, aug_params):
        if not isinstance(aug_params, dict):
            raise TypeError('The augmentation parameters must be supplied as a dictionary of str -> dict')
        self._aug_params = aug_params

    def _add_augmentation_layer(self, layer_name, **kwargs):
        """
        Private method for adding a single augmentation layer to the model.

        Parameters
        ----------
        layer_name : str
            The name of the augmentation layer.
        **kwargs : Dict[str, Any]
            The parameters for the augmentation layer as a dictionary. The keys are the parameter names 
            and the values are the parameter values.
        """
        # Intantiate a layer from a config dictionary
        layer = tf.keras.layers.deserialize(config={
            'class_name': layer_name,
            'config': kwargs
        })
        # Add the layer to the base model
        self.base_model.add(layer)

    def build_augmented_model(self):
        """
        Build the augmented model with the specified data augmentation layers.

        Returns
        -------
        tf.keras.Model
            The augmented model.
        """
        for layer_name, args in self.aug_params.items():
            if not isinstance(args, dict):
                raise ValueError(f'Augmentation layer arguments should be provided as a dictionary for layer: {layer_name}')
            self._add_augmentation_layer(layer_name, **args)

        model = tf.keras.Sequential([self.base_model], name='data_augmentation_layers')

        return model

# -------------------------- Load model as a dataset ------------------------- #

def load_datasets(dir, batch_size: int, val: bool) -> tf.data.Dataset:
    """
    Read in the dataset from the specified directory and return a tf.data.Dataset object.

    Parameters
    ----------
    batch_size : int
        The batch size.
    val : bool
        Whether the dataset is a validation dataset.

    Returns
    -------
    tf.data.Dataset
        The dataset with the specified batch size.
    """
     # Load data as tensorflow dataset
    dataset = tf.data.Dataset.load(dir).unbatch().batch(batch_size)

    if val:
        return dataset.repeat()
    else:
        return dataset
    
# ----------------------------- Pretty print code ---------------------------- #

def pretty_print_code(filename: str) -> Union[IPython.core.display.HTML, None]:
    """
    Function to pretty print Python code from a file.

    Parameters
    ----------
    filename : str
        The path to the Python file to be pretty printed.

    Returns
    -------
    IPython.core.display.HTML or None
        The HTML object containing the pretty printed code, or None if the file could not be read.
    """
    try:
        with open(filename, 'r') as file:
            code = file.read()
    except OSError:
        return None

    formatter = HtmlFormatter(style='default')
    result = highlight(code, PythonLexer(), formatter)
    return IPython.display.HTML('<style type="text/css">{}</style>{}'.format(
        formatter.get_style_defs('.highlight'),
        result
    ))