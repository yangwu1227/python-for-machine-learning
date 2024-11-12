import argparse
import ast
import base64
import json
import logging
import os
import sys
from functools import partial
from typing import Any, Dict, List, Tuple, Union

import boto3
import numpy as np
from botocore.exceptions import ClientError

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Nopep8
import tensorflow as tf
from bokeh.models import HoverTool
from sagemaker.analytics import HyperparameterTuningJobAnalytics

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

    log_format = "%(asctime)s %(levelname)s %(name)s: %(message)s"
    formatter = logging.Formatter(log_format)
    # No matter how many processes we spawn, we only want one StreamHandler attached to the logger
    if not any(
        isinstance(handler, logging.StreamHandler) for handler in logger.handlers
    ):
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.setLevel(logging.INFO)

    return logger


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

    # AWS
    parser.add_argument("--s3_key", type=str, default="weather-classification")
    parser.add_argument("--s3_bucket", type=str, default="yang-ml-sagemaker")

    # Optuna database
    parser.add_argument("--host", type=str)
    parser.add_argument("--db_name", type=str, default="optuna")
    parser.add_argument("--db_secret", type=str, default="optuna/db")
    parser.add_argument("--study_name", type=str, default="optuna_cnn")
    parser.add_argument("--region_name", type=str, default="us-east-1")
    parser.add_argument("--n_trials", type=int, default=20)

    # Hyperparameters for fine-tuning via sagemaker tuning job
    parser.add_argument("--dense_units", type=int)
    parser.add_argument("--dense_weight_decay", type=float)
    parser.add_argument("--random_contrast_factor", type=float)
    parser.add_argument("--random_flip_mode", type=str)
    parser.add_argument("--random_rotation_factor", type=float)
    parser.add_argument("--random_zoom_factor", type=float)
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--clipnorm", type=float)
    parser.add_argument("--dropout_rate", type=float)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--epochs", type=int)

    # Data, model, and output directories
    parser.add_argument("--model_dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--train", type=str, default=os.environ["SM_CHANNEL_TRAIN"])
    parser.add_argument("--val", type=str, default=os.environ["SM_CHANNEL_VAL"])
    # For storing job name as a 'optuna.trial' attribute
    parser.add_argument(
        "--training_env", type=str, default=json.loads(os.environ["SM_TRAINING_ENV"])
    )

    args, _ = parser.parse_known_args()

    return args


# --------------------------- Function to load data -------------------------- #


def load_data(paths: Dict[str, str], test_mode: bool = False) -> Tuple[np.ndarray]:
    """
    Load data from the given path.

    Parameters
    ----------
    path : List[str]
        List of key: path to the data directories.
    test_mode : bool, optional
        Whether to load the test set, by default False.

    Returns
    -------
    Tuple[np.ndarray]
        A tuple of numpy arrays--- X_train, y_train, X_val, y_val, X_test, y_test
    """
    X_train = np.load(file=os.path.join(paths["train"], "X_train.npy"))
    y_train = np.load(file=os.path.join(paths["train"], "y_train.npy"))
    X_val = np.load(file=os.path.join(paths["val"], "X_val.npy"))
    y_val = np.load(file=os.path.join(paths["val"], "y_val.npy"))
    if test_mode:
        X_test = np.load(file=os.path.join(paths["test"], "X_test.npy"))
        y_test = np.load(file=os.path.join(paths["test"], "y_test.npy"))
        return X_train, y_train, X_val, y_val, X_test, y_test

    return X_train, y_train, X_val, y_val


# --------------------- Function for setting up database --------------------- #


def get_secret(secret_name: str, region_name: str = "ur-east-1") -> Union[Dict, bytes]:
    """
    Get secret from AWS Secrets Manager.

    Parameters
    ----------
    secret_name : str
        Name of the secret to retrieve.
    region_name : str, optional
        Region, by default 'ur-east-1'

    Returns
    -------
    Union[Dict, bytes]
        Secret retrieved from AWS Secrets Manager.
    """

    # Create a Secrets manager client
    session = boto3.session.Session()
    client = session.client(service_name="secretsmanager", region_name=region_name)
    try:
        get_secret_value_response = client.get_secret_value(SecretId=secret_name)
    except ClientError as e:
        if e.response["Error"]["Code"] == "DecryptionFailureException":
            # Secrets Manager can't decrypt the protected secret text using the provided KMS key
            raise e
        elif e.response["Error"]["Code"] == "InternalServiceErrorException":
            # An error occurred on the server side
            raise e
        elif e.response["Error"]["Code"] == "InvalidParameterException":
            # We provided an invalid value for a parameter
            raise e
        elif e.response["Error"]["Code"] == "InvalidRequestException":
            # We provided a parameter value that is not valid for the current state of the resource
            raise e
        elif e.response["Error"]["Code"] == "ResourceNotFoundException":
            # Can't find the resource that we asked for
            raise e
    else:
        # If the secret was a JSON-encoded dictionary string, convert it to dictionary
        if "SecretString" in get_secret_value_response:
            secret = get_secret_value_response["SecretString"]
            secret = ast.literal_eval(secret)  # Convert string to dictionary
            return secret
        # If the secret was binary, decode it
        else:
            decoded_binary_secret = base64.b64decode(
                get_secret_value_response["SecretBinary"]
            )
            return decoded_binary_secret


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

    def tools(
        self, standard_tools="pan,crosshair,wheel_zoom,zoom_in,zoom_out,undo,reset"
    ) -> List:
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


# ------------------------- Function for building cnn ------------------------ #


def baseline_cnn(
    conv_params: Dict[str, Any],
    dense_params: Dict[str, Any],
    aug_params: Dict[str, Any],
    opt_params: Dict[str, Any],
    input_shape: Tuple[int] = (256, 256, 3),
) -> tf.keras.models.Sequential:
    """
    Build and compile a convolutional neural network.

    Parameters
    ----------
    conv_params : Dict[str, Any]
        Hyperparameters for convolutional layers.
    dense_params : Dict[str, Any]
        Hyperparameters for dense layers.
    aug_params : Dict[str, Any]
        Hyperparameters for data augmentation.
    opt_params : Dict[str, Any]
        Hyperparameters for optimizer.
    input_shape : Tuple[int], optional
        Dimension of the input feature vector, by default (256, 256, 3).

    Returns
    -------
    tf.keras.models.Sequential
        A compiled convolutional neural network
    """
    # Default convolutional layer
    DefaultConv2D = partial(
        tf.keras.layers.Conv2D,
        kernel_size=3,
        padding="same",
        activation="relu",
        kernel_initializer="he_normal",
        kernel_regularizer=tf.keras.regularizers.l2(conv_params["conv2d_weight_decay"]),
    )
    # Default dense layer
    DefaultDense = partial(
        tf.keras.layers.Dense,
        activation="relu",
        kernel_initializer="he_normal",
        kernel_regularizer=tf.keras.regularizers.l2(dense_params["dense_weight_decay"]),
    )
    # Data augmentation layers
    data_augmentation = tf.keras.Sequential(
        [
            # tf.keras.layers.RandomBrightness(factor=aug_params['random_brightness_factor'], value_range=(0.0, 255.0)), # Scaling applied after augmentation
            tf.keras.layers.RandomContrast(factor=aug_params["random_contrast_factor"]),
            tf.keras.layers.RandomFlip(mode=aug_params["random_flip_mode"]),
            tf.keras.layers.RandomRotation(factor=aug_params["random_rotation_factor"]),
            tf.keras.layers.RandomZoom(height_factor=aug_params["random_zoom_factor"]),
        ]
    )

    # Model architecture
    inputs = tf.keras.Input(shape=input_shape)
    x = data_augmentation(inputs)
    x = tf.keras.layers.Rescaling(scale=(1.0 / 255.0))(x)

    for i in range(conv_params["n_conv_layers"]):
        x = DefaultConv2D(filters=conv_params["filters_list"][i])(x)
        x = tf.keras.layers.MaxPool2D(pool_size=conv_params["pool_size"])(x)
        x = tf.keras.layers.BatchNormalization(
            momentum=conv_params["conv_batch_norm_momentum"]
        )(x)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dropout(rate=dense_params["dropout_rate"])(x)

    for j in range(dense_params["n_dense_layers"]):
        x = DefaultDense(units=dense_params["units_list"][j])(x)
        x = tf.keras.layers.BatchNormalization(
            momentum=dense_params["dense_batch_norm_momentum"]
        )(x)

    # Output layers (softmax for multi-class classification)
    outputs = tf.keras.layers.Dense(units=4, activation="softmax")(x)
    cnn_model = tf.keras.Model(inputs=inputs, outputs=outputs)
    cnn_model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=opt_params["learning_rate"], clipnorm=opt_params["clipnorm"]
        ),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )

    return cnn_model
