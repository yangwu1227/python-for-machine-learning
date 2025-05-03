import argparse
import logging
import os
import sys
from functools import partial
from typing import Dict, List, Tuple, Union

import numpy as np
import sagemaker
from PIL import Image

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Nopep8
import keras.backend as K
import tensorflow as tf
from tensorflow.keras import layers

# ---------------------------------- Logger ---------------------------------- #


def setup_logger(name: str) -> logging.Logger:
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
    parser.add_argument("--s3_key", type=str, default="lesion-segmentation")
    parser.add_argument("--s3_bucket", type=str, default="yang-ml-sagemaker")

    # Hyperparameters for fine-tuning via sagemaker tuning job
    parser.add_argument("--back_bone", type=str, default="resnet34")
    parser.add_argument("--random_rotation_factor", type=float)
    parser.add_argument("--random_flip_mode", type=str)
    parser.add_argument("--random_contrast_factor", type=float)
    parser.add_argument("--random_zoom_factor", type=float)
    parser.add_argument("--entry_block_batch_norm_momentum", type=float)
    parser.add_argument("--down_sample_kernel_size_0", type=int)
    parser.add_argument("--down_sample_kernel_size_1", type=int)
    parser.add_argument("--down_sample_kernel_size_2", type=int)
    parser.add_argument("--down_sample_pool_size_0", type=int)
    parser.add_argument("--down_sample_pool_size_1", type=int)
    parser.add_argument("--down_sample_pool_size_2", type=int)
    parser.add_argument("--down_sample_batch_norm_momentum_0", type=float)
    parser.add_argument("--down_sample_batch_norm_momentum_1", type=float)
    parser.add_argument("--down_sample_batch_norm_momentum_2", type=float)
    parser.add_argument("--up_sample_kernel_size_0", type=int)
    parser.add_argument("--up_sample_kernel_size_1", type=int)
    parser.add_argument("--up_sample_kernel_size_2", type=int)
    parser.add_argument("--up_sample_kernel_size_3", type=int)
    parser.add_argument("--up_sample_batch_norm_momentum_0", type=float)
    parser.add_argument("--up_sample_batch_norm_momentum_1", type=float)
    parser.add_argument("--up_sample_batch_norm_momentum_2", type=float)
    parser.add_argument("--up_sample_batch_norm_momentum_3", type=float)
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--clipnorm", type=float)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--epochs", type=int)

    # Data, model, and output directories
    parser.add_argument("--model_dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--train", type=str, default=os.environ["SM_CHANNEL_TRAIN"])
    parser.add_argument("--val", type=str, default=os.environ["SM_CHANNEL_VAL"])

    args, _ = parser.parse_known_args()

    return args


# --------------------------------- Load data -------------------------------- #


def load_data(
    paths: Dict[str, str], test_mode: bool = False
) -> Union[
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
]:
    """
    Load data from the given path.

    Parameters
    ----------
    paths : Dict[str, str]
        Dictionary of key: path to the data directories.
    test_mode : bool, optional
        Whether to load the test set, by default False.

    Returns
    -------
    Union[
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ]
        A tuple containing the training and validation data, and optionally test data.
    """
    train_images = np.load(file=os.path.join(paths["train"], "train_images.npy"))
    train_masks = np.load(file=os.path.join(paths["train"], "train_masks.npy"))
    val_images = np.load(file=os.path.join(paths["val"], "val_images.npy"))
    val_masks = np.load(file=os.path.join(paths["val"], "val_masks.npy"))

    if test_mode:
        test_images = np.load(file=os.path.join(paths["test"], "test_images.npy"))
        test_masks = np.load(file=os.path.join(paths["test"], "test_masks.npy"))
        return train_images, train_masks, val_images, val_masks, test_images, test_masks

    return train_images, train_masks, val_images, val_masks


# ---------------------------- Image deserializer ---------------------------- #


class ImageDeserializer(sagemaker.deserializers.BaseDeserializer):
    def __init__(self, accept: str = "image/png"):
        """
        Construct a ImageDeserializer.

        Parameters
        ----------
        accept : str, optional
            The accept content type expected by the ImageDeserializer.  By default, 'image/png'.
        """
        self.accept = accept

    @property
    def ACCEPT(self):
        """
        Get the accept content type expected by the ImageDeserializer.
        """
        return (self.accept,)

    def deserialize(self, stream, content_type) -> np.ndarray:
        """
        Deserialize a stream of bytes returned from an inference endpoint.

        Parameters
        ----------
        stream : botocore.response.StreamingBody
            A stream of bytes.
        content_type : str
            The MIME type of the data.

        Returns
        -------
        numpy.ndarray
            The numpy array of class labels per pixel
        """
        try:
            return np.array(Image.open(stream))
        finally:
            stream.close()


# ----------------------------------- U-net ---------------------------------- #


def unet_model(
    image_size: Tuple[int, int],
    aug_params: Dict[str, Union[float, str]],
    entry_block_filters: int,
    entry_block_kernel_size: int,
    entry_block_strides: int,
    entry_block_batch_norm_momentum: float,
    down_sample_strides: Union[List[int], Tuple[int]],
    down_sample_kernel_sizes: Union[List[int], Tuple[int]],
    down_sample_batch_norm_momentums: Union[List[float], Tuple[float]],
    down_sample_pool_sizes: Union[List[int], Tuple[int]],
    up_sample_strides: Union[List[int], Tuple[int]],
    up_sample_kernel_sizes: Union[List[int], Tuple[int]],
    up_sample_batch_norm_momentums: Union[List[float], Tuple[float]],
    up_sample_size: int,
    output_kernel_size: int,
    num_channels: int = 1,
) -> tf.keras.Model:
    """
    Modified U-net model for binary segmentation of lesions.

    Parameters
    ----------
    image_size : Tuple[int, int]
        The dimensions of the input images.
    aug_params : Dict[str, Union[float, str]]
        A dictionary of data augmentation parameters.
    entry_block_filters : int
        The number of filters for the entry block.
    entry_block_kernel_size : int
        The kernel size for the entry block.
    entry_block_strides : int
        The strides for the entry block.
    entry_block_batch_norm_momentum : float
        The batch normalization momentum for the entry block.
    down_sample_strides : Union[List[int], Tuple[int]]
        A list of three strides for the down sampling blocks.
    down_sample_kernel_sizes : Union[List[int], Tuple[int]]
        A list of three kernel sizes for the down sampling blocks.
    down_sample_batch_norm_momentums : Union[List[float], Tuple[float]]
        A list of three batch normalization momentums for the down sampling blocks.
    down_sample_pool_sizes : Union[List[int], Tuple[int]]
        A list of three pool sizes for the down sampling blocks.
    up_sample_strides : Union[List[int], Tuple[int]]
        A list of three strides for the up sampling blocks.
    up_sample_kernel_sizes : Union[List[int], Tuple[int]]
        A list of three kernel sizes for the up sampling blocks.
    up_sample_batch_norm_momentums : Union[List[float], Tuple[float]]
        A list of three batch normalization momentums for the up sampling blocks.
    up_sample_size : int
        The up sampling size, which should be 2 for this use case.
    output_kernel_size : int
        The kernel size for the output layer.
    num_channels : int, optional
        Number of channels, which should be 1 for this use case, by default 1.

    Returns
    -------
    tf.keras.Model
        A U-net model for binary segmentation of lesion images.
    """
    # Data augmentation layers
    data_augmentation = tf.keras.Sequential(
        [
            # layers.RandomBrightness(factor=aug_params['random_brightness_factor'], value_range=(0.0, 255.0)), # Scaling applied after augmentation
            layers.RandomContrast(factor=aug_params["random_contrast_factor"]),
            layers.RandomFlip(mode=aug_params["random_flip_mode"]),
            layers.RandomRotation(factor=aug_params["random_rotation_factor"]),
            layers.RandomZoom(height_factor=aug_params["random_zoom_factor"]),
        ]
    )
    # Default separable convolutional layer
    default_separable_conv2d = partial(
        layers.SeparableConv2D,
        padding="same",
        depthwise_initializer="he_normal",
        pointwise_initializer="he_normal",
    )
    # Default traditional 2D convolutional layer
    default_conv2d = partial(
        layers.Conv2D, padding="same", kernel_initializer="he_normal"
    )
    # Default convolutional transpose layer
    default_conv2d_transpose = partial(
        layers.Conv2DTranspose, padding="same", kernel_initializer="he_normal"
    )

    inputs = tf.keras.Input(shape=image_size + (num_channels,))
    x = data_augmentation(inputs)
    x = tf.keras.layers.Rescaling(scale=(1.0 / 255.0))(x)

    # Entry block
    x = default_conv2d(
        filters=entry_block_filters,
        kernel_size=entry_block_kernel_size,
        strides=entry_block_strides,
    )(x)
    x = layers.BatchNormalization(momentum=entry_block_batch_norm_momentum)(x)
    x = layers.Activation("relu")(x)

    # ------------------------------- Down sampling ------------------------------ #

    # Set aside the residual
    previous_block_activation = x

    # Blocks 1, 2, 3 in the encoder
    for filters, stride, kernel_size, momentum, pool_size in zip(
        [64, 128, 256],
        down_sample_strides,
        down_sample_kernel_sizes,
        down_sample_batch_norm_momentums,
        down_sample_pool_sizes,
    ):
        x = layers.Activation("relu")(x)
        # Proposed to be faster than traditional Conv2D
        x = default_separable_conv2d(filters=filters, kernel_size=kernel_size)(x)
        x = layers.BatchNormalization(momentum=momentum)(x)

        x = layers.Activation("relu")(x)
        x = default_separable_conv2d(filters=filters, kernel_size=kernel_size)(x)
        x = layers.BatchNormalization(momentum=momentum)(x)

        x = layers.MaxPooling2D(pool_size=pool_size, strides=stride)(x)

        # Project residual
        residual = default_conv2d(filters=filters, kernel_size=1, strides=stride)(
            previous_block_activation
        )
        # Add residual from previous to current
        x = layers.add([x, residual])
        # Next residual
        previous_block_activation = x

    # ------------------------------- Up sampling ------------------------------ #

    # Blocks 4, 5, 6 in the decoder
    for filters, stride, kernel_size, momentum in zip(
        [256, 128, 64, 32],
        up_sample_strides,
        up_sample_kernel_sizes,
        up_sample_batch_norm_momentums,
    ):
        x = layers.Activation("relu")(x)
        x = default_conv2d_transpose(filters=filters, kernel_size=kernel_size)(x)
        x = layers.BatchNormalization(momentum=momentum)(x)

        x = layers.Activation("relu")(x)
        x = default_conv2d_transpose(filters=filters, kernel_size=kernel_size)(x)
        x = layers.BatchNormalization(momentum=momentum)(x)

        x = layers.UpSampling2D(size=up_sample_size)(x)

        residual = layers.UpSampling2D(size=up_sample_size)(previous_block_activation)
        residual = default_conv2d(filters=filters, kernel_size=1)(residual)
        x = layers.add([x, residual])
        previous_block_activation = x

    # ------------------------------- Output block ------------------------------ #

    # Binary semantic segmentation with single channel output (either background or lesion) and we do not use the default layer defined above
    outputs = default_conv2d(
        filters=num_channels, kernel_size=output_kernel_size, activation="sigmoid"
    )(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    return model


# --------------------------------- Dice Loss -------------------------------- #


def dice_loss(y_true: tf.Tensor, y_pred: tf.Tensor, epsilon: float = 1e-7) -> tf.Tensor:
    """
    Computes the dice loss between the true and predicted masks for each class.

    Parameters
    ----------
    y_true : tf.Tensor
        The true multi-class masks.
    y_pred : tf.Tensor
        The predicted multi-class masks.
    epsilon : float, optional
        A small constant to avoid division by zero, by default 1e-7.

    Returns
    -------
    tf.Tensor
        The dice loss averaged across all classes.
    """
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)

    intersection = K.sum(y_true * y_pred)
    union = K.sum(y_true) + K.sum(y_pred)
    dice_coef = (2.0 * intersection + epsilon) / (union + epsilon)
    dice_loss = 1.0 - dice_coef

    return dice_loss
