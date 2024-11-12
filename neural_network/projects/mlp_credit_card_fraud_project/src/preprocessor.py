from typing import Union

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

# ------------------------------ Column dropper ------------------------------ #


class ColumnDropper(keras.layers.Layer):
    def __init__(self):
        super(ColumnDropper, self).__init__()

    def call(self, inputs: pd.DataFrame) -> pd.DataFrame:
        """
        ColumnDropper layer.

        Parameters
        ----------
        inputs : pd.DataFrame
            Data to be transformed.

        Returns
        -------
        pd.DataFrame
            Data with 'time' column dropped.
        """
        return inputs.drop("time", axis=1)


# ------------------------------ Log transformer ----------------------------- #


class LogTransformer(keras.layers.Layer):
    def __init__(self):
        super(LogTransformer, self).__init__()

    def call(self, inputs: pd.DataFrame) -> pd.DataFrame:
        """
        LogTransformer layer.

        Parameters
        ----------
        inputs : pd.DataFrame
            Data to be transformed.

        Returns
        -------
        pd.DataFrame
            Data with log transformation applied to the 'amount' column.
        """
        return inputs.assign(
            amount=np.log(inputs["amount"] + 1e-6)
        )  # Add small constant to avoid log(0)


# ------------------------------- Custom Scaler ------------------------------ #


class CustomScaler(keras.layers.Layer):
    def __init__(self):
        super(CustomScaler, self).__init__()

    def build(self, input_shape: tf.TensorShape):
        """
        The __call__() method of the layer will automatically run build the first time it is called. Implementing build() separately separates creating
        weights adapted from training data from using weights in every call on unseen data.

        Parameters
        ----------
        input_shape :
            Instance of TensorShape, tf.TensorShape(dims)
        """
        self.mean = self.add_weight(
            name="mean", shape=(input_shape[-1],), initializer="zeros", trainable=False
        )
        self.std = self.add_weight(
            name="std", shape=(input_shape[-1],), initializer="zeros", trainable=False
        )

    def call(
        self, inputs: Union[pd.DataFrame, np.array, tf.Tensor], training=None
    ) -> tf.Tensor:
        """
        A transformation from inputs to outputs-- the layer's forward pass.


        Parameters
        ----------
        inputs :
            Data to be transformed.
        training : bool, optional
            Flag for training (adapt to training data and compute weights) or inference (reuse weights on unseen data), by default None.

        Returns
        -------
        tf.Tensor
            Standardized data matrix.
        """
        if type(inputs) == pd.DataFrame:
            inputs = tf.convert_to_tensor(inputs, dtype=tf.float64)
        if training:
            self.mean = tf.math.reduce_mean(inputs, axis=0)
            self.std = tf.math.reduce_std(inputs, axis=0)
            return (inputs - self.mean) / self.std
        return (inputs - self.mean) / self.std


# ---------- Convert a Pandas DataFrame to a tf.data.Dataset object ---------- #


def df_to_dataset(data, shuffle=True, batch_size=32, seed=None):
    df = data.copy()
    labels = df.pop("class")
    # DataFrame.item returns tuple with column name and the content as a Series
    # Use tf.newaxis to add an axis to convert Series with shape (n,) to (n, 1) column vectors
    df = {key: value[:, tf.newaxis] for key, value in data.items()}
    ds = tf.data.Dataset.from_tensor_slices((dict(df), labels))
    if shuffle:
        # Faster than data.shape[0]
        ds = ds.shuffle(buffer_size=len(data), seed=seed)
    # Batch size is the number of training examples
    ds = ds.batch(batch_size)
    # ('batch_size' (32 by default) batches, 'batch_size' (32 by default) training examples per batch)
    ds = ds.prefetch(batch_size)
    return ds
