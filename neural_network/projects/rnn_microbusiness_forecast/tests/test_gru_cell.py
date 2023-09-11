import os
from shutil import rmtree
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Nopep8
from typing import Dict, Union, Callable, Tuple
import tensorflow as tf
from src.gru_entry import CustomGRUCell, SymmetricMeanAbsolutePercentageError
import pytest

@pytest.fixture(scope='class')
def generate_data():
    """
    Fixture factory returning a function that generates data for the tests.
    """
    def _generate_data(batch_size: int, num_months: int, num_predictions: int) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Random generated data for training a small GRU model.

        Parameters
        ----------
        batch_size : int
            The number of examples in a batch.
        num_months : int
            The number of months of data to use for training.
        num_predictions : int
            The number of months of data to predict.

        Returns
        -------
        Tuple[tf.Tensor, tf.Tensor]
            A tuple of the training data and the target data.
        """
        # X train has 10 examples and 5 months of data
        X_train = tf.random.uniform(shape=(batch_size, num_months), minval=-5, maxval=5, dtype=tf.float32)
        # y train has 10 examples and 3 month of target data
        y_train = tf.random.uniform(shape=(batch_size, num_predictions), minval=-5, maxval=5, dtype=tf.float32)

        return X_train, y_train

    return _generate_data

class TestCustomGRUCell(object):
    """
    This class implements test for the custom GRU cell. Specifically, we
    test the saving and loading of a model that uses the custom GRU cell.
    """
    def test_save_load_model(self, generate_data):
        """
        Test that the model using the custom GRU cell can be saved and loaded properly.
        """
        # Data
        batch_size = 10
        num_months = 5
        num_predictions = 3
        X_train, y_train = generate_data(batch_size, num_months, num_predictions)

        # Model
        inputs = tf.keras.Input(shape=(num_months, 1))
        x = tf.keras.layers.RNN(
            cell=CustomGRUCell(layer_norm=tf.keras.layers.LayerNormalization(), units=2),
            return_sequences=True,
            name='custom_gru_1'
        )(inputs)
        x = tf.keras.layers.RNN(
            cell=CustomGRUCell(layer_norm=tf.keras.layers.LayerNormalization(), units=2),
            return_sequences=False,
            name='custom_gru_2'
        )(x)
        outputs = tf.keras.layers.Dense(units=num_predictions)(x)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)

        model.compile(optimizer='adam', loss=SymmetricMeanAbsolutePercentageError(reduction=tf.keras.losses.Reduction.AUTO, name='smape'))
        model.fit(x=X_train,y=y_train, epochs=1, batch_size=2)

        # Save and Load
        model_dir = '/tmp/0'
        model.save(model_dir)
        loaded_model = tf.keras.models.load_model(model_dir, custom_objects={'CustomGRUCell': CustomGRUCell, 'SymmetricMeanAbsolutePercentageError': SymmetricMeanAbsolutePercentageError})
        rmtree(model_dir)

        # Compare predictions of the model in memory and the loaded model
        assert tf.reduce_all(tf.equal(model.predict(X_train), loaded_model.predict(X_train)))