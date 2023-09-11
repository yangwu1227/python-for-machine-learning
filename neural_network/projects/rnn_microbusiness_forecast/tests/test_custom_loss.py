import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Nopep8
from typing import Dict, Union, Callable, Tuple
import torch
from torchmetrics.regression import SymmetricMeanAbsolutePercentageError as smape_torch
import tensorflow as tf
from numpy import isclose
from src.gru_entry import SymmetricMeanAbsolutePercentageError
import pytest

@pytest.fixture(scope='class')
def generate_data():
    """
    Fixture factory returning a function that generates `y_true` and `y_pred` tensors for testing.
    """
    def _generate_data(batch_size: int, num_predictions: int) -> Dict[str, Union[Tuple[tf.Tensor, tf.Tensor], Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Generate `y_true` and `y_pred` tensors for testing based on the given batch size and number of predictions.

        Parameters
        ----------
        batch_size : int
            The batch size.
        num_predictions : int
            The number of predictions.

        Returns
        -------
        Dict[str, Union[Tuple[tf.Tensor, tf.Tensor], Tuple[torch.Tensor, torch.Tensor]]]
            A dictionary containing the `y_true` and `y_pred` tensors for both TensorFlow and PyTorch.
        """
        y_true_tf = tf.random.uniform(shape=(batch_size, num_predictions), minval=-5, maxval=5, dtype=tf.float32)
        y_pred_tf = tf.random.uniform(shape=(batch_size, num_predictions), minval=-5, maxval=5, dtype=tf.float32)

        # Randomly set some values to 0
        y_true_tf = tf.where(y_true_tf > 0.5, y_true_tf, 0)
        y_pred_tf = tf.where(y_pred_tf > 0.5, y_pred_tf, 0)

        y_true_torch = torch.from_numpy(y_true_tf.numpy())
        y_pred_torch = torch.from_numpy(y_pred_tf.numpy())

        return {'tf': (y_true_tf, y_pred_tf), 'torch': (y_true_torch, y_pred_torch)}

    return _generate_data

class TestCustomLoss(object):

    def test_init(self):
        """
        Test the constructor of the SymmetricMeanAbsolutePercentageError class.
        """
        loss_fn = SymmetricMeanAbsolutePercentageError(reduction='sum_over_batch_size', name='smape')
        assert isinstance(loss_fn, SymmetricMeanAbsolutePercentageError)
        assert loss_fn.reduction == 'sum_over_batch_size'
        assert loss_fn.name == 'smape'

    def test_get_config(self):
        """
        Test that the `get_config` method gets all the necessary attributes of the
        SymmetricMeanAbsolutePercentageError class.
        """
        loss_fn = SymmetricMeanAbsolutePercentageError(reduction='sum_over_batch_size', name='smape')
        config = loss_fn.get_config()
        assert isinstance(config, dict)
        assert config['reduction'] == 'sum_over_batch_size'
        assert config['name'] == 'smape'

    def test_from_config(self, config: Dict[str, str] = {'reduction': 'sum', 'name': 'smap'}):
        """
        Test that the `from_config` method gets all the necessary attributes of the
        SymmetricMeanAbsolutePercentageError class.
        """
        loss_fn = SymmetricMeanAbsolutePercentageError.from_config(config)
        assert isinstance(loss_fn, SymmetricMeanAbsolutePercentageError)
        assert loss_fn.reduction == 'sum'
        assert loss_fn.name == 'smap'

    @pytest.mark.parametrize(
        'batch_size, num_predictions',
        [
            (5, 10),
            (10, 5),
            (5, 5),
            (10, 10)
        ],
        scope='function'
    )
    def test_call(self, batch_size, num_predictions, generate_data):
        """
        Test the custom loss function against a reference implementation from TorchMetrics.
        """
        data = generate_data(batch_size, num_predictions)
        y_true_tf, y_pred_tf = data['tf']
        y_true_torch, y_pred_torch = data['torch']

        loss_fn_tf = SymmetricMeanAbsolutePercentageError(reduction='sum_over_batch_size', name='smape')
        loss_fn_torch = smape_torch()

        # Compute the losses
        loss_tf = loss_fn_tf(y_true_tf, y_pred_tf)
        loss_torch = loss_fn_torch(y_pred_torch, y_true_torch)

        # Pytorch's implementation divides the sum of all instance losses by (batch_size * num_predictions)
        sum_of_instance_losses_torch = (loss_torch * (y_true_torch.shape[0] * y_true_torch.shape[1])).detach().numpy().item()
        # Our custom loss function divides the sum of all instance losses by batch_size
        sum_of_instance_losses_tf = (loss_tf * y_true_tf.shape[0]).numpy()

        # Check that the losses are close
        assert isclose(sum_of_instance_losses_torch, sum_of_instance_losses_tf, rtol=1e-5)