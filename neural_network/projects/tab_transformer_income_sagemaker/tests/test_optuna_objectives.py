import sys
import os

from hydra import compose, initialize
from omegaconf import OmegaConf
import pytest
import numpy as np
import pandas as pd
import optuna
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Nopep8
import tensorflow as tf

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.tf_keras_entry import tf_objective
from src.custom_utils import get_logger, dataset_from_csv

# --------------------------- Module level fixtures -------------------------- #

@pytest.fixture(scope='module')
def config():
    # Load config file relative to the tests directory
    initialize(version_base='1.2', config_path='../src/config', job_name='test_optuna_objectives')
    config = OmegaConf.to_container(compose(config_name='main'), resolve=True)
    return config

@pytest.fixture(scope='module')
def logger():
    return get_logger(name='test_optuna_objectives')

# ----------------------------------- Tests ---------------------------------- #

@pytest.fixture(scope='class')
def fixed_params():
    return optuna.trial.FixedTrial({
        'num_transformer_blocks': 3,
        'num_heads': 2,
        'embedding_dims': 16,
        'mlp_num_hidden_layers': 2,
        'mlp_hidden_units_factor_0': 2,
        'mlp_hidden_units_factor_1': 2,
        'dropout_rate': 0.5,
        'use_column_embedding': True,
        'optimizer': 'adam',
        'adam_learning_rate': 1e-3,
        'adam_beta_1': 0.9,
        'adam_beta_2': 0.999,
        'adam_epsilon': 1e-8,
        'adam_clipnorm': 1.0,
        'loss_apply_class_balancing': True,
        'loss_alpha': 0.5,
        'loss_gamma': 1.0,
        'fit_epochs': 1
    })

class TestTFObjective:
    """
    Tests for the optuna objective function for the Keras implementation of the TabTransformer.
    """
    def test_returns_float(self, config, logger, fixed_params):
        """
        Tests that the objective function returns a float.
        """
        train = dataset_from_csv(
            file_path='tests/test_data/train.csv',
            config=config,
            train=True
        )
        val = dataset_from_csv(
            file_path='tests/test_data/val.csv',
            config=config,
            train=False
        )

        def tf_objective_wrapper(trial: optuna.Trial):
            return tf_objective(
                trial=trial,
                train_data=train,
                val_data=train,
                strategy=None,
                config=config,
                logger=logger,
                test_mode=1,
                epsilon=1e-16,
                job_name='test_tf_objective',
                verbose=0
            )

        objective_value = tf_objective_wrapper(trial=fixed_params)

        # Check that the objective value is a float
        assert isinstance(objective_value, float)