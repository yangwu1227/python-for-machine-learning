import os
import sys

import optuna
import pytest
from hydra import compose, initialize
from omegaconf import OmegaConf

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Nopep8

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.custom_utils import dataset_from_csv, get_logger
from src.tf_keras_entry import tf_objective

# --------------------------- Module level fixtures -------------------------- #


@pytest.fixture(scope="module")
def config():
    # Load config file relative to the tests directory
    initialize(
        version_base="1.2",
        config_path="../src/config",
        job_name="test_optuna_objectives",
    )
    config = OmegaConf.to_container(compose(config_name="main"), resolve=True)
    return config


@pytest.fixture(scope="module")
def logger():
    return get_logger(name="test_optuna_objectives")


# ----------------------------------- Tests ---------------------------------- #


@pytest.fixture(scope="class")
def fixed_params():
    return optuna.trial.FixedTrial(
        {
            "transformer_num_layers": 3,
            "transformer_num_heads": 2,
            "transformer_embedding_dims": 32,
            "transformer_dropout_rate": 0.2,
            "mlp_num_hidden_layers": 1,
            "mlp_hidden_units_multiple_0": 2,
            "mlp_dropout_rate": 0.2,
            "use_focal_loss": True,
            "loss_apply_class_balancing": True,
            "loss_alpha": 0.5,
            "loss_gamma": 1.0,
            "adam_learning_rate": 0.001,
            "adam_beta_1": 0.9,
            "adam_beta_2": 0.999,
            "adam_epsilon": 1e-7,
            "adam_clipnorm": 1.0,
            "fit_epochs": 10,
            "fit_validation_steps": 2,
        }
    )


class TestTFObjective:
    """
    Tests for the optuna objective function for the Keras implementation of the TabTransformer.
    """

    def test_returns_float(self, config, logger, fixed_params):
        """
        Tests that the objective function returns a float.
        """
        train_num_batches, train_dataset = dataset_from_csv(
            file_path="tests/test_data/train.csv",
            config=config,
            train=True,
            batch_size=config["tf_keras"]["batch_size"],
        )
        val_num_batches, val_dataset = dataset_from_csv(
            file_path="tests/test_data/val.csv",
            config=config,
            train=False,
            batch_size=config["tf_keras"]["batch_size"],
        )

        def tf_objective_wrapper(trial: optuna.Trial):
            return tf_objective(
                trial=trial,
                config=config,
                job_name="test_tf_objective",
                train_dataset=train_dataset,
                train_num_batches=train_num_batches,
                val_dataset=val_dataset,
                val_num_batches=val_num_batches,
                distributed=False,
                strategy=None,
                model_dir=None,
                logger=logger,
            )

        objective_value = tf_objective_wrapper(trial=fixed_params)

        # Check that the objective value is a float
        assert isinstance(objective_value, float)
