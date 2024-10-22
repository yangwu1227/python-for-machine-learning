import argparse
import json
import logging
import os
from functools import partial
from itertools import chain
from typing import Any, Dict, List, Tuple

import numpy as np
import s3fs
from hydra import compose, core, initialize
from omegaconf import OmegaConf

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Nopep8
import tensorflow as tf
from base_trainer import BaseTrainer

# ------------------------------- Trainer class ------------------------------ #


class BaselineTrainer(BaseTrainer):
    """
    This class is used to train an image classification model.
    """

    def __init__(
        self,
        hyperparameters: Dict[str, Any],
        config: Dict[str, Any],
        job_name: str,
        train_dataset: tf.data.Dataset,
        val_dataset: tf.data.Dataset,
        train_class_weights: Dict[str, float],
        distributed: bool,
        strategy: tf.distribute.Strategy,
        model_dir: str,
        logger: logging.Logger,
    ) -> None:
        """
        Constructor for the BaselineTrainer class.

        Parameters
        ----------
        hyperparameters : Dict[str, Any]
            A dictionary containing the hyperparameters for model training.
        config : Dict[str, Any]
            A dictionary containing the configuration for model training.
        job_name : str
            The name of the job.
        train_dataset : tf.data.Dataset
            A tf.data.Dataset object that contains the training data.
        val_dataset : tf.data.Dataset
            The validation data is recommend to be a repeated dataset.
        train_class_weights : Dict[str, float]
            Class weights for the training data.
        distributed : bool
            A boolean that specifies whether to use distributed training.
        strategy : tf.distribute.Strategy
            A tf.distribute.Strategy object that specifies the strategy for distributed training.
        model_dir : str
            Path to the directory where the model will be saved.
        logger : logging.Logger
            A logger object.

        Returns
        -------
        None
        """
        super().__init__(
            hyperparameters=hyperparameters,
            config=config,
            job_name=job_name,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            train_class_weights=train_class_weights,
            distributed=distributed,
            strategy=strategy,
            model_dir=model_dir,
            logger=logger,
        )

    def _create_model(self) -> tf.keras.Model:
        """
        Function that creates the compiled model.

        Returns
        -------
        tf.keras.Model
            The compiled model.
        """
        # Default convolutional layer
        DefaultConv2D = partial(
            tf.keras.layers.Conv2D,
            kernel_size=self.hyperparameters["conv2d_kernel_size"],
            padding="same",
            activation="linear",
            use_bias=False,  # Not needed if batch normalization is used
            kernel_initializer="he_normal",
            kernel_regularizer=tf.keras.regularizers.l2(),
        )
        # Default dense layer
        DefaultDense = partial(
            tf.keras.layers.Dense,
            activation="linear",
            use_bias=False,
            kernel_initializer="he_normal",
            kernel_regularizer=tf.keras.regularizers.l2(),
        )

        # ---------------------------- Model architecture ---------------------------- #

        # Data augmentation
        data_augmentation = AugmentationModel(
            aug_params={
                "RandomRotation": {"factor": 0.5},
                "RandomContrast": {"factor": 0.3},
            }
        ).build_augmented_model()

        inputs = tf.keras.Input(
            shape=(
                self.config["image_size"],
                self.config["image_size"],
                self.config["num_channels"],
            ),
            name="input_layer",
        )
        x = data_augmentation(inputs)
        x = tf.keras.layers.Rescaling(scale=1.0 / 255.0, name="rescaling_layer")(x)

        for i in range(5):
            x = DefaultConv2D(
                filters=self.hyperparameters[f"conv2d_num_filters_block_{i}"],
                name=f"conv2d_{i}",
            )(x)
            x = tf.keras.layers.BatchNormalization(name=f"conv2d_batch_norm_{i}")(x)
            x = tf.keras.layers.Activation("relu", name=f"conv2d_relu_{i}")(x)
            x = tf.keras.layers.MaxPooling2D(
                pool_size=self.hyperparameters["conv2d_pooling_size"],
                name=f"conv2d_pooling_{i}",
            )(x)

        x = tf.keras.layers.Flatten(name="flatten_layer")(x)

        for i in range(3):
            x = DefaultDense(
                units=self.hyperparameters[f"dense_num_units_{i}"], name=f"dense_{i}"
            )(x)
            x = tf.keras.layers.BatchNormalization(name=f"dense_batch_norm_{i}")(x)
            # Dropout before activation is the same as after for 'RELU' based on https://sebastianraschka.com/faq/docs/dropout-activation.html
            x = tf.keras.layers.Dropout(
                rate=self.hyperparameters["dense_dropout_rate"],
                name=f"dense_drop_out_{i}",
            )(x)
            x = tf.keras.layers.Activation("relu", name=f"dense_relu_{i}")(x)

        outputs = tf.keras.layers.Dense(
            units=self.config["num_classes"], activation="linear", name="output_layer"
        )(x)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)

        # ---------------------------------- Compile --------------------------------- #

        optimizer = self._create_optimizer(
            learning_rate=self.hyperparameters["opt_learning_rate"]
        )
        loss_fn = self._create_loss_fn()
        metrics = self._create_metrics()
        model.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics)

        return model

    def fit(self) -> None:
        """
        Function that fits the models.

        Returns
        -------
        None
        """
        # ------------------------------- Create model ------------------------------- #

        if self.distributed:
            with self.strategy.scope():
                model = self._create_model()
        else:
            model = self._create_model()

        # --------------------------------- Callbacks -------------------------------- #

        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=3, mode="min", restore_best_weights=True
        )
        back_and_restore = tf.keras.callbacks.BackupAndRestore(
            backup_dir=os.path.join(os.getcwd(), "backup")
        )
        callbacks = [early_stopping, back_and_restore]

        if self.distributed:
            tensorboard = tf.keras.callbacks.TensorBoard(
                log_dir=f's3://{self.config["s3_bucket"]}/{self.config["s3_key"]}/tensorboard_logs/{self.job_name}'
            )
            callbacks.append(tensorboard)

        # ------------------------------------ Fit ----------------------------------- #

        model.fit(
            x=self.train_dataset,
            epochs=self.hyperparameters["fit_epochs"],
            validation_data=self.val_dataset,
            callbacks=callbacks,
            # Number of steps (batches of samples) to draw from before stopping validation
            validation_steps=self.hyperparameters["fit_validation_steps"],
            class_weight=self.train_class_weights,
        )

        logger.info(f"Best validation loss: {early_stopping.best}")

        # ---------------------------------- Save model --------------------------------- #

        if self.distributed:
            model_dir = self._create_model_dir(
                self.model_dir,
                self.strategy.cluster_resolver.task_type,
                self.strategy.cluster_resolver.task_id,
            )
            model.save(os.path.join(model_dir, "00000000"))
        else:
            model.save(os.path.join(self.model_dir, "00000000"))

        return None


if __name__ == "__main__":
    from custom_utils import (
        AugmentationModel,
        add_additional_args,
        get_logger,
        load_datasets,
        parser,
    )

    # ---------------------------------- Set up ---------------------------------- #

    logger = get_logger(name="baseline_training")

    # Hyra
    core.global_hydra.GlobalHydra.instance().clear()
    initialize(version_base="1.2", config_path="config", job_name="baseline_training")
    config = OmegaConf.to_container(compose(config_name="main"), resolve=True)

    # Parser hyperparameters specified by the SageMaker
    filters = {f"conv2d_num_filters_block_{i}": int for i in range(0, 5)}
    dense_layer_units = {f"dense_num_units_{i}": int for i in range(0, 3)}
    loss_hyperparams = {"loss_alpha": float, "loss_gamma": float}
    other_hyperparams = {
        "conv2d_pooling_size": int,
        "conv2d_kernel_size": int,
        "dense_dropout_rate": float,
        "opt_learning_rate": float,
        "opt_adam_beta_1": float,
        "opt_adam_beta_2": float,
        "opt_clipnorm": float,
        "fit_epochs": int,
        "use_focal_loss": int,
    }
    additional_args = dict(
        chain(
            filters.items(),
            dense_layer_units.items(),
            loss_hyperparams.items(),
            other_hyperparams.items(),
        )
    )

    args = add_additional_args(parser_func=parser, additional_args=additional_args)()

    job_name = args.training_env["job_name"]

    # --------------------------------- Load data -------------------------------- #

    if args.test_mode:
        distributed = False
        strategy = None
    else:
        distributed = True
        strategy = tf.distribute.MultiWorkerMirroredStrategy()

    if not distributed:
        # Sample three batches from the training dataset
        train_dataset = load_datasets(
            dir=args.train, batch_size=config["batch_size"], val=False
        ).take(3)

        # Sample three batches from the validation dataset
        val_dataset = load_datasets(
            dir=args.val, batch_size=config["batch_size"], val=True
        ).take(3)

    else:
        tf_config = json.loads(os.environ["TF_CONFIG"])
        num_workers = len(tf_config["cluster"]["worker"])
        global_batch_size = config["batch_size"] * num_workers

        # Load the training dataset
        train_dataset = load_datasets(
            dir=args.train, batch_size=global_batch_size, val=False
        )

        # Load the validation dataset
        val_dataset = load_datasets(
            dir=args.val, batch_size=global_batch_size, val=True
        )

    # Load training set weights
    fs = s3fs.S3FileSystem()

    with fs.open(
        f's3://{config["s3_bucket"]}/{config["s3_key"]}/input-data/train_weights.json',
        "rb",
    ) as f:
        train_class_weights = json.load(f)
    # Convert all keys to integers
    train_class_weights = {int(k): v for k, v in train_class_weights.items()}

    # --------------------------------- Train model --------------------------------- #

    trainer = BaselineTrainer(
        hyperparameters={
            "conv2d_num_filters_block_0": args.conv2d_num_filters_block_0,
            "conv2d_num_filters_block_1": args.conv2d_num_filters_block_1,
            "conv2d_num_filters_block_2": args.conv2d_num_filters_block_2,
            "conv2d_num_filters_block_3": args.conv2d_num_filters_block_3,
            "conv2d_num_filters_block_4": args.conv2d_num_filters_block_4,
            "conv2d_pooling_size": args.conv2d_pooling_size,
            "conv2d_kernel_size": args.conv2d_kernel_size,
            "dense_num_units_0": args.dense_num_units_0,
            "dense_num_units_1": args.dense_num_units_1,
            "dense_num_units_2": args.dense_num_units_2,
            "dense_dropout_rate": args.dense_dropout_rate,
            "opt_learning_rate": args.opt_learning_rate,
            "opt_adam_beta_1": args.opt_adam_beta_1,
            "opt_adam_beta_2": args.opt_adam_beta_2,
            "opt_clipnorm": args.opt_clipnorm,
            "loss_alpha": args.loss_alpha,
            "loss_gamma": args.loss_gamma,
            "fit_epochs": args.fit_epochs,
            "fit_validation_steps": (
                1 if args.test_mode else int(config["val_size"] / config["batch_size"])
            ),
            "use_focal_loss": args.use_focal_loss,
        },
        config=config,
        job_name=job_name,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        train_class_weights=train_class_weights,
        distributed=distributed,
        strategy=strategy,
        model_dir=args.model_dir,
        logger=logger,
    )

    trainer.fit()

    del trainer
