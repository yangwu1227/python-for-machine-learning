import argparse
import json
import logging
import os
from typing import Any, Dict, List, Tuple

import numpy as np
import s3fs
from hydra import compose, core, initialize
from omegaconf import OmegaConf

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Nopep8
import tensorflow as tf
from base_trainer import BaseTrainer

# ------------------------------- Trainer class ------------------------------ #


class FineTuneTrainer(BaseTrainer):
    """
    This class performs transfer learning using the ResNet50V2 architecture as the convolutional base. The model is first
    trained with a single dense classifier at the top. Then, a few top convolutional layers are unfrozen and the model is
    trained again with a reduced learning rate.
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
        Constructor for the FineTuneTrainer class.

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

    def _create_model(self) -> Dict[str, tf.keras.Model]:
        """
        Function that creates the compiled model.

        Returns
        -------
        Dict[str, tf.keras.Model]
            A dictionary containing the compiled model and the convolutional base.
        """
        # Convolutional base
        conv_base = tf.keras.applications.ResNet50V2(
            include_top=False,
            weights="imagenet",
            pooling=self.hyperparameters["pooling"],
        )
        conv_base.trainable = False

        # ---------------------------- Model architecture ---------------------------- #

        # Data augmentation
        data_augmentation = AugmentationModel(
            aug_params={
                "RandomRotation": {"factor": 0.5},
                "RandomContrast": {"factor": 0.3},
                "RandomFlip": {"mode": "horizontal_and_vertical"},
            }
        ).build_augmented_model()

        inputs = tf.keras.Input(
            shape=(self.config["image_size"], self.config["image_size"], 3),
            name="input_layer",
        )
        x = data_augmentation(inputs)
        # This scales the pixel values to the range [-1, 1]
        x = tf.keras.applications.resnet_v2.preprocess_input(x)
        # This includes a 'global' pooling layer
        x = conv_base(x, training=False)
        # Regularization
        x = tf.keras.layers.Dropout(rate=self.hyperparameters["dense_dropout_rate"])(x)

        # Classifier
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

        return {"model": model, "conv_base": conv_base}

    def _recompile_model(
        self, model: tf.keras.Model, conv_base: tf.keras.Model
    ) -> tf.keras.Model:
        """
        Function that recompiles the model with a reduced learning rate.

        Parameters
        ----------
        model : tf.keras.Model
            A tf.keras.Model object.
        conv_base : tf.keras.Model
            A tf.keras.Model object that contains the convolutional base.

        Returns
        -------
        tf.keras.Model
            A tf.keras.Model object that has been recompiled.
        """
        # ----------------- Release layers in the convolutional base ----------------- #

        conv_base.trainable = True
        # Everything before the last three conv2d layers (conv5 block3) are frozen
        for layer in conv_base.layers:
            if layer.name == "conv5_block3_preact_bn":
                break
            layer.trainable = False

        # --------------------------------- Recompile -------------------------------- #

        optimizer = self._create_optimizer(
            learning_rate=self.hyperparameters["opt_learning_rate"] / 10
        )
        loss_fn = self._create_loss_fn()
        metrics = self._create_metrics()
        model.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics)

        return model

    def _count_trainable_weights(self, model: tf.keras.Model) -> int:
        """
        Function that counts the number of trainable weights in a model.

        Parameters
        ----------
        model : tf.keras.Model
            A tf.keras.Model object.

        Returns
        -------
        int
            The number of trainable weights in the model.
        """
        trainable_params_count = np.sum(
            [tf.keras.backend.count_params(w) for w in model.trainable_weights]
        )
        return trainable_params_count

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
                model_conv_base = self._create_model()
                model = model_conv_base["model"]
                conv_base = model_conv_base["conv_base"]
        else:
            model_conv_base = self._create_model()
            model = model_conv_base["model"]
            conv_base = model_conv_base["conv_base"]

        trainable_params_count = self._count_trainable_weights(model)
        self.logger.info(
            f"Number of trainable parameters for training classifier at the top: {trainable_params_count}"
        )
        del trainable_params_count

        # ----------------------------- Train classifier ----------------------------- #

        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=3, mode="min", restore_best_weights=True
        )
        back_and_restore = tf.keras.callbacks.BackupAndRestore(
            backup_dir=os.path.join(os.getcwd(), "backup")
        )
        callbacks = [early_stopping, back_and_restore]

        model.fit(
            x=self.train_dataset,
            epochs=self.hyperparameters["fit_epochs"],
            validation_data=self.val_dataset,
            callbacks=callbacks,
            # Number of steps (batches of samples) to draw from before stopping validation
            validation_steps=self.hyperparameters["fit_validation_steps"],
            class_weight=self.train_class_weights,
        )

        self.logger.info(
            f"Best validation loss after training classifier at the top: {early_stopping.best}"
        )

        # -------------------------------- Fine-tuning ------------------------------- #

        # Recompile model with a reduced learning rate
        if self.distributed:
            with self.strategy.scope():
                model = self._recompile_model(model, conv_base)
        else:
            model = self._recompile_model(model, conv_base)

        trainable_params_count = self._count_trainable_weights(model)
        self.logger.info(
            f"Number of trainable parameters for fine-tuning: {trainable_params_count}"
        )
        del trainable_params_count

        # Add Tensorboard callback
        if self.distributed:
            tensorboard = tf.keras.callbacks.TensorBoard(
                log_dir=f's3://{self.config["s3_bucket"]}/{self.config["s3_key"]}/tensorboard_logs/{self.job_name}'
            )
            callbacks.append(tensorboard)

        model.fit(
            x=self.train_dataset,
            epochs=self.hyperparameters["fit_epochs"],
            validation_data=self.val_dataset,
            callbacks=callbacks,
            # Number of steps (batches of samples) to draw from before stopping validation
            validation_steps=self.hyperparameters["fit_validation_steps"],
            class_weight=self.train_class_weights,
        )

        self.logger.info(
            f"Best validation loss after fine-tuning: {early_stopping.best}"
        )

        # -------------------------------- Save model -------------------------------- #

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

    logger = get_logger(name="fine_tune_resnet50v2")

    # Hyra
    core.global_hydra.GlobalHydra.instance().clear()
    initialize(
        version_base="1.2", config_path="config", job_name="fine_tune_resnet50v2"
    )
    config = OmegaConf.to_container(compose(config_name="main"), resolve=True)

    # Parser hyperparameters specified by the SageMaker
    additional_args = {
        "loss_alpha": float,
        "loss_gamma": float,
        "pooling": str,
        "dense_dropout_rate": float,
        "opt_learning_rate": float,
        "opt_adam_beta_1": float,
        "opt_adam_beta_2": float,
        "opt_clipnorm": float,
        "fit_epochs": int,
        "use_focal_loss": int,
    }

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

    trainer = FineTuneTrainer(
        hyperparameters={
            "dense_dropout_rate": args.dense_dropout_rate,
            "pooling": args.pooling,
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
