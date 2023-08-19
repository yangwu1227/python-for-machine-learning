import os
from typing import Tuple, Union, List, Dict, Any
import pickle
import boto3
import json
import logging
from functools import partial
import s3fs

import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Nopep8
import tensorflow as tf

from hydra import compose, initialize, core
from omegaconf import OmegaConf

from base_trainer import BaseTrainer

# ----------------------- Pretrained model instantiator ---------------------- #

def instantiate_pretrained_model(name: str, pooling: str) -> tf.keras.Model:
    """
    This function instantiates a pretrained model from the tf.keras.applications
    module based on the name provided.

    Parameters
    ----------
    name : str
        The name of the pretrained model to instantiate. The possible names are
        'vgg19', 'resnet50v2', and 'xception'.
    pooling : str
        The type of pooling to use.

    Returns
    -------
    tf.keras.engine.functional.Functional
        The pretrained model.

    Raises
    ------
    ValueError
        The 'pooling' is not one of 'avg', 'max', or 'none'.
    ValueError
        If the 'name' provided is not a valid pretrained model or one that is implemented.
    """
    name = name.lower()
    pooling = pooling.lower()
    if pooling not in ['none', 'avg', 'max']:
        raise ValueError("The argument 'pooling' must be one of 'none, 'avg', or 'max'")

    if name == 'vgg19':
        conv_base = tf.keras.applications.VGG19(
            include_top=False,
            weights='imagenet',
            pooling=pooling
        )
    elif name == 'resnet50v2':
        conv_base = tf.keras.applications.ResNet50V2(
            include_top=False,
            weights='imagenet',
            pooling=pooling
        )
    elif name == 'xception':
        conv_base = tf.keras.applications.Xception(
            include_top=False,
            weights='imagenet',
            pooling=pooling
        )
    else:
        raise ValueError(f'Not implemented pretrained model name: {name}')

    return conv_base

# ------------------------------- Trainer class ------------------------------ #

class FineTuneTrainer(BaseTrainer):
    """
    This class performs transfer learning using one of the three supported architectures as the convolutional base--- ResNet50V2, VGG19, and Xception. 
    The model is first trained with a three-dense-layer classifier at the top. Then, a few top convolutional layers are unfrozen and the model is trained 
    again with a reduced learning rate.
    """
    def __init__(self, 
                 hyperparameters: Dict[str, Any],
                 config: Dict[str, Any],
                 job_name: str,
                 train_dataset: tf.data.Dataset,
                 val_dataset: tf.data.Dataset,
                 train_class_weights: Dict[str, float],
                 distributed: bool,
                 strategy: tf.distribute.Strategy,
                 model_dir: str,
                 logger: logging.Logger) -> None:
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
            logger=logger
        )

    def _create_model(self) -> Dict[str, tf.keras.Model]:
        """
        Function that creates the compiled model.

        Returns
        -------
        Dict[str, tf.keras.Model]
            A dictionary containing the compiled model and the convolutional base.
        """
        # -------------------- Download and instantiate pretrained ------------------- #

        conv_base = instantiate_pretrained_model(
            name=self.hyperparameters['pretrained_name'],
            pooling=self.hyperparameters['pretrained_pooling']
        )
        conv_base.trainable = False

        # ---------------------------- Model architecture ---------------------------- #
        
        # Default dense layer
        DefaultDense = partial(
            tf.keras.layers.Dense,
            # We apply Relu after Batch Normalization
            activation='linear',
            # Since we use Batch Norm, we don't need to use bias terms for the Dense layers 
            use_bias=False,
            kernel_initializer='he_normal',
            kernel_regularizer=tf.keras.regularizers.l2(self.hyperparameters['dense_weight_decay'])
        )

        # Data augmentation layers
        data_augmentation = AugmentationModel(aug_params={
                'RandomFlip': {'mode': self.hyperparameters['random_flip_mode']},
                'RandomRotation': {'factor': self.hyperparameters['random_rotation_factor']},
                'RandomZoom': {'height_factor': self.hyperparameters['random_zoom_height_factor'], 'width_factor': self.hyperparameters['random_zoom_width_factor']},
                'RandomContrast': {'factor': self.hyperparameters['random_contrast_factor']}
        }).build_augmented_model()

        # Preprocessing mapping for each pretrained model
        preprocess_input_mapping = {
            'vgg19': tf.keras.applications.vgg19.preprocess_input,
            'resnet50v2': tf.keras.applications.resnet_v2.preprocess_input,
            'xception': tf.keras.applications.xception.preprocess_input
        }
        preprocess_input_func = preprocess_input_mapping.get(self.hyperparameters['pretrained_name'], lambda x: x)

        # Functional API
        inputs = tf.keras.Input(shape=(self.config['image_size'], self.config['image_size'], self.config['num_channels']), name='input_layer')
        x = data_augmentation(inputs)
        x = preprocess_input_func(x)
        # Use training=False in case pre-trained models contains batch normalization (inference-mode)
        x = conv_base(x, training=False)
        # Flatten the output of the pretrained model
        x = tf.keras.layers.Flatten()(x)

        # Classifier 
        for i in range(self.hyperparameters['dense_num_layers']):
            x = DefaultDense(self.hyperparameters[f'dense_units_{i}'], name=f'dense_{i}')(x)
            x = tf.keras.layers.BatchNormalization(name=f'dense_batch_norm_{i}')(x)
            x = tf.keras.layers.Dropout(self.hyperparameters['dense_dropout_rate'], name=f'dense_dropout_{i}')(x)
            x = tf.keras.layers.Activation('relu', name=f'dense_relu_{i}')(x)
            
        outputs = tf.keras.layers.Dense(units=self.config['num_classes'], activation='linear', name='output_layer')(x)
        model = tf.keras.Model(inputs, outputs)

        # ---------------------------------- Compile --------------------------------- #

        optimizer = self._create_optimizer(learning_rate=self.hyperparameters['adam_initial_lr'])
        loss_fn = self._create_loss_fn()
        metrics = self._create_metrics()
        model.compile(
            optimizer=optimizer,
            loss=loss_fn,
            metrics=metrics
        )

        return {'model': model, 'conv_base': conv_base}

    def _recompile_model(self, model: tf.keras.Model, conv_base: tf.keras.Model) -> tf.keras.Model:
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
        # For vgg19, fine-tune the last 4 convolutional layers of 'block5' (with 4 convolutional layers, max pooling and global pooling)
        if self.hyperparameters['pretrained_name'] == 'vgg19':
            for layer in conv_base.layers:
                if layer.name not in [f'block5_conv{i}' for i in range(1, 5)]:
                    layer.trainable = False
        # For resnet50v2, fine-tune the conv layers in the last residual block 'conv5_block3'
        elif self.hyperparameters['pretrained_name'] == 'resnet50v2':
            for layer in conv_base.layers:
                if layer.name not in [f'conv5_block3_{i}_conv' for i in [1, 2, 3]]:
                    layer.trainable = False
        # For xception, fine-tune the last 2 depthwise separable convolutional layers of 'block14'
        elif self.hyperparameters['pretrained_name'] == 'xception':
            for layer in conv_base.layers:
                if layer.name not in [f'block14_sepconv{i}' for i in [1, 2]]:
                    layer.trainable = False

        # --------------------------------- Recompile -------------------------------- #

        optimizer = self._create_optimizer(learning_rate=self.hyperparameters['adam_initial_lr'] / 10)
        loss_fn = self._create_loss_fn()
        metrics = self._create_metrics()
        model.compile(
            optimizer=optimizer,
            loss=loss_fn,
            metrics=metrics
        )

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
        trainable_params_count = np.sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
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
                model = model_conv_base['model']
                conv_base = model_conv_base['conv_base']
        else:
            model_conv_base = self._create_model()
            model = model_conv_base['model']
            conv_base = model_conv_base['conv_base']

        trainable_params_count = self._count_trainable_weights(model)
        self.logger.info(f'Number of trainable parameters for training classifier at the top: {trainable_params_count}')
        del trainable_params_count

        # ----------------------------- Train classifier ----------------------------- #

        # The 'on_train_begin' method resets the 'self.wait' attribute to 0 so this can be reused across multiple calls to 'fit'
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=self.config['patience'],
            mode='max',
            restore_best_weights=True
        )
        back_and_restore = tf.keras.callbacks.BackupAndRestore(
            backup_dir=os.path.join(os.getcwd(), 'backup'),
            # Delete the backup directory after the training is completed, so the next call to 'fit' will create a new backup directory
            delete_checkpoint=True
        )
        callbacks = [early_stopping, back_and_restore]

        model.fit(
            x=self.train_dataset,
            epochs=self.hyperparameters['fit_epochs'],
            validation_data=self.val_dataset,
            callbacks=callbacks,
            # Number of steps (batches of samples) to draw from before stopping validation
            validation_steps=self.hyperparameters['fit_validation_steps'],
            class_weight=self.train_class_weights
        )

        self.logger.info(f'Best validation loss after training classifier at the top: {early_stopping.best}')

        # -------------------------------- Fine-tuning ------------------------------- #

        # Recompile model with a reduced learning rate
        if self.distributed:
            with self.strategy.scope():
                model = self._recompile_model(model, conv_base)
        else:
            model = self._recompile_model(model, conv_base)

        trainable_params_count = self._count_trainable_weights(model)
        self.logger.info(f'Number of trainable parameters for fine-tuning: {trainable_params_count}')
        del trainable_params_count

        # Add Tensorboard callback
        if self.distributed:
            tensorboard = tf.keras.callbacks.TensorBoard(
                log_dir=f's3://{self.config["s3_bucket"]}/{self.config["s3_key"]}/tensorboard_logs/{self.job_name}'
            )
            callbacks.append(tensorboard)

        model.fit(
            x=self.train_dataset,
            epochs=self.hyperparameters['fit_epochs'],
            validation_data=self.val_dataset,
            callbacks=callbacks,
            # Number of steps (batches of samples) to draw from before stopping validation
            validation_steps=self.hyperparameters['fit_validation_steps'],
            class_weight=self.train_class_weights
        )

        self.logger.info(f'Best validation accuracy after fine-tuning: {early_stopping.best}')

        # -------------------------------- Save model -------------------------------- #

        if self.distributed:
            # For single-host multi-gpu training, there is no cluster resolver so we specify the type and id manually
            if self.strategy.cluster_resolver is None:
                model_dir = self._create_model_dir(
                    self.model_dir, 
                    'worker', 
                    0
                )
            else:
                # If the cluster resolver is not None, we are in multi-host training mode
                model_dir = self._create_model_dir(
                    self.model_dir, 
                    self.strategy.cluster_resolver.task_type, 
                    self.strategy.cluster_resolver.task_id
                )
            model.save(os.path.join(model_dir, '0'))
        else:
            model.save(os.path.join(self.model_dir, '0'))

        return None

if __name__ == '__main__':

    from custom_utils import get_logger, parser, add_additional_args, load_dataset, AugmentationModel

    # ---------------------------------- Set up ---------------------------------- #

    logger = get_logger(name='fine_tune')

    # Hydra
    core.global_hydra.GlobalHydra.instance().clear()
    initialize(version_base='1.2', config_path='config', job_name='fine_tune')
    config = OmegaConf.to_container(compose(config_name='main'), resolve=True)

    # Create a dictionary of name: type for the additional arguments
    additional_args = {
        # Data augmentation parameters
        'random_flip_mode': str,
        'random_rotation_factor': float,
        'random_contrast_factor': float,
        'random_zoom_height_factor': float,
        'random_zoom_width_factor': float,
        # Architecture parameters
        'dense_num_layers': int,
        'dense_units_0': int,
        'dense_units_1': int,
        'dense_units_2': int,
        'dense_weight_decay': float,
        'dense_dropout_rate': float,
        # Optimization, loss, and fit parameters
        'adam_initial_lr': float,
        'adam_beta_1': float,
        'adam_beta_2': float,
        'adam_clipnorm': float,
        'use_focal_loss': int,
        'loss_gamma': float,
        'loss_alpha': float,
        'fit_epochs': int,
        # Pretraining parameters
        'pretrained_name': str,
        'pretrained_pooling': str,
        # Distributed training parameters
        'distributed_multi_worker': int
    }

    args = add_additional_args(parser_func=parser, additional_args=additional_args)()

    job_name = args.training_env['job_name']

    # Strategy for distributed training
    if args.test_mode:
        distributed = False
        strategy = None
    else:
        distributed = True
        if args.distributed_multi_worker:
            strategy = tf.distribute.MultiWorkerMirroredStrategy()
        else:
            strategy = tf.distribute.MirroredStrategy()

    # --------------------------------- Load data -------------------------------- #

    if args.test_mode:

        train_dataset = load_dataset(
            dir=args.train,
            batch_size=config['batch_size']
        ).take(2)

        val_dataset = load_dataset(
            dir=args.val,
            batch_size=config['batch_size']
        ).take(2)

    else:
        tf_config = json.loads(os.environ['TF_CONFIG'])
        num_workers = len(tf_config['cluster']['worker'])
        global_batch_size = config['batch_size'] * num_workers

        train_dataset = load_dataset(
            dir=args.train,
            batch_size=global_batch_size
        )

        val_dataset = load_dataset(
            dir=args.val,
            batch_size=global_batch_size
        )

    fs = s3fs.S3FileSystem()
    with fs.open(f's3://{config["s3_bucket"]}/{config["s3_key"]}/input-data/train_weights.json', 'rb') as f:
        train_class_weights = json.load(f)
    # Convert all keys to integers
    train_class_weights = {int(k): v for k, v in train_class_weights.items()}

    # --------------------------------- Train model --------------------------------- #

    trainer = FineTuneTrainer(
        hyperparameters={
            # Data augmentation parameters
            'random_flip_mode': args.random_flip_mode,
            'random_rotation_factor': args.random_rotation_factor,
            'random_contrast_factor': args.random_contrast_factor,
            'random_zoom_height_factor': args.random_zoom_height_factor,
            'random_zoom_width_factor': args.random_zoom_width_factor,
            # Architecture parameters
            'dense_num_layers': args.dense_num_layers,
            'dense_units_0': args.dense_units_0,
            'dense_units_1': args.dense_units_1,
            'dense_units_2': args.dense_units_2,
            'dense_weight_decay': args.dense_weight_decay,
            'dense_dropout_rate': args.dense_dropout_rate,
            # Optimization, loss, and fit parameters
            'adam_initial_lr': args.adam_initial_lr,
            'adam_beta_1': args.adam_beta_1,
            'adam_beta_2': args.adam_beta_2,
            'adam_clipnorm': args.adam_clipnorm,
            'use_focal_loss': args.use_focal_loss,
            'loss_gamma': args.loss_gamma,
            'loss_alpha': args.loss_alpha,
            'fit_epochs': args.fit_epochs,
            'fit_validation_steps': 1 if args.test_mode else len(val_dataset),
            # Pretraining parameters
            'pretrained_name': args.pretrained_name,
            'pretrained_pooling': args.pretrained_pooling
        },
        config=config,
        job_name=job_name,
        train_dataset=train_dataset,
        val_dataset=val_dataset.repeat(),
        train_class_weights=train_class_weights,
        distributed=distributed,
        strategy=strategy,
        model_dir=args.model_dir,
        logger=logger
    )

    trainer.fit()

    del trainer