import os
from typing import Tuple, Union, List, Dict, Any, Callable, Iterable
import pickle
import logging
from functools import partial
import argparse

import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Nopep8
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight

import numpy as np

from hydra import compose, initialize
from custom_utils import get_logger, parser, add_additional_args, load_data, AugmentationModel

# ------------------------------- Baseline CNN ------------------------------- #

def train_baseline_cnn(
    train_data: Dict[str, np.ndarray],
    val_data: Dict[str, np.ndarray],
    conv2d_params: Dict[str, Any],
    dense_params: Dict[str, Any],
    aug_params: Dict[str, Any],
    opt_params: Dict[str, Any],
    fit_params: Dict[str, Any],
    num_classes: int,
    input_shape: Tuple[int, int]
    ) -> Tuple[tf.keras.Model, tf.keras.callbacks.EarlyStopping]:
    """
    Build, compile, and train a convolutional neural network model for 
    multiclass classification.

    Parameters
    ----------
    train_data : Dict[str, np.ndarray]
        A dictionary containing the training data--- X_train and y_train.
    val_data : Dict[str, np.ndarray]
        A dictionary containing the validation data--- X_val and y_val.
    conv2d_params : Dict[str, Any]
        Parameters for the convolutional layers.
    dense_params : Dict[str, Any]
        Parameters for the dense layers.
    aug_params : Dict[str, Any]
        Parameters for the image augmentation layers.
    opt_params : Dict[str, Any]
        Parameters for the optimizer.
    fit_params : Dict[str, Any]
        Parameters for the fit method.
    num_classes : int
        The number of classes in the dataset.
    input_shape : Tuple[int, int]
        The shape of the input data.
        
    Returns
    -------
    Tuple[tf.keras.Model, tf.keras.callbacks.EarlyStopping]
        A tuple containing the trained model and the early stopping callback.
    """
    # Default convolutional layer
    DefaultConv2D = partial(
        tf.keras.layers.Conv2D,
        kernel_size=3,
        activation='relu',
        padding='same',
        kernel_initializer='he_normal',
        kernel_regularizer=tf.keras.regularizers.l2(conv2d_params['conv2d_weight_decay'])
    )

    # Default dense layer
    DefaultDense = partial(
        tf.keras.layers.Dense,
        activation='relu',
        kernel_initializer='he_normal',
        kernel_regularizer=tf.keras.regularizers.l2(dense_params['dense_weight_decay'])
    )

    # Data augmentation layers
    data_augmentation = AugmentationModel(aug_params={
            'RandomFlip': {'mode': aug_params['random_flip_mode']},
            'RandomRotation': {'factor': aug_params['random_rotation_factor']},
            'RandomZoom': {'height_factor': aug_params['random_zoom_height_factor'], 'width_factor': aug_params['random_zoom_width_factor']},
            'RandomContrast': {'factor': aug_params['random_contrast_factor']}
    }).build_augmented_model()

    # ---------------------------- Model architecture ---------------------------- #

    inputs = tf.keras.Input(shape=input_shape)
    x = data_augmentation(inputs)
    x = tf.keras.layers.Rescaling(1./255.0)(x)

    for i in range(5):
        x = DefaultConv2D(conv2d_params['conv2d_filters_list'][i])(x)
        # Max pooling first then batch normalization
        x = tf.keras.layers.MaxPool2D()(x)
        x = tf.keras.layers.BatchNormalization(
            momentum=conv2d_params['conv2d_batch_norm_momentum'],
        )(x)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dropout(dense_params['dropout_rate'])(x)

    for i in range(2):
        x = DefaultDense(dense_params['dense_units_list'][i])(x)
        x = tf.keras.layers.BatchNormalization(
            momentum=dense_params['dense_batch_norm_momentum']
        )(x)

    outputs = tf.keras.layers.Dense(units=num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        optimizer=tf.keras.optimizers.SGD(
            learning_rate=tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=opt_params['sgd_initial_lr'],
                decay_steps=opt_params['sgd_lr_decay_steps'],
                decay_rate=opt_params['sgd_lr_decay_rate']
            ),
            momentum=opt_params['sgd_momentum'],
            clipnorm=opt_params['sgd_clipnorm']
        ),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')]
    )

    # ---------------------------- Model training ---------------------------- #

    early_stopper = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        restore_best_weights=True
    )

    sample_weights = compute_sample_weight(
        class_weight='balanced',
        y=train_data['y_train']
    )

    model.fit(
        x=train_data['X_train'],
        y=train_data['y_train'],
        batch_size=fit_params['batch_size'],
        epochs=fit_params['epochs'],
        verbose=2,
        callbacks=[early_stopper],
        validation_data=(val_data['X_val'], val_data['y_val']),
        # If both sample_weight and class_weight are provided, the weights are multiplied as per https://github.com/keras-team/keras/blob/master/keras/engine/training_utils_v1.py
        class_weight=None,
        sample_weight=sample_weights if fit_params['use_sample_weights'] else None
    )

    return model, early_stopper

if __name__ == '__main__':

    # -------------------------- Compose configurations -------------------------- #

    initialize(version_base='1.2', config_path='config', job_name='baseline')
    config = compose(config_name='main')

    # ---------------------------------- Set up ---------------------------------- #

    logger = get_logger(__name__)

    # Create a dictionary of name: type for the additional arguments
    additional_args = {
        # Data augmentation parameters
        'random_flip_mode': str,
        'random_rotation_factor': float,
        'random_contrast_factor': float,
        'random_zoom_height_factor': float,
        'random_zoom_width_factor': float,
        # Architecture parameters
        'conv2d_filters_1': int,
        'conv2d_filters_2': int,
        'conv2d_filters_3': int,
        'conv2d_filters_4': int,
        'conv2d_filters_5': int,
        'conv2d_weight_decay': float,
        'conv2d_batch_norm_momentum': float,
        'dense_units_1': int,
        'dense_units_2': int,
        'dense_weight_decay': float,
        'dense_batch_norm_momentum': float,
        'dropout_rate': float,
        # Optimization and fit parameters
        'sgd_initial_lr': float,
        'sgd_lr_decay_steps': int,
        'sgd_lr_decay_rate': float,
        'sgd_momentum': float,
        'sgd_clipnorm': float,
        'batch_size': int,
        'epochs': int,
        'use_sample_weights': int
    }

    args = add_additional_args(parser_func=parser, additional_args=additional_args)()

    # --------------------------------- Load data -------------------------------- #

    logger.info('Loading data...')

    X_train, y_train, X_val, y_val = load_data({'train': args.train, 'val': args.val}, test_mode=False)

    # Take a sample of 200 images if using 'local_test_mode', where 0 is False and 1 is True
    if args.local_test_mode:
        X_train = X_train[:200]
        y_train = y_train[:200]
        X_val = X_val[:200]
        y_val = y_val[:200]

    logger.info(f'Shape of the training data: {X_train.shape}')
    logger.info(f'Shape of the validation data: {X_val.shape}')

    # ------------------------------ Model training ------------------------------ #

    logger.info('Training model...')

    trained_model, early_stopper = train_baseline_cnn(
        train_data={'X_train': X_train, 'y_train': y_train},
        val_data={'X_val': X_val, 'y_val': y_val},
        conv2d_params={
            'conv2d_filters_list': [args.conv2d_filters_1, args.conv2d_filters_2, args.conv2d_filters_3, args.conv2d_filters_4, args.conv2d_filters_5],
            'conv2d_weight_decay': args.conv2d_weight_decay,
            'conv2d_batch_norm_momentum': args.conv2d_batch_norm_momentum
        },
        dense_params={
            'dense_units_list': [args.dense_units_1, args.dense_units_2],
            'dense_weight_decay': args.dense_weight_decay,
            'dense_batch_norm_momentum': args.dense_batch_norm_momentum,
            'dropout_rate': args.dropout_rate
        },
        aug_params={
            'random_flip_mode': args.random_flip_mode,
            'random_rotation_factor': args.random_rotation_factor,
            'random_contrast_factor': args.random_contrast_factor,
            'random_zoom_height_factor': args.random_zoom_height_factor,
            'random_zoom_width_factor': args.random_zoom_width_factor
        },
        opt_params={
            'sgd_initial_lr': args.sgd_initial_lr,
            'sgd_lr_decay_steps': args.sgd_lr_decay_steps,
            'sgd_lr_decay_rate': args.sgd_lr_decay_rate,
            'sgd_momentum': args.sgd_momentum,
            'sgd_clipnorm': args.sgd_clipnorm
        },
        fit_params={
            'batch_size': args.batch_size,
            'epochs': args.epochs,
            'use_sample_weights': args.use_sample_weights
        },
        num_classes=config.num_classes,
        input_shape=tuple(config.image_size + [config.num_channels])
    )

    logger.info(f'Best validation accuracy: {early_stopper.best}')

    # ------------------------------ Model saving ------------------------------ #

    trained_model.save(os.path.join(args.model_dir, '00000000'))