import os
from typing import Tuple, Union, List, Dict, Any
import pickle
import boto3
import logging
from functools import partial

import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Nopep8
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight

from hydra import compose, initialize, core
from omegaconf import OmegaConf
from custom_utils import get_logger, parser, add_additional_args, load_data, AugmentationModel

# ----------------------- Pretrained model instantiator ---------------------- #

def instantiate_pretrained_model(
    name: str, 
    pooling: str,
    input_shape: Tuple[int, int]) -> tf.keras.Model:
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
    input_shape : Tuple[int, int]
        The shape of the input data.

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
    pooling = pooling.lower() if pooling != 'none' else None
    if pooling not in [None, 'avg', 'max']:
        raise ValueError("The argument 'pooling' must be one of None, 'avg', or 'max'")

    if name == 'vgg19':
        conv_base = tf.keras.applications.VGG19(
            include_top=False,
            weights='imagenet',
            input_shape=input_shape,
            pooling=pooling
        )
    elif name == 'resnet50v2':
        conv_base = tf.keras.applications.ResNet50V2(
            include_top=False,
            weights='imagenet',
            input_shape=input_shape,
            pooling=pooling
        )
    elif name == 'xception':
        conv_base = tf.keras.applications.Xception(
            include_top=False,
            weights='imagenet',
            input_shape=input_shape,
            pooling=pooling
        )
    else:
        raise ValueError(f'Not implemented pretrained model name: {name}')

    return conv_base

# ------------------------ Fine-tuning training entry ------------------------ #

def fine_tune(
    train_data: Dict[str, np.ndarray],
    val_data: Dict[str, np.ndarray],
    pretrained_params: Dict[str, Any],
    dense_params: Dict[str, Any],
    aug_params: Dict[str, Any],
    opt_params: Dict[str, Any],
    fit_params: Dict[str, Any],
    num_classes: int,
    logger: logging.Logger,
    input_shape: Tuple[int, int]
    ) -> tf.keras.Model:
    """
    Fine-tune a pretrained convolutional neural network model for multiclass
    classification. 

    Parameters
    ----------
    train_data : Dict[str, np.ndarray]
        A dictionary containing the training data--- X_train and y_train.
    val_data : Dict[str, np.ndarray]
        A dictionary containing the validation data--- X_val and y_val.
    pretrained_params : Dict[str, Any]
        A dictionary containing the parameters for the pretrained model.
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
    logger : logging.Logger
        The logger object.
    input_shape : Tuple[int, int]
        The shape of the input data.

    Returns
    -------
    tf.keras.Model
        The fine-tuned model.
    """
    # -------------------- Download and instantiate pretrained ------------------- #

    logger.info('Instantiating pretrained model base...')

    conv_base = instantiate_pretrained_model(
        name=pretrained_params['name'],
        input_shape=input_shape,
        pooling=pretrained_params['pooling']
    )

    # ------------------------ Train classifier at the top ----------------------- #

    logger.info('Training classifier at the top...')

    # Freeze all layers for the base
    conv_base.trainable = False

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

    # Preprocessing mapping for each pretrained model
    preprocess_input_mapping = {
        'vgg19': tf.keras.applications.vgg19.preprocess_input,
        'resnet50v2': tf.keras.applications.resnet_v2.preprocess_input,
        'xception': tf.keras.applications.xception.preprocess_input
    }
    preprocess_input_func = preprocess_input_mapping.get(pretrained_params['name'], lambda x: x)

    inputs = tf.keras.Input(shape=input_shape)
    x = data_augmentation(inputs)
    x = preprocess_input_func(x)
    # Use training=False in case pre-trained models contains batch normalization (inference-mode)
    x = conv_base(x, training=False)
    # Flatten the output of the pretrained model
    x = tf.keras.layers.Flatten()(x)
    for i in range(3):
        x = DefaultDense(dense_params['dense_units_list'][i])(x)
        x = tf.keras.layers.BatchNormalization(momentum=dense_params['dense_batch_norm_momentum'])(x)
        x = tf.keras.layers.Dropout(dense_params['dropout_rate'])(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=opt_params['adam_initial_lr'],
                decay_steps=opt_params['adam_lr_decay_steps'],
                decay_rate=opt_params['adam_lr_decay_rate']
            ),
            beta_1=opt_params['adam_beta_1'],
            beta_2=opt_params['adam_beta_2'],
            clipnorm=opt_params['adam_clipnorm']
        ),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')]
    )

    # ------------------------------ Model training ------------------------------ #

    trainable_params_count = np.sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    logger.info(f'Number of trainable parameters for training classifier at the top: {trainable_params_count}')
    del trainable_params_count

    early_stopper_clf = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True)

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
        callbacks=[early_stopper_clf],
        validation_data=(val_data['X_val'], val_data['y_val']),
        # If both sample_weight and class_weight are provided, the weights are multiplied as per https://github.com/keras-team/keras/blob/master/keras/engine/training_utils_v1.py
        class_weight=None,
        sample_weight=sample_weights if fit_params['use_sample_weights'] else None
    )

    logger.info(f'Best validation accuracy after training classifier at the top: {early_stopper_clf.best}')

    # -------------------------------- Fine-tuning ------------------------------- #

    logger.info('Releasing layers for fine-tuning...')

    # For vgg19, fine-tune the last 4 convolutional layers of block5 (with 4 convolutional layers, max pooling and global pooling)
    if pretrained_params['name'] == 'vgg19':
        conv_base.trainable = True
        # Everything before is frozen
        for layer in conv_base.layers:
            if layer.name == 'block5_conv1':
                break
            layer.trainable = False
    # For resnet50v2, fine-tune the last residual module 'conv5'
    elif pretrained_params['name'] == 'resnet50v2':
        conv_base.trainable = True
        # Everything before is frozen
        for layer in conv_base.layers:
            if layer.name == 'conv5_block1_preact_bn':
                break
            layer.trainable = False
    # For xception, fine-tune the last two blocks 13 and 14
    elif pretrained_params['name'] == 'xception':
        conv_base.trainable = True
        # Everything before block 13 is frozen
        for layer in conv_base.layers:
            if layer.name == 'block13_sepconv1_act':
                break
            layer.trainable = False

    # Recompile model with reduced initial learning rate by a factor of 10
    reduced_initial_lr = opt_params['adam_initial_lr'] / 10
    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=reduced_initial_lr,
                decay_steps=opt_params['adam_lr_decay_steps'],
                decay_rate=opt_params['adam_lr_decay_rate']
            ),
            beta_1=opt_params['adam_beta_1'],
            beta_2=opt_params['adam_beta_2'],
            clipnorm=opt_params['adam_clipnorm']
        ),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')]
    )

    trainable_params_count = np.sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    logger.info(f'Number of trainable parameters for fine-tuning: {trainable_params_count}')
    del trainable_params_count

    early_stopper_fine_tune = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True)

    model.fit(
        x=train_data['X_train'],
        y=train_data['y_train'],
        batch_size=fit_params['batch_size'],
        epochs=fit_params['epochs'],
        verbose=2,
        callbacks=[early_stopper_fine_tune],
        validation_data=(val_data['X_val'], val_data['y_val']),
        sample_weight=sample_weights if fit_params['use_sample_weights'] else None
    )

    logger.info(f'Best validation accuracy after fine-tuning: {early_stopper_fine_tune.best}')

    return model

if __name__ == '__main__':

    # -------------------------- Compose configurations -------------------------- #
    
    core.global_hydra.GlobalHydra.instance().clear()
    initialize(version_base='1.2', config_path='config', job_name='fine_tune')
    config = OmegaConf.to_container(compose(config_name='main'), resolve=True)

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
        'dense_units_1': int,
        'dense_units_2': int,
        'dense_units_3': int,
        'dense_weight_decay': float,
        'dense_batch_norm_momentum': float,
        'dropout_rate': float,
        # Optimization and fit parameters
        'adam_initial_lr': float,
        'adam_lr_decay_steps': int,
        'adam_lr_decay_rate': float,
        'adam_beta_1': float,
        'adam_beta_2': float,
        'adam_clipnorm': float,
        'batch_size': int,
        'epochs': int,
        'use_sample_weights': int,
        # Pretraining parameters
        'pretrained_name': str,
        'pretrained_pooling': str
    }

    args = add_additional_args(parser_func=parser, additional_args=additional_args)()

    # --------------------------------- Load data -------------------------------- #

    logger.info('Loading data...')

    X_train, y_train, X_val, y_val = load_data({'train': args.train, 'val': args.val}, test_mode=False)

    # Take a sample of 20 images if using 'local_test_mode', where 0 is False and 1 is True
    if args.local_test_mode:
        X_train = X_train[:20]
        y_train = y_train[:20]
        X_val = X_val[:20]
        y_val = y_val[:20]

    logger.info(f'Shape of the training data: {X_train.shape}')
    logger.info(f'Shape of the validation data: {X_val.shape}')

    # ------------------------------ Model training ------------------------------ #

    logger.info('Fine-tuning model...')    

    fine_tuned_model = fine_tune(
        train_data={'X_train': X_train, 'y_train': y_train},
        val_data={'X_val': X_val, 'y_val': y_val},
        pretrained_params={
            'name': args.pretrained_name,
            'pooling': args.pretrained_pooling
        },
        dense_params={
            'dense_units_list': [args.dense_units_1, args.dense_units_2, args.dense_units_3],
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
            'adam_initial_lr': args.adam_initial_lr,
            'adam_lr_decay_steps': args.adam_lr_decay_steps,
            'adam_lr_decay_rate': args.adam_lr_decay_rate,
            'adam_beta_1': args.adam_beta_1,
            'adam_beta_2': args.adam_beta_2,
            'adam_clipnorm': args.adam_clipnorm
        },
        fit_params={
            'batch_size': args.batch_size,
            'epochs': args.epochs,
            'use_sample_weights': args.use_sample_weights
        },
        num_classes=config['num_classes'],
        logger=logger,
        input_shape=tuple(config['image_size'] + [config['num_channels']])
    )

    # ------------------------------- Model saving ------------------------------- #

    logger.info('Saving model...')

    fine_tuned_model.save(os.path.join(args.model_dir, '00000000'))