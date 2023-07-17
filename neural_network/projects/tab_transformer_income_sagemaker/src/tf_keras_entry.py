import os
import argparse
from typing import Tuple, Union, List, Dict, Any, Optional, Callable
import logging
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Nopep8
import tensorflow as tf

from hydra import compose, initialize, core
from omegaconf import OmegaConf
import numpy as np
import optuna 

# -------------------------------- Model input ------------------------------- #

def create_model_inputs(config: Dict[str, Any]) -> tf.keras.Input:
    """
    This function creates model inputs as a dictionary, where the keys are the
    feature names and the values are tf.keras.layers.Input tensors with corresponding
    shapes and data types.

    Parameters
    ----------
    config : Dict[str, Any]
        Configuration dictionary.
    
    Returns
    -------
    tf.keras.Input
        Model inputs.
    """
    inputs = {}
    feature_names = config['tf_keras']['num_feat'] + list(config['tf_keras']['cat_feat_vocab'].keys())
    for feature_name in feature_names:
        if feature_name in config['tf_keras']['num_feat']:
            inputs[feature_name] = tf.keras.layers.Input(
                name=feature_name, shape=(), dtype=tf.float32
            )
        else:
            inputs[feature_name] = tf.keras.layers.Input(
                name=feature_name, shape=(), dtype=tf.string
            )
    return inputs

# ------------------------------- Encode input ------------------------------- #

def encode_inputs(inputs: tf.keras.Input, embedding_dims: int, config: Dict[str, Any]) -> Tuple[List[tf.Tensor], List[tf.Tensor]]:
    """
    This function encodes the model inputs for the tab-transformer model.

    Parameters
    ----------
    inputs : tf.keras.Input
        Model inputs.
    embedding_dims : int
        Embedding dimensions shared by all categorical features.
    config : Dict[str, Any]
        Configuration dictionary.

    Returns
    -------
    Tuple[List[tf.Tensor], List[tf.Tensor]]
        List of continuous and categorical embeddings.
    """
    encoded_categorical_feature_list = []
    numerical_feature_list = []

    for feature_name in inputs:

        if feature_name in config['tf_keras']['cat_feat_vocab'].keys():

            # Get the vocabulary of the categorical feature
            vocabulary = config['tf_keras']['cat_feat_vocab'][feature_name]

            # Create a lookup to convert string values to an integer indices
            lookup = tf.keras.layers.StringLookup(
                vocabulary=vocabulary,
                mask_token=None,
                num_oov_indices=config['tf_keras']['num_oov_indices'],
                output_mode='int'
            )

            # Convert the string input values into integer indices
            encoded_feature = lookup(inputs[feature_name])

            # Create an embedding layer with the specified dimensions
            embedding = tf.keras.layers.Embedding(
                input_dim=len(vocabulary), output_dim=embedding_dims
            )

            # Convert the index values to embedding representations.
            encoded_categorical_feature = embedding(encoded_feature)
            encoded_categorical_feature_list.append(encoded_categorical_feature)

        else:

            # Use the numerical features as-is
            numerical_feature = tf.expand_dims(inputs[feature_name], -1)
            numerical_feature_list.append(numerical_feature)

    return encoded_categorical_feature_list, numerical_feature_list

# ------------------------------------ MLP ----------------------------------- #

def mlp_classifier(hidden_units: List[int],
                   dropout_rate: float, 
                   activation: tf.keras.activations, 
                   norm_layer: tf.keras.layers,
                   name: str = 'mlp') -> tf.keras.Sequential:
    """
    Function to create a multi-layer perceptron (MLP) classifier.

    Parameters
    ----------
    hidden_units : List[int]
        List of hidden units for each layer.
    dropout_rate : float
        Dropout rate.
    activation : tf.keras.activations
        Activation function.
    norm_layer : tf.keras.layers
        Normalization layer.
    name : str, optional
        Name of the model, by default 'mlp'.

    Returns
    -------
    tf.keras.Sequential
        MLP classifier.
    """
    mlp_layers = []
    for units in hidden_units:
        # Batch norm applied after dropout and after activation except for the first dense layer
        mlp_layers.append(norm_layer)
        mlp_layers.append(tf.keras.layers.Dense(units, activation=activation))
        mlp_layers.append(tf.keras.layers.Dropout(dropout_rate))

    return tf.keras.Sequential(mlp_layers, name=name)

# ------------------------------ Tabtransformer ------------------------------ #

def tab_transformer_classifier(num_transformer_blocks: int,
                               num_heads: int,
                               embedding_dims: int,
                               dropout_rate: float,
                               config: Dict[str, Any],
                               mlp_hidden_units_factors: List[int],
                               use_column_embedding: bool = False,
                               epsilon: float = 1e-16) -> tf.keras.Model:
    """
    Function to create a tab-transformer classifier.

    Parameters
    ----------
    num_transformer_blocks : int
        Number of transformer blocks.
    num_heads : int
        Number of attention heads.
    embedding_dims : int
        Embedding dimensions shared by all categorical features.
    dropout_rate : float
        Dropout rate.
    config : Dict[str, Any]
        Configuration dictionary.
    mlp_hidden_units_factors : List[int]
        List of factors by which we multiply the embedding dimensions, which are inputs to the MLP block, to get the number of units for each layer of the MLP.
    use_column_embedding : bool, optional
        Whether to use column embeddings, by default False.
    epsilon : float, optional
        Small number to add to the variance to avoid dividing by zero, by default 1e-16.
    
    Returns
    -------
    tf.keras.Model
        Tab-transformer classifier.
    """
    # Model inputs
    inputs = create_model_inputs(config=config)
    # Encode features
    encoded_categorical_feature_list, numerical_feature_list = encode_inputs(
        inputs=inputs, embedding_dims=embedding_dims, config=config
    )
    # Stack categorical feature embeddings with shape (batch_size, num_cat_feat, embedding_dims)
    encoded_categorical_features = tf.stack(encoded_categorical_feature_list, axis=1)
    # Concatenate numerical features with shape (batch_size, num_cont_feat)
    numerical_features = tf.keras.layers.concatenate(numerical_feature_list)

    # Add column embedding to categorical feature embeddings
    if use_column_embedding:

        # The second dimension is the number of categorical features
        num_cat_feat = encoded_categorical_features.shape[1]

        column_embedding = tf.keras.layers.Embedding(
            input_dim=num_cat_feat, output_dim=embedding_dims
        )

        # This is a tensor with shape shape=(num_cat_feat, )
        col_indices = tf.range(start=0, limit=num_cat_feat, delta=1)

        # Note column_embedding(col_indices) maps from (num_cat_feat, ) to (num_cat_feat, embedding_dims)
        encoded_categorical_features = encoded_categorical_features + column_embedding(col_indices)

    # Create multiple layers of the Transformer block
    for block_idx in range(num_transformer_blocks):

        # Create a multi-head attention layer
        attention_output = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embedding_dims,
            dropout=dropout_rate,
            name=f'multihead_attention_{block_idx}',
        )(encoded_categorical_features, encoded_categorical_features)

        # Skip connection 1
        x = tf.keras.layers.Add(name=f'skip_connection1_{block_idx}')(
            [attention_output, encoded_categorical_features]
        )

        # Layer normalization 1
        x = tf.keras.layers.LayerNormalization(name=f'layer_norm1_{block_idx}', epsilon=epsilon)(x)

        # Feedforward
        feedforward_output = mlp_classifier(
            hidden_units=[embedding_dims],
            dropout_rate=dropout_rate,
            activation=tf.keras.activations.gelu,
            norm_layer=tf.keras.layers.LayerNormalization(epsilon=epsilon),
            name=f'feedforward_{block_idx}',
        )(x)

        # Skip connection 2
        x = tf.keras.layers.Add(name=f'skip_connection2_{block_idx}')([feedforward_output, x])

        # Layer normalization 2
        encoded_categorical_features = tf.keras.layers.LayerNormalization(
            name=f'layer_norm2_{block_idx}', epsilon=epsilon
        )(x)

    # Flatten the 'contextualized' embeddings (batch_size, num_cat_feat, embedding_dims) of the categorical features
    categorical_features = tf.keras.layers.Flatten()(encoded_categorical_features)
    # Apply layer normalization to the numerical features
    numerical_features = tf.keras.layers.LayerNormalization(epsilon=epsilon)(numerical_features)
    # Concatenate the input for the final MLP block (batch_size, (num_cat_feat * embedding_dims) + num_cont_feat)
    features = tf.keras.layers.concatenate([categorical_features, numerical_features])

    # Compute MLP hidden units as multiples of feature matrix column dimension, which is (num_cat_feat * embedding_dims) + num_cont_feat)
    mlp_hidden_units = [factor * features.shape[-1] for factor in mlp_hidden_units_factors]
    features = mlp_classifier(
        hidden_units=mlp_hidden_units,
        dropout_rate=dropout_rate,
        # The original paper uses selu as activation
        activation=tf.keras.activations.selu,
        norm_layer=tf.keras.layers.BatchNormalization(),
        name='mlp',
    )(features)

    # Use linear and apply sigmoid later
    outputs = tf.keras.layers.Dense(units=1, activation='linear', name='output_layer')(features)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    return model

# ----------------------------- Trainer function ----------------------------- #

def trainer(train_data: tf.data.Dataset,
            val_data: tf.data.Dataset,
            tensorboard_log_dir: str,
            hyperparameters: Dict[str, Any],
            config: Dict[str, Any],
            strategy: tf.distribute.Strategy,
            epsilon: float = 1e-16,
            early_stopping_patience: int = 3,
            verbose: int = 2) -> Tuple[tf.keras.Model, Dict[str, Any]]:
    """
    Trainer function for the tab-transformer model.

    Parameters
    ----------
    model : tf.keras.Model
        Model to train.
    train_data : tf.data.Dataset
        Training dataset.
    val_data : tf.data.Dataset
        Validation dataset.
    tensorboard_log_dir : str
        Path to the tensorboard log directory.
    hyperparameters : Dict[str, Any]
        Hyperparameters for the model.
    config : Dict[str, Any]
        Config dictionary.
    strategy : tf.distribute.Strategy
        Distribution strategy.
    epsilon : float, optional
        Epsilon value for layer normalization, by default 1e-16.
    early_stopping_patience : int, optional
        Early stopping patience, by default 3.
    verbose : int, optional
        Verbosity level, by default 2.

    Returns
    -------
    Tuple[tf.keras.Model, Dict[str, Any]]
        Trained model with best weights and dictionary of loss and metrics.
    """
    # Model creation and compilation should be inside the strategy scope
    if strategy is not None:
        with strategy.scope():

            # Optimizer
            if hyperparameters['optimizer'] == 'adam':
                optimizer = tf.keras.optimizers.Adam(
                    learning_rate=hyperparameters['adam_learning_rate'],
                    beta_1=hyperparameters['adam_beta_1'],
                    beta_2=hyperparameters['adam_beta_2'],
                    epsilon=hyperparameters['adam_epsilon'],
                    clipnorm=hyperparameters['adam_clipnorm'],
                    name='adam'
                )
            elif hyperparameters['optimizer'] == 'sgd':
                optimizer = tf.keras.optimizers.SGD(
                    learning_rate=hyperparameters['sgd_learning_rate'],
                    momentum=hyperparameters['sgd_momentum'],
                    clipnorm=hyperparameters['sgd_clipnorm'],
                    name='sgd'
                )

            # Loss function
            loss = tf.keras.losses.BinaryFocalCrossentropy(
                apply_class_balancing=hyperparameters['loss_apply_class_balancing'],
                alpha=hyperparameters['loss_alpha'],
                gamma=hyperparameters['loss_gamma'],
                from_logits=True,
                name='focal_loss'
            )
            
            # Create model
            tab_transformer_model = tab_transformer_classifier(
                num_transformer_blocks=hyperparameters['num_transformer_blocks'],
                num_heads=hyperparameters['num_heads'],
                embedding_dims=hyperparameters['embedding_dims'],
                mlp_hidden_units_factors=[hyperparameters[f'mlp_hidden_units_factor_{i}'] for i in range(hyperparameters['mlp_num_hidden_layers'])],
                dropout_rate=hyperparameters['dropout_rate'],
                use_column_embedding=hyperparameters['use_column_embedding'],
                config=config,
                epsilon=epsilon
            )
            
            # Compile model
            tab_transformer_model.compile(
                optimizer=optimizer,
                loss=loss,
                # Use weighted metrics since the datasets return (X, y, sample_weight)
                weighted_metrics=[
                    tf.keras.metrics.AUC(curve='PR', name='aucpr', from_logits=True),
                    # If used with a loss function with from_logits=True, threshold should be 0
                    tf.keras.metrics.Recall(thresholds=0, name='recall'),
                    tf.keras.metrics.Precision(thresholds=0, name='precision'),
                    # Argmax of logits and probabilities and so we can use the accuracy metric on logits by default
                    tf.keras.metrics.BinaryAccuracy(name='accuracy')
                ]
            )
    else:

        # Optimizer
        if hyperparameters['optimizer'] == 'adam':
            optimizer = tf.keras.optimizers.Adam(
                learning_rate=hyperparameters['adam_learning_rate'],
                beta_1=hyperparameters['adam_beta_1'],
                beta_2=hyperparameters['adam_beta_2'],
                epsilon=hyperparameters['adam_epsilon'],
                clipnorm=hyperparameters['adam_clipnorm'],
                name='adam'
            )
        elif hyperparameters['optimizer'] == 'sgd':
            optimizer = tf.keras.optimizers.SGD(
                learning_rate=hyperparameters['sgd_learning_rate'],
                momentum=hyperparameters['sgd_momentum'],
                clipnorm=hyperparameters['sgd_clipnorm'],
                name='sgd'
            )

        # Loss function
        loss = tf.keras.losses.BinaryFocalCrossentropy(
            apply_class_balancing=hyperparameters['loss_apply_class_balancing'],
            alpha=hyperparameters['loss_alpha'],
            gamma=hyperparameters['loss_gamma'],
            from_logits=True,
            name='focal_loss'
        )

        tab_transformer_model = tab_transformer_classifier(
            num_transformer_blocks=hyperparameters['num_transformer_blocks'],
            num_heads=hyperparameters['num_heads'],
            embedding_dims=hyperparameters['embedding_dims'],
            mlp_hidden_units_factors=[hyperparameters[f'mlp_hidden_units_factor_{i}'] for i in range(hyperparameters['mlp_num_hidden_layers'])],
            dropout_rate=hyperparameters['dropout_rate'],
            use_column_embedding=hyperparameters['use_column_embedding'],
            config=config,
            epsilon=epsilon
        )

        tab_transformer_model.compile(
            optimizer=optimizer,
            loss=loss,
            # Use weighted metrics since the datasets return (X, y, sample_weight)
            weighted_metrics=[
                tf.keras.metrics.AUC(curve='PR', name='aucpr', from_logits=True),
                # If used with a loss function with from_logits=True, threshold should be 0
                tf.keras.metrics.Recall(thresholds=0, name='recall'),
                tf.keras.metrics.Precision(thresholds=0, name='precision'),
                # Argmax of logits and probabilities and so we can use the accuracy metric on logits by default
                tf.keras.metrics.BinaryAccuracy(name='accuracy')
            ]
        )

    # Callbacks
    early_stopper = tf.keras.callbacks.EarlyStopping(
        monitor='val_aucpr',
        mode='max',
        patience=early_stopping_patience,
        restore_best_weights=True
    )
    if tensorboard_log_dir is not None:
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=tensorboard_log_dir)

    # Training
    tab_transformer_model.fit(
        x=train_data,
        epochs=hyperparameters['fit_epochs'],
        verbose=verbose,
        callbacks=[early_stopper, tensorboard_callback] if tensorboard_log_dir is not None else [early_stopper],
        validation_data=val_data,
        # Class weight is already applied in the loss function
        class_weight=None,
        # Sample weights are already applied in the dataset
        sample_weight=None
    )

    # Evaluation
    val_loss_metrics = tab_transformer_model.evaluate(
        x=val_data,
        verbose=verbose,
        return_dict=True
    )

    return tab_transformer_model, val_loss_metrics

# ----------------------------- Optuna objective ----------------------------- #

def tf_objective(trial: optuna.Trial,
                 train_data: tf.data.Dataset,
                 val_data: tf.data.Dataset,
                 strategy: tf.distribute.Strategy,
                 config: Dict[str, Any],
                 logger: logging.Logger,
                 test_mode: int,
                 job_name: str,
                 epsilon: int = 1e-16,
                 early_stopping_patience: int = 3,
                 verbose: int = 2) -> float:
    """
    Objective function for Optuna hyperparameter optimization.

    Parameters
    ----------
    trial : optuna.Trial
        Optuna trial.
    train_data : tf.data.Dataset
        Training data.
    val_data : tf.data.Dataset
        Validation data.
    strategy : tf.distribute.Strategy
        Distribution strategy.
    config : Dict[str, Any]
        Configuration dictionary.
    logger : logging.Logger
        Logger object.
    test_mode: int
        Whether to run objective function in test mode.
    job_name : str
        The sagemaker training job name.
    epsilon : int, optional
        Small number to add to the variance to avoid dividing by zero, by default 1e-16.
    early_stopping_patience : int, optional
        Number of epochs to wait before early stopping, by default 3.
    verbose : int, optional
        Verbosity level, by default 2.

    Returns
    -------
    float
        Validation metric.
    """
    hyperparameters = {
        'num_transformer_blocks': trial.suggest_int('num_transformer_blocks', 3, 6),
        'num_heads': trial.suggest_int('num_heads', 2, 8), 
        'embedding_dims': trial.suggest_categorical('embedding_dims', [2**power for power in range(4, 8)]), 
        'mlp_num_hidden_layers': trial.suggest_int('mlp_num_hidden_layers', 2, 4),
        'dropout_rate': trial.suggest_float('dropout_rate', 0.0, 0.5),
        'use_column_embedding': trial.suggest_categorical('use_column_embedding', [True, False]),
        'optimizer': trial.suggest_categorical('optimizer', ['adam', 'sgd']),
        'loss_apply_class_balancing': trial.suggest_categorical('loss_apply_class_balancing', [True, False]),
        'loss_alpha': trial.suggest_float('loss_alpha', 0.1, 0.9),
        'loss_gamma': trial.suggest_float('loss_gamma', 0.5, 2.0),
        'fit_epochs': trial.suggest_int('fit_epochs', 10, 25)
    }
    for i in range(hyperparameters['mlp_num_hidden_layers']):
        # The most number of units will occur when the embedding_dims is 128, so 128 * 4 = 512
        hyperparameters[f'mlp_hidden_units_factor_{i}'] = trial.suggest_int(f'mlp_hidden_units_factor_{i}', 2, 4)
    if hyperparameters['optimizer'] == 'adam':
        hyperparameters['adam_learning_rate'] = trial.suggest_float('adam_learning_rate', 1e-4, 1e-1, log=True)
        hyperparameters['adam_beta_1'] = trial.suggest_float('adam_beta_1', 0.7, 0.9)
        hyperparameters['adam_beta_2'] = trial.suggest_float('adam_beta_2', 0.9, 0.999)
        hyperparameters['adam_epsilon'] = trial.suggest_float('adam_epsilon', 1e-9, 1e-1)
        hyperparameters['adam_clipnorm'] = trial.suggest_float('adam_clipnorm', 1e-2, 1.0)
    elif hyperparameters['optimizer'] == 'sgd':
        hyperparameters['sgd_learning_rate'] = trial.suggest_float('sgd_learning_rate', 1e-4, 1e-1, log=True)
        hyperparameters['sgd_momentum'] = trial.suggest_float('sgd_momentum', 0.7, 0.9)
        hyperparameters['sgd_clipnorm'] = trial.suggest_float('sgd_clipnorm', 1e-2, 1.0)

    # ---------------------------- Train and validate ---------------------------- #

    logger.info(f'Training and validating model for trial {trial.number}...')

    # Tensorboard log directory
    if not test_mode:
        tensorboard_log_dir = f's3://{config["s3_bucket"]}/{config["s3_key"]}/tensorboard_logs/{job_name}_{trial.number}/'

    trained_model, val_loss_metrics = trainer(
        train_data=train_data,
        val_data=val_data,
        tensorboard_log_dir=tensorboard_log_dir if not test_mode else None,
        hyperparameters=hyperparameters,
        config=config,
        strategy=strategy,
        epsilon=epsilon,
        early_stopping_patience=early_stopping_patience,
        verbose=verbose
    )

    # -------- Retrain on the entire dataset with the best hyperparameters ------- #

    logger.info(f'Retraining model for trial {trial.number} on the entire training dataset...')

    # Concateate the train and validation datasets
    full_train_data = train_data.concatenate(val_data).shuffle(buffer_size=20000)

    full_trained_model, _ = trainer(
        train_data=full_train_data,
        val_data=full_train_data, # Use the full dataset for early stopping
        tensorboard_log_dir=None,
        hyperparameters=hyperparameters,
        config=config,
        strategy=strategy,
        epsilon=epsilon,
        early_stopping_patience=early_stopping_patience,
        verbose=verbose
    )

    # Save model
    logger.info(f'Saving model to tmp directory for trial {trial.number}...')
    local_model_dir = os.path.join('/tmp', f'model_trial_{trial.number}.keras')
    full_trained_model.save(local_model_dir)

    # Save training job name as an attribute
    trial.set_user_attr('job_name', job_name)
        
    return val_loss_metrics['aucpr']

if __name__ == '__main__':

    # These imports are only needed when running this file on SageMaker
    from custom_utils import (get_logger, parser, add_additional_args, get_db_url, 
                              create_study, study_report, dataset_from_csv, stratified_sample)
    import pandas as pd

    # ---------------------------------- Set up ---------------------------------- #

    additional_args = {
        'study_name': str,
        'multi_gpu': int,
        'epsilon': float,
        'early_stopping_patience': int,
        'verbose': int
    }

    args = add_additional_args(parser, additional_args)()

    logger = get_logger('tab_transformer_hpo')

    job_name = args.training_env['job_name']
    
    # Hydra
    core.global_hydra.GlobalHydra.instance().clear()
    initialize(version_base='1.2', config_path='config', job_name='tab_transformer_hpo')
    config = OmegaConf.to_container(compose(config_name='main'), resolve=True)

    # --------------------------------- Load data -------------------------------- #

    logger.info('Loading data...')

    if args.test_mode:

        logger.info('Running in test mode...')

        # Read in csv file from s3
        train = pd.read_csv(f's3://{config["s3_bucket"]}/{config["s3_key"]}/input-data/train/train.csv', names=config['csv_header'])
        val = pd.read_csv(f's3://{config["s3_bucket"]}/{config["s3_key"]}/input-data/val/val.csv', names=config['csv_header'])

        # Stratified sampling
        train = stratified_sample(train, list(config['tf_keras']['cat_feat_vocab'].keys()), 500)
        val = stratified_sample(val, list(config['tf_keras']['cat_feat_vocab'].keys()), 500)

        # Write to temporary directory
        train.to_csv('/tmp/train.csv', index=config['index'], header=config['tf_keras']['header'])
        val.to_csv('/tmp/val.csv', index=config['index'], header=config['tf_keras']['header'])

        train = dataset_from_csv(
            file_path='/tmp/train.csv',
            config=config,
            train=True
        )
        val = dataset_from_csv(
            file_path='/tmp/val.csv',
            config=config,
            train=False
        )
    else:

        logger.info('Running in SageMaker mode...')
        
        train = dataset_from_csv(
            file_path=os.path.join(args.train, 'train.csv'),
            config=config,
            train=True
        )

        val = dataset_from_csv(
            file_path=os.path.join(args.val, 'val.csv'),
            config=config,
            train=False
        )

    train_num_batches = train.reduce(np.int64(0), lambda x, _: x + 1).numpy()
    val_num_batches = val.reduce(np.int64(0), lambda x, _: x + 1).numpy()

    # Get the distributions of the target labels
    y_train = np.concatenate([element[1].numpy() for element in train], axis=0)
    y_val = np.concatenate([element[1].numpy() for element in val], axis=0)

    logger.info(f'Train dataset contains {train_num_batches} batches of at most {config["tf_keras"]["batch_size"]} samples each')
    logger.info(f'Validation dataset contains {val_num_batches} batches of at most {config["tf_keras"]["batch_size"]} samples each')
    logger.info(f'Train class distribution: {{0: {np.sum(y_train == 0) / len(y_train):.2f}, 1: {np.sum(y_train == 1) / len(y_train):.2f}}}')
    logger.info(f'Validation class distribution: {{0: {np.sum(y_val == 0) / len(y_val):.2f}, 1: {np.sum(y_val == 1) / len(y_val):.2f}}}')

    # --------------------------------- HPO setup -------------------------------- #

    logger.info('Setting up optuna database...')

    db_url = get_db_url(host=args.host, db_name=args.db_name, db_secret=args.db_secret, region_name=args.region_name)

    logger.info(f'Database URL: {db_url}')

    # ------------------------------- Optimization ------------------------------- #

    logger.info('Optimizing objective function...')

    # If not in test mode, multi-gpu training is enabled
    if args.multi_gpu:
        strategy = tf.distribute.MirroredStrategy(['GPU:0', 'GPU:1', 'GPU:2', 'GPU:3'])
        logger.info(f'Number of devices: {strategy.num_replicas_in_sync}')
    else:
        strategy = None

    def tf_objective_wrapper(trial: optuna.Trial) -> Callable:
        return tf_objective(
            trial=trial,
            train_data=train,
            val_data=val,
            strategy=strategy,
            config=config,
            logger=logger,
            test_mode=args.test_mode,
            job_name=job_name,
            epsilon=args.epsilon,
            early_stopping_patience=args.early_stopping_patience,
            verbose=args.verbose
        )

    study = create_study(study_name=args.study_name, storage=db_url, direction='maximize')
    study.optimize(tf_objective_wrapper, n_trials=args.n_trials)

    study_report(study=study, logger=logger)

    # ----------------------- Retrieve and save best model ----------------------- #

    logger.info('Retrieving best model and saving it to model directory...')

    best_model = tf.keras.models.load_model(os.path.join('/tmp', f'model_trial_{study.best_trial.number}.keras'))

    best_model.save(os.path.join(args.model_dir, '00000000'))