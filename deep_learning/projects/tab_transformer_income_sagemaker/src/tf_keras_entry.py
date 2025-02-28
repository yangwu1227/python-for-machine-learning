import logging
import os
import subprocess
from typing import Any, Dict, List, Tuple, cast

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Nopep8
import optuna
import tensorflow as tf
from hydra import compose, core, initialize
from model_utils import (
    add_additional_args,
    create_study,
    dataset_from_csv,
    get_db_url,
    get_logger,
    parser,
    study_report,
    test_sample,
)
from omegaconf import OmegaConf

# ------------------------------- Trainer class ------------------------------ #


class TabTransformerTrainer(object):
    """
    This class implements the composition, training, and hyperparameter optimization
    of the tab-transformer model.
    """

    def __init__(
        self,
        hyperparameters: Dict[str, Any],
        config: Dict[str, Any],
        job_name: str,
        trial_number: int,
        train_dataset: tf.data.Dataset,
        val_dataset: tf.data.Dataset,
        distributed: bool,
        strategy: tf.distribute.Strategy,
        model_dir: str,
        logger: logging.Logger,
    ) -> None:
        """
        Constructor for the TabTransformerTrainer class.

        Parameters
        ----------
        hyperparameters : Dict[str, Any]
            A dictionary containing the hyperparameters for model training.
        config : Dict[str, Any]
            A dictionary containing the configuration for model training.
        job_name : str
            The name of the job.
        trial_number : int
            The trial number for Optuna hyperparameter optimization.
        train_dataset : tf.data.Dataset
            A tf.data.Dataset object that contains the training data.
        val_dataset : tf.data.Dataset
            The validation data is recommend to be a repeated dataset.
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
        self.hyperparameters = hyperparameters
        self.config = config
        self.job_name = job_name
        self.trial_number = trial_number
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.distributed = distributed
        self.strategy = strategy
        self.model_dir = model_dir
        self.logger = logger

    def _create_optimizer(self) -> tf.keras.optimizers.Optimizer:
        """
        This function creates an optimizer based on the hyperparameters.

        Returns
        -------
        tf.keras.optimizers.Optimizer
            An optimizer.
        """
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.hyperparameters["adam_learning_rate"],
            beta_1=self.hyperparameters["adam_beta_1"],
            beta_2=self.hyperparameters["adam_beta_2"],
            epsilon=self.hyperparameters["adam_epsilon"],
            clipnorm=self.hyperparameters["adam_clipnorm"],
            name="adam",
        )

        return optimizer

    def _create_loss_fn(self) -> tf.keras.losses.Loss:
        """
        This function creates a loss function based on the hyperparameters.
        The loss function can either be categorical cross entropy or focal loss.

        Returns
        -------
        tf.keras.losses.Loss
            A loss function.
        """
        if self.hyperparameters["use_focal_loss"]:
            loss_fn = tf.keras.losses.BinaryFocalCrossentropy(
                apply_class_balancing=self.hyperparameters[
                    "loss_apply_class_balancing"
                ],
                alpha=self.hyperparameters["loss_alpha"],
                gamma=self.hyperparameters["loss_gamma"],
                from_logits=True,
                name="loss",
            )
        else:
            loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True, name="loss")

        return loss_fn

    def _create_metrics(self) -> List[tf.keras.metrics.Metric]:
        """
        This function creates a list of metrics for model evaluation.

        Returns
        -------
        List[tf.keras.metrics.Metric]
            A list of metrics--- accuracy, precision, recall, and area under PR curve.
        """
        metrics = [
            # Argmax of logits and probabilities are the same and so we can use the accuracy metric on logits by default
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            # If used with a loss function with from_logits=True, threshold should be 0
            tf.keras.metrics.Recall(thresholds=0, name="recall"),
            tf.keras.metrics.Precision(thresholds=0, name="precision"),
            tf.keras.metrics.AUC(curve="PR", from_logits=True, name="auc_pr"),
        ]

        return metrics

    def _create_inputs(self) -> tf.keras.Input:
        """
        This function creates model inputs as a dictionary, where the keys are the
        feature names and the values are tf.keras.layers.Input tensors with corresponding
        shapes and data types.

        Returns
        -------
        tf.keras.Input
            Model inputs.
        """
        inputs = {}
        feature_names = self.config["tf_keras"]["num_feat"] + list(
            self.config["tf_keras"]["cat_feat_vocab"].keys()
        )
        for feature_name in feature_names:
            if feature_name in self.config["tf_keras"]["num_feat"]:
                inputs[feature_name] = tf.keras.layers.Input(
                    name=feature_name, shape=(), dtype=tf.float32
                )
            else:
                inputs[feature_name] = tf.keras.layers.Input(
                    name=feature_name, shape=(), dtype=tf.string
                )

        return inputs

    def _encode_inputs(
        self, inputs: tf.keras.Input
    ) -> Tuple[List[tf.Tensor], List[tf.Tensor]]:
        """
        This function encodes the model inputs for the tab-transformer model.

        Parameters
        ----------
        inputs : tf.keras.Input
            Model inputs.

        Returns
        -------
        Tuple[List[tf.Tensor], List[tf.Tensor]]
            List of continuous and categorical embeddings.
        """
        encoded_categorical_feature_list = []
        numerical_feature_list = []

        for feature_name in inputs:
            if feature_name in self.config["tf_keras"]["cat_feat_vocab"].keys():
                # Get the vocabulary of the categorical feature
                vocabulary = self.config["tf_keras"]["cat_feat_vocab"][feature_name]

                # Create a lookup to convert string values to an integer indices
                lookup = tf.keras.layers.StringLookup(
                    vocabulary=vocabulary,
                    mask_token=None,
                    num_oov_indices=self.config["tf_keras"]["num_oov_indices"],
                    output_mode="int",
                )

                # Convert the string input values into integer indices
                encoded_feature = lookup(inputs[feature_name])

                # Create an embedding layer with the specified dimensions
                embedding = tf.keras.layers.Embedding(
                    input_dim=len(vocabulary),
                    output_dim=self.hyperparameters["transformer_embedding_dims"],
                )

                # Integer indices to embedding representations (embedding space's dimensionality is a hyperparameter)
                encoded_categorical_feature = embedding(encoded_feature)
                encoded_categorical_feature_list.append(encoded_categorical_feature)

            else:
                # Use the numerical features as-is (create a tensor with shape (None, 1) for each inputs[feature_name]])
                numerical_feature = tf.expand_dims(inputs[feature_name], -1)
                numerical_feature_list.append(numerical_feature)

        return encoded_categorical_feature_list, numerical_feature_list

    def _mlp(
        self,
        hidden_units: List[int],
        dropout_rate: float,
        activation: tf.keras.activations,
        norm_layer: tf.keras.layers,
        name: str,
    ) -> tf.keras.Sequential:
        """
        Function to create a multi-layer perceptron (MLP) classifier.

        Parameters
        ----------
        hidden_units : List[int]
            List of hidden units for each layer of the MLP.
        dropout_rate : float
            Dropout rate.
        activation : tf.keras.activations
            Activation function.
        norm_layer : tf.keras.layers
            Normalization layer.
        name : str, optional
            Name of the model.

        Returns
        -------
        tf.keras.Sequential
            MLP classifier.
        """
        mlp_layers = []
        for units in hidden_units:
            # Normalization applied after dropout and after activation except for the first dense layer
            mlp_layers.append(norm_layer)
            mlp_layers.append(tf.keras.layers.Dense(units=units, activation=activation))
            mlp_layers.append(tf.keras.layers.Dropout(rate=dropout_rate))

        return tf.keras.Sequential(mlp_layers, name=name)

    def _create_model(self, epsilon: float = 1e-16) -> tf.keras.Model:
        """
        Function to create a tab-transformer classifier.

        Parameters
        ----------
        epsilon : float, optional
            Small number to add to the variance to avoid dividing by zero, by default 1e-16.

        Returns
        -------
        tf.keras.Model
            The compiled tab-transformer model.
        """
        # ----------------------- Inputs to transformer blocks ----------------------- #

        inputs = self._create_inputs()

        encoded_categorical_feature_list, numerical_feature_list = self._encode_inputs(
            inputs=inputs
        )

        # Stack categorical feature embeddings with shape (batch_size, num_cat_feat, embedding_dims)
        encoded_categorical_features = tf.stack(
            encoded_categorical_feature_list, axis=1
        )
        # Concatenate numerical features with shape (batch_size, num_cont_feat)
        numerical_features = tf.keras.layers.concatenate(numerical_feature_list)

        # ----------------------------- Column embedding ----------------------------- #

        # The second dimension is the number of categorical features
        num_cat_feat = encoded_categorical_features.shape[1]
        # The goal of having column embedding is to enable the model to distinguish the classes in one categorical feature from those in others
        column_embedding = tf.keras.layers.Embedding(
            input_dim=num_cat_feat,
            output_dim=self.hyperparameters["transformer_embedding_dims"],
        )
        # This is a 1D-tensor with shape shape=(num_cat_feat, )
        col_indices = tf.range(start=0, limit=num_cat_feat, delta=1)
        # Note column_embedding(col_indices) maps from (num_cat_feat, ) to (num_cat_feat, embedding_dims)
        # This addition is broadcasted to (batch_size, num_cat_feat, embedding_dims), so each example gets the same column embedding added
        encoded_categorical_features = encoded_categorical_features + column_embedding(
            col_indices
        )

        # ----------------------------- Transformer block ----------------------------- #

        for block in range(self.hyperparameters["transformer_num_layers"]):
            # Create a multi-head attention layer output
            attention_output = tf.keras.layers.MultiHeadAttention(
                num_heads=self.hyperparameters["transformer_num_heads"],
                key_dim=self.hyperparameters["transformer_embedding_dims"],
                dropout=self.hyperparameters["transformer_dropout_rate"],
                kernel_initializer="he_normal",
                name=f"transformer_block_{block}_multi_head_attention",
            )(
                query=encoded_categorical_features,
                value=encoded_categorical_features,
                key=encoded_categorical_features,
            )

            # Skip connection 1
            skip_con_1 = tf.keras.layers.Add(
                name=f"transformer_block_{block}_skip_connection_1"
            )([attention_output, encoded_categorical_features])
            # Layer normalization 1
            norm_1 = tf.keras.layers.LayerNormalization(
                epsilon=epsilon, name=f"transformer_block_{block}_layer_normalization_1"
            )(skip_con_1)

            # Feedforward
            mlp_output = self._mlp(
                # The mlp block inside the transformer block should end with a layer with 'embedding_dims' hidden units for the skip connection to work
                hidden_units=[self.hyperparameters["transformer_embedding_dims"]],
                dropout_rate=self.hyperparameters["mlp_dropout_rate"],
                activation=tf.keras.activations.gelu,
                norm_layer=tf.keras.layers.LayerNormalization(epsilon=epsilon),
                name=f"transformer_block_{block}_mlp",
            )(norm_1)

            # Skip connection 2
            skip_con_2 = tf.keras.layers.Add(
                name=f"transformer_block_{block}_skip_connection_2"
            )([mlp_output, norm_1])

            # Layer normalization 2
            encoded_categorical_features = tf.keras.layers.LayerNormalization(
                epsilon=epsilon, name=f"transformer_block_{block}_layer_normalization_2"
            )(skip_con_2)

        # Flatten the 'contextualized' embeddings (batch_size, num_cat_feat, embedding_dims) of the categorical features
        categorical_features = tf.keras.layers.Flatten()(encoded_categorical_features)
        # Apply layer normalization to the numerical features
        numerical_features = tf.keras.layers.LayerNormalization(
            epsilon=epsilon, name="numerical_layer_normalization"
        )(numerical_features)
        # Concatenate the input for the final MLP block (batch_size, (num_cat_feat * embedding_dims) + num_cont_feat)
        features = tf.keras.layers.concatenate(
            inputs=[categorical_features, numerical_features], axis=-1
        )

        # ----------------------------- MLP classifier ----------------------------- #

        # Compute MLP hidden units as multiples of feature matrix column dimension, which is (num_cat_feat * embedding_dims) + num_cont_feat)
        mlp_hidden_units = [
            factor * features.shape[-1]
            for factor in self.hyperparameters["mlp_hidden_units_multiples"]
        ]

        features = self._mlp(
            hidden_units=mlp_hidden_units,
            dropout_rate=self.hyperparameters["mlp_dropout_rate"],
            # The original paper uses selu as activation
            activation=tf.keras.activations.selu,
            norm_layer=tf.keras.layers.BatchNormalization(),
            name="mlp_classifier",
        )(features)

        # Use linear and apply sigmoid later
        outputs = tf.keras.layers.Dense(
            units=1, activation="linear", name="output_layer"
        )(features)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)

        # ---------------------------------- Compile --------------------------------- #

        optimizer = self._create_optimizer()
        loss_fn = self._create_loss_fn()
        metrics = self._create_metrics()
        model.compile(
            optimizer=optimizer,
            loss=loss_fn,
            # Use weighted metrics since the datasets return (X, y, sample_weight)
            weighted_metrics=metrics,
        )

        return model

    def fit(self) -> Tuple[tf.keras.Model, float]:
        """
        Function for training the model.

        Returns
        -------
        Tuple[tf.keras.Model, float]
            Trained model and validation metric.
        """
        # ------------------------------- Create model ------------------------------- #

        if self.distributed:
            with self.strategy.scope():
                model = self._create_model()
        else:
            model = self._create_model()

        # ------------------------------- Callbacks ------------------------------- #

        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_auc_pr",
            mode="max",
            patience=self.config["tf_keras"]["patience"],
            restore_best_weights=True,
        )
        back_and_restore = tf.keras.callbacks.BackupAndRestore(
            backup_dir=os.path.join(os.getcwd(), "backup"),
            # Delete the backup directory after the training is completed, so the next call to 'fit' will create a new backup directory
            delete_checkpoint=True,
        )
        callbacks = [early_stopping, back_and_restore]

        retrain_on_full = self.train_dataset is self.val_dataset
        # Add tensorboard call back only when using train-val loop (i.e., check that train_dataset is not the same as val_dataset)
        if not retrain_on_full:
            tensorboard_local_dir = (
                f"/tmp/tensorboard/{self.job_name}_{self.trial_number}"
            )
            tensorboard = tf.keras.callbacks.TensorBoard(log_dir=tensorboard_local_dir)
            callbacks.append(tensorboard)

        # --------------------------------- Fit model -------------------------------- #

        model.fit(
            x=self.train_dataset,
            epochs=self.hyperparameters["fit_epochs"],
            validation_data=self.val_dataset,
            callbacks=callbacks,
            # When training on full dataset, set validation_steps to None to evaluate on all batches
            validation_steps=(
                self.hyperparameters["fit_validation_steps"]
                if not retrain_on_full
                else None
            ),
        )

        if not retrain_on_full:
            subprocess.run(
                [
                    "aws",
                    "s3",
                    "cp",
                    tensorboard_local_dir,
                    f"s3://{self.config['s3_bucket']}/{self.config['s3_key']}/tensorboard_logs/{self.job_name}_{self.trial_number}",
                    "--recursive",
                ]
            )

        return model, early_stopping.best


# ----------------------------- Optuna objective ----------------------------- #


def tf_objective(
    trial: optuna.Trial,
    config: Dict[str, Any],
    job_name: str,
    train_dataset: tf.data.Dataset,
    train_num_batches: int,
    val_dataset: tf.data.Dataset,
    val_num_batches: int,
    distributed: bool,
    strategy: tf.distribute.Strategy,
    model_dir: str,
    logger: logging.Logger,
) -> float:
    """
    This function is the objective function for Optuna hyperparameter optimization. For each trial,
    it creates a TabTransformerTrainer object and trains the model. Then retrains the model on the
    entire training dataset with the best hyperparameters. Finally, it emits the validation metric
    for CloudWatch hyperparameter tuning jobs and saves the model to the model directory.

    Parameters
    ----------
    trial : optuna.Trial
        An Optuna trial object.
    config : Dict[str, Any]
        A dictionary containing the configuration for model training.
    job_name : str
        The name of the job.
    train_dataset : tf.data.Dataset
        A tf.data.Dataset object that contains the training data.
    train_num_batches : int
        The number of batches in the training dataset.
    val_dataset : tf.data.Dataset
        The validation data is recommend to be a repeated dataset.
    val_num_batches : int
        The number of batches in the validation dataset.
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
    float
        Validation metric.
    """
    hyperparameters = {
        "transformer_num_layers": trial.suggest_int("transformer_num_layers", 3, 6),
        "transformer_num_heads": trial.suggest_int("transformer_num_heads", 2, 8),
        "transformer_embedding_dims": trial.suggest_categorical(
            "transformer_embedding_dims", [2**power for power in range(5, 8)]
        ),
        "transformer_dropout_rate": trial.suggest_float(
            "transformer_dropout_rate", 1e-2, 0.5
        ),
        "mlp_num_hidden_layers": trial.suggest_int("mlp_num_hidden_layers", 1, 3),
        "mlp_dropout_rate": trial.suggest_float("mlp_dropout_rate", 1e-2, 0.5),
        "use_focal_loss": trial.suggest_categorical("use_focal_loss", [True, False]),
        "adam_learning_rate": trial.suggest_float(
            "adam_learning_rate", 1e-4, 1e-1, log=True
        ),
        "adam_beta_1": trial.suggest_float("adam_beta_1", 0.6, 0.9),
        "adam_beta_2": trial.suggest_float("adam_beta_2", 0.8, 0.999),
        "adam_epsilon": trial.suggest_float("adam_epsilon", 1e-9, 1e-1),
        "adam_clipnorm": trial.suggest_float("adam_clipnorm", 1e-2, 1.0),
        "fit_epochs": trial.suggest_int("fit_epochs", 10, 20),
    }

    # Conditionally add multiples for each hidden layer of the MLP classifier
    hyperparameters["mlp_hidden_units_multiples"] = []
    for i in range(hyperparameters["mlp_num_hidden_layers"]):
        # The most number of units will occur when the embedding_dims is 128, so 128 * 4 = 512
        hyperparameters["mlp_hidden_units_multiples"].append(
            trial.suggest_int(f"mlp_hidden_units_multiple_{i}", 2, 4)
        )

    # Conditionally add focal loss hyperparameters
    if hyperparameters["use_focal_loss"]:
        hyperparameters["loss_apply_class_balancing"] = trial.suggest_categorical(
            "loss_apply_class_balancing", [True, False]
        )
        hyperparameters["loss_alpha"] = trial.suggest_float("loss_alpha", 0.1, 0.9)
        hyperparameters["loss_gamma"] = trial.suggest_float("loss_gamma", 0.5, 5.0)

    # Add number of validation steps
    hyperparameters["fit_validation_steps"] = val_num_batches

    # ---------------------------- Train and validate ---------------------------- #

    logger.info(f"Training and validating model for trial {trial.number}...")

    trainer = TabTransformerTrainer(
        hyperparameters=hyperparameters,
        config=config,
        job_name=job_name,
        trial_number=trial.number,
        train_dataset=train_dataset,
        # Repeat valdidation set
        val_dataset=val_dataset.repeat(),
        distributed=distributed,
        strategy=strategy,
        model_dir=model_dir,
        logger=logger,
    )

    _, val_auc_pr = trainer.fit()

    # -------- Retrain on the entire dataset with the best hyperparameters ------- #

    logger.info(
        f"Retraining model for trial {trial.number} on the entire training dataset..."
    )

    # Concatenate the train and validation datasets
    full_train_data = train_dataset.concatenate(val_dataset).shuffle(
        buffer_size=train_num_batches + val_num_batches
    )

    trainer = TabTransformerTrainer(
        hyperparameters=hyperparameters,
        config=config,
        job_name=job_name,
        trial_number=trial.number,
        train_dataset=full_train_data,
        val_dataset=full_train_data,
        distributed=distributed,
        strategy=strategy,
        model_dir=model_dir,
        logger=logger,
    )

    full_trained_model, _ = trainer.fit()

    # Save model
    logger.info(f"Saving model to tmp directory for trial {trial.number}...")
    local_model_dir = os.path.join("/tmp", f"model_trial_{trial.number}.keras")
    full_trained_model.save(local_model_dir)

    # Save training job name as an attribute
    trial.set_user_attr("job_name", job_name)

    return val_auc_pr


def main() -> int:
    # ---------------------------------- Set up ---------------------------------- #

    logger = get_logger("tab_transformer_hpo")

    # Hydra
    core.global_hydra.GlobalHydra.instance().clear()
    initialize(version_base="1.2", config_path="config", job_name="tab_transformer_hpo")
    # OmegaConf's return type is too broad, type narrowing is needed
    config: Dict[str, Any] = cast(
        Dict[str, Any],
        OmegaConf.to_container(compose(config_name="main"), resolve=True),
    )

    additional_args = {"study_name": str}

    args = add_additional_args(parser, additional_args)()

    job_name = args.training_env["job_name"]

    # Strategy for distributed training
    if args.test_mode:
        distributed = False
        strategy = None
    else:
        distributed = True
        strategy = tf.distribute.MirroredStrategy()

    # --------------------------------- Load data -------------------------------- #

    logger.info("Loading data...")

    if args.test_mode:
        logger.info("Running in test mode...")

        train_num_batches, train_dataset = test_sample(
            file_path=f"s3://{config['s3_bucket']}/{config['s3_key']}/input-data/train/train.csv",
            config=config,
            train=True,
            batch_size=config["tf_keras"]["batch_size"],
        )

        val_num_batches, val_dataset = test_sample(
            file_path=f"s3://{config['s3_bucket']}/{config['s3_key']}/input-data/val/val.csv",
            config=config,
            train=False,
            batch_size=config["tf_keras"]["batch_size"],
        )
    else:
        logger.info("Running in SageMaker mode...")

        # In distributed mode, need the number of replicas to scale the batch size
        num_replicas_in_sync = strategy.num_replicas_in_sync if strategy else 1
        global_batch_size = config["tf_keras"]["batch_size"] * num_replicas_in_sync

        train_num_batches, train_dataset = dataset_from_csv(
            file_path=os.path.join(args.train, "train.csv"),
            config=config,
            train=True,
            batch_size=global_batch_size,
        )

        val_num_batches, val_dataset = dataset_from_csv(
            file_path=os.path.join(args.val, "val.csv"),
            config=config,
            train=False,
            batch_size=global_batch_size,
        )

    # --------------------------------- HPO setup -------------------------------- #

    logger.info("Setting up optuna database...")

    db_url = get_db_url(
        host=args.host,
        db_name=args.db_name,
        db_secret=args.db_secret,
        region_name=args.region_name,
    )

    # ------------------------------- Optimization ------------------------------- #

    logger.info("Optimizing objective function...")

    def tf_objective_wrapper(trial: optuna.Trial) -> float:
        return tf_objective(
            trial=trial,
            config=config,
            job_name=job_name,
            train_dataset=train_dataset,
            train_num_batches=train_num_batches,
            val_dataset=val_dataset,
            val_num_batches=val_num_batches,
            distributed=distributed,
            strategy=strategy,
            model_dir=args.model_dir,
            logger=logger,
        )

    study = create_study(
        study_name=args.study_name, storage=db_url, direction="maximize"
    )
    study.optimize(tf_objective_wrapper, n_trials=args.n_trials)
    study_report(study=study, logger=logger)

    # ----------------------- Retrieve and save best model ----------------------- #

    logger.info("Retrieving best model and saving it to model directory...")

    best_model = tf.keras.models.load_model(
        os.path.join("/tmp", f"model_trial_{study.best_trial.number}.keras")
    )

    best_model.save(os.path.join(args.model_dir, "00000000"))

    return 0


if __name__ == "__main__":
    main()
