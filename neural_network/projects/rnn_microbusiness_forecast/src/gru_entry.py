from __future__ import annotations

import logging
import os
from functools import partial
from typing import Any, Dict, List, Tuple, cast

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Nopep8
import numpy as np
import polars as pl
import tensorflow as tf
from hydra import compose, core, initialize
from omegaconf import OmegaConf
from sklearn.model_selection import GroupKFold

from model_utils import add_additional_args, get_logger, parser

# --------------------------- Custom loss function --------------------------- #


class SymmetricMeanAbsolutePercentageError(tf.keras.losses.Loss):
    """
    This class implements the symmetric mean absolute percentage error (SMAPE) loss function usually of the form:

    SMAPE = (100 / n) * sum(abs(y_pred - y_true) / (abs(y_pred) + abs(y_true)))

    Note that the 0.5 factor in the denominator is not included in the loss function to keep the error between 0% and 100%.
    In addition, we implement another formulation of the SMAPE loss function, which is:

    SMAPE = (2 / n) * sum(abs(y_pred - y_true) / max((abs(y_pred) + abs(y_true)), epsilon)

    where epsilon is a small value to avoid division by zero. This formulation is used in the TorchMetrics library.
    """

    def __init__(self, **kwargs) -> None:
        """
        Constructor for the SymmetricMeanAbsolutePercentageError class.
        """
        super().__init__(**kwargs)

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        This function computes the SMAPE losses given a batch of predictions and targets.
        Both predictions and targets are assumed to be of shapes (batch_size, num_predictions).
        The returned tensor is of the shape (batch_size,), which contains the SMAPE loss for
        each training example. The `tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE` reduction
        type is used, which means that the loss is summed over all training examples in the
        batch and then divided by the batch size. Note that this is only supported with
        tf.distribute.Strategy when used in conjunction with the Keras compile/fit API (not
        with custom training loops).

        Parameters
        ----------
        y_true : tf.Tensor
            A tensor containing the ground truth values.
        y_pred : tf.Tensor
            A tensor containing the predicted values.

        Returns
        -------
        tf.Tensor
            A tensor containing the SMAPE loss for each training example.
        """
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)

        numerator = tf.abs(y_pred - y_true)
        # Clip the values to [epsilon, inf]
        denominator = tf.clip_by_value(
            tf.abs(y_true) + tf.abs(y_pred),
            clip_value_min=tf.keras.backend.epsilon(),
            clip_value_max=float("inf"),
        )

        # Sum across the `num_predictions` dimension and multiply by 2
        instance_losses = tf.multiply(
            2.0, tf.math.reduce_sum(numerator / denominator, axis=-1)
        )

        return instance_losses

    def get_config(self) -> Dict[str, Any]:
        """
        This function returns the configuration for the loss function.

        Returns
        -------
        Dict[str, Any]
            A dictionary containing the configuration for the loss function.
        """
        base_config = super(SymmetricMeanAbsolutePercentageError, self).get_config()
        return {**base_config}

    @classmethod
    def from_config(
        cls, config: Dict[str, Any]
    ) -> SymmetricMeanAbsolutePercentageError:
        """
        This function implements the from_config method of the custom loss function class.
        This is a no-op since all the parameters that are passed to the constructor of the
        class are already serializable.


        Parameters
        ----------
        config : Dict[str, Any]
            A dictionary containing the configuration for the loss function.

        Returns
        -------
        SymmetricMeanAbsolutePercentageError
            A loss function.
        """
        return cls(**config)


# -------------------------- Custom GRU memory cell -------------------------- #


class CustomGRUCell(tf.keras.layers.GRUCell):
    """
    This class implements a custom GRU cell that contains layer normalization, which
    is a form of normalization that often works better than batch normalization for
    recurrent neural networks. Instead of normalizing across the batch dimension, layer
    normalization normalizes across the features dimension. Layer normalization also
    behaves the same way duing training and inference, and it does not use exponential
    moving averages.
    """

    def __init__(self, layer_norm: tf.keras.layers.LayerNormalization, **kwargs):
        """
        Constructor for the CustomGRUCell class.

        Parameters
        ----------
        layer_norm : tf.keras.layers.LayerNormalization
            A layer normalization layer.
        **kwargs
            Keyword arguments passed to `tf.keras.layers.GRUCell`.
        """
        # Create a standard GRU cell with the given keyword arguments
        super().__init__(**kwargs)
        self.layer_norm = layer_norm
        self.activation_func = tf.keras.activations.get(self.activation)

    def call(
        self, inputs: tf.Tensor, states: tf.Tensor, **kwargs
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        This function implements the call method of the custom GRU cell.

        Parameters
        ----------
        inputs : tf.Tensor
            A 2D tensor, with shape of [batch, feature].
        states : tf.Tensor
            A 2D tensor with shape of [batch, units], which is the state from the previous time step.
            For timestep 0, the initial state provided by user will be fed to the cell.
        **kwargs
            Keyword arguments passed to the call method of `tf.keras.layers.GRUCell`.

        Returns
        -------
        Tuple[tf.Tensor, tf.Tensor]
            A tuple containing the new hidden for the current time step and the new hidden state but wrapped in a list in anticipation of potential RNN layers downstream.
            The hidden state is a 2D tensor with shape of [batch, units] where units means the number of hidden units of the cell.
        """
        # Call the parent class' call method, returning the new hidden state 'h', which is a convex combination of the previous hidden state and the candidate hidden state
        # New state is essentially the same as `h` but wrapped in a list to be consistent with the expected return format for RNN cells in Keras
        h, new_states = super().call(inputs, states, **kwargs)
        # Apply layer normalization and then activation
        normalized_h = self.activation_func(self.layer_norm(h))
        return normalized_h, new_states

    def get_config(self) -> Dict[str, Any]:
        """
        This function returns the configuration for the GRU cell. Since `self.activation_func`
        uses the parent class's `self.activation` attribute, we do not need to include `self.activation_func`
        in the configuration. It should be obtained from `super().get_config()` and can be passed to the
        `from_config` method directory for deserialization.

        Returns
        -------
        Dict[str, Any]
            A dictionary containing the configuration for the GRU cell.
        """
        base_config = super(CustomGRUCell, self).get_config()
        config = {"layer_norm": tf.keras.layers.serialize(self.layer_norm)}
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> CustomGRUCell:
        """
        This function implements the from_config method of the custom GRU cell class, deserializing
        the layer normalization layer and passing it to the constructor of the class. Note again
        that `self.activation_func` is not included here since passing the 'self.activation' attribute
        inside of `config` will automatically lead to the creation of the activation function correctly
        in the constructor.

        Parameters
        ----------
        config : Dict[str, Any]
            A dictionary containing the configuration for the GRU cell.

        Returns
        -------
        CustomGRUCell
            A GRU cell.
        """
        layer_norm_config = config.pop("layer_norm")
        layer_norm = tf.keras.layers.deserialize(layer_norm_config)
        return cls(layer_norm=layer_norm, **config)


# ------------------------------- Trainer class ------------------------------ #


class GRUTrainer(object):
    """
    Trainer class for GRU model. This is a sequence-to-vector model, which
    predicts the next five months of microbusiness density ratios in a single
    shot.
    """

    def __init__(
        self,
        hyperparameters: Dict[str, Any],
        config: Dict[str, Any],
        job_name: str,
        train_data: pl.DataFrame,
        target_raw_densities: np.ndarray,
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
        train_data : pl.DataFrame
            A Polars DataFrame containing the training data.
        target_raw_densities : np.ndarray
            A numpy array containing the target raw densities for computing the SMAPE loss after cross validation.
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
        self.train_data = train_data
        self.target_raw_densities = target_raw_densities
        self.distributed = distributed
        self.strategy = strategy
        self.model_dir = model_dir
        self.logger = logger

    def _create_optimizer(self) -> tf.keras.optimizers.Optimizer:
        """
        This function creates an Adam optimizer based on the hyperparameters.

        Returns
        -------
        optimizer : tf.keras.optimizers.Optimizer
            An Adam optimizer.
        """
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.hyperparameters["adam_initial_lr"],
            beta_1=self.hyperparameters["adam_beta_1"],
            beta_2=self.hyperparameters["adam_beta_2"],
            epsilon=self.hyperparameters["adam_epsilon"],
            clipnorm=self.hyperparameters["adam_clipnorm"],
        )

        return optimizer

    def _create_loss_fn(self) -> tf.keras.losses.Loss:
        """
        This function creates a loss function, which is the SMAPE.

        Returns
        -------
        tf.keras.losses.Loss
            A loss function.
        """
        loss_fn = SymmetricMeanAbsolutePercentageError(
            reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE, name="smape"
        )

        return loss_fn

    def _create_metrics(self) -> List[tf.keras.metrics.Metric]:
        """
        This function creates a list of metrics for model evaluation.

        Returns
        -------
        List[tf.keras.metrics.Metric]
            A list of metrics--- MSE, MAPE, and MAE.
        """
        metrics = [
            tf.keras.metrics.MeanSquaredError(name="mse"),
            tf.keras.metrics.MeanAbsolutePercentageError(name="mape"),
            tf.keras.metrics.MeanAbsoluteError(name="mae"),
        ]

        return metrics

    def _create_model(self) -> tf.keras.Model:
        """
        This function creates a GRU model.

        Returns
        -------
        tf.keras.Model
            A compiled GRU model.
        """
        # Default GRU cell
        default_gru_cell = partial(
            CustomGRUCell,
            activation="tanh",
            recurrent_activation="sigmoid",
            kernel_initializer="he_normal",
            # Applied to the weight matrices that transform input X_t
            kernel_regularizer=tf.keras.regularizers.l2(
                l2=self.hyperparameters["gru_kernel_l2_factor"]
            ),
            # Applied to the weight matrices that transform hidden state h_t-1
            recurrent_regularizer=tf.keras.regularizers.l2(
                l2=self.hyperparameters["gru_recurrent_l2_factor"]
            ),
            dropout=self.hyperparameters["gru_dropout"],
            recurrent_dropout=self.hyperparameters["gru_recurrent_dropout"],
        )

        # ------------------------------------ GRU ----------------------------------- #

        inputs = tf.keras.Input(
            shape=(
                self.config["gru"]["series_len"]
                - self.config["gru"]["num_predictions"]
                - 1,
                1,
            ),
            name="input_layer",
        )
        # Normalize the inputs
        normalized_inputs = tf.keras.layers.LayerNormalization(name="layer_norm_input")(
            inputs
        )

        # Stacked GRU layers
        x = tf.keras.layers.RNN(
            cell=default_gru_cell(
                layer_norm=tf.keras.layers.LayerNormalization(name="layer_norm_gru_0"),
                units=self.hyperparameters["gru_units_0"],
                name="gru_cell_0",
            ),
            return_sequences=(
                False if self.hyperparameters["gru_num_layers"] == 1 else True
            ),  # If using a single GRU layer, them this layer should only return the last output `h_t`
            name="gru_layer_0",
        )(normalized_inputs)
        # Set return_sequences = True to output a sequence of vectors each with 'gru_units_i' elements
        for i in range(1, self.hyperparameters["gru_num_layers"]):
            x = tf.keras.layers.RNN(
                cell=default_gru_cell(
                    layer_norm=tf.keras.layers.LayerNormalization(
                        name=f"layer_norm_gru_{i}"
                    ),
                    units=self.hyperparameters[f"gru_units_{i}"],
                    name=f"gru_cell_{i}",
                ),
                # If this is the last GRU layer, set return_sequences = False
                return_sequences=(
                    False if i == (self.hyperparameters["gru_num_layers"] - 1) else True
                ),
                name=f"gru_layer_{i}",
            )(x)

        # Dense layer
        outputs = tf.keras.layers.Dense(
            units=self.config["gru"]["num_predictions"], name="output_layer"
        )(x)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)

        # ---------------------------------- Compile --------------------------------- #

        optimizer = self._create_optimizer()
        loss_fn = self._create_loss_fn()
        metrics = self._create_metrics()
        model.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics)

        return model

    def _generate_sample_weights(
        self, train_indices: np.ndarray, window_size: int
    ) -> np.ndarray:
        """
        Given the training indices for a particular fold, this function generates
        sample weights for the training examples. For each county (group) in the
        training data, we increase the weights for the more recent months since they
        have more predictive power.

        Parameters
        ----------
        train_indices : np.ndarray
            A numpy array containing the training indices for a particular fold.
        window_size : int
            The size of the window (number of most recent months) for which to increase the sample weights.

        Returns
        -------
        np.ndarray
            A numpy array containing the sample weights for the training examples.
        """
        # Number of counties in the training data for this fold
        num_counties = len(train_indices) // self.config["gru"]["series_len"]
        # Weights for a single county initilized to 1
        county_weights = np.ones(shape=(self.config["gru"]["series_len"],))
        # Double the weights for the most recent months
        county_weights[-window_size:] = 2

        # Repeat county weight pattern 'num_counties' times
        sample_weights = np.tile(county_weights, num_counties)

        return sample_weights

    def _fit(
        self,
        fold_train: Dict[str, np.ndarray],
        fold_val: Dict[str, np.ndarray],
        fold_sample_weights: np.ndarray,
    ) -> tf.keras.Model:
        """
        This function trains the model.

        Parameters
        ----------
        fold_train : Dict[str, np.ndarray]
            A dictionary containing the training data for one particular fold--- `X_train` and `y_train`.
        fold_val : Dict[str, np.ndarray]
            A dictionary containing the validation data for one particular fold--- `X_val` and `y_val`.
        fold_sample_weights : np.ndarray
            A numpy array containing the sample weights for the training examples.

        Return
        ------
        tf.keras.Model
            A trained model.
        """
        # ------------------------------- Create model ------------------------------- #

        if self.distributed:
            with self.strategy.scope():
                model = self._create_model()
        else:
            model = self._create_model()

        # --------------------------------- Callbacks -------------------------------- #

        # The 'on_train_begin' method resets the 'self.wait' attribute to 0 so this can be reused across multiple calls to 'fit'
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=self.config["patience"],
            mode="min",
            restore_best_weights=True,
        )
        back_and_restore = tf.keras.callbacks.BackupAndRestore(
            backup_dir=os.path.join(os.getcwd(), "backup"),
            # Delete the backup directory after the training is completed, so the next call to 'fit' will create a new backup directory
            delete_checkpoint=True,
        )
        callbacks = [early_stopping, back_and_restore]

        # --------------------------------- Fit model -------------------------------- #

        model.fit(
            x=fold_train["X_train"],
            y=fold_train["y_train"],
            batch_size=self.hyperparameters["fit_batch_size"],
            epochs=self.hyperparameters["fit_epochs"],
            callbacks=callbacks,
            validation_data=(fold_val["X_val"], fold_val["y_val"]),
            sample_weight=fold_sample_weights,
        )

        return model

    def cross_validate(self) -> None:
        """
        This function performs cross validation, returning the
        average best validation SMAPE losses across all folds.
        We use the GroupKFold cross validation strategy, which
        forms a complete partition of all the cfips in the training
        data. In other words, each cfips will appear exactly once in
        the validation set over all folds. Given a number of folds,
        the number validation examples in each fold is approximately
        equal to `(num_training_examples / num_folds)`. Since each cfips
        has `seq_len` number of training examples, each validation
        set contains about `(num_training_examples / num_folds) * (1 /
        seq_len)` unique cfips.

        For example, if we have 1000 training examples and 5 folds, then
        each validation set contains about 200 unique cfips. If `seq_len`
        is 60, then each validation set contains about 3 unique cfips.

        The goal is to expose the model to as many unique cfips as possible
        during mulple rounds of training. This is because we want the model to
        learn the general patterns of microbusiness density ratios across all
        cfips in the training data.

        Returns
        -------
        None
        """
        # If not running in SageMaker mode, set the number of folds to 2 (smallest possible for Scikit-Learn)
        num_folds = 2 if not self.distributed else self.config["num_folds"]
        # Out-of-fold predictions container (num_training_examples, num_predictions)
        oof_preds = np.zeros(
            shape=(self.train_data.shape[0], self.config["gru"]["num_predictions"])
        )
        # Training and target data
        X_train = self.train_data.select(
            [col for col in self.train_data.columns if col.startswith("x")]
        ).to_numpy()
        y_train = self.train_data.select(
            [col for col in self.train_data.columns if col.startswith("y")]
        ).to_numpy()
        cfips = self.train_data["cfips"].to_numpy()
        baseline_raw_density = self.train_data["baseline_raw_density"]

        # ----------------------------- Cross validation ----------------------------- #

        gkf = GroupKFold(n_splits=num_folds)
        gkf_splitter = gkf.split(X=X_train, y=y_train, groups=cfips)
        for fold, (train_indices, val_indices) in enumerate(gkf_splitter):
            self.logger.info(f"Fold {fold + 1} / {num_folds}")

            # Fold data
            fold_train = {
                "X_train": X_train[train_indices],
                "y_train": y_train[train_indices],
            }
            fold_val = {"X_val": X_train[val_indices], "y_val": y_train[val_indices]}

            # Fold sample weights
            fold_sample_weights = self._generate_sample_weights(
                train_indices=train_indices,
                window_size=self.hyperparameters["fit_sample_weights_window_size"],
            )

            # Fold model training
            fold_model = self._fit(
                fold_train=fold_train,
                fold_val=fold_val,
                fold_sample_weights=fold_sample_weights,
            )

            self.logger.info(f"Saving model for fold {fold + 1} / {num_folds}")

            # Fold model persistence (use new keras format rather than SavedModel format since we do not need to serve the model on SageMaker with TF serving)
            fold_model.save(
                os.path.join(self.model_dir, f"fold_{fold + 1}_model.keras")
            )

            self.logger.info(f"Predicting for fold {fold + 1} / {num_folds}")

            # Make predictions on the validation set
            fold_oof_preds = fold_model.predict(fold_val["X_val"])
            # Assign validation predictions to the corresponding indices in the out-of-fold predictions container
            # This ensure that the out-of-fold predictions are in the same order as the training data
            oof_preds[val_indices] = fold_oof_preds

        # Convert to raw densities by recursively multiplying the baseline raw density by the predicted ratios
        oof_predicted_densities = np.zeros_like(oof_preds)
        oof_predicted_densities[:, 0] = (
            baseline_raw_density.to_numpy() * oof_preds[:, 0]
        )
        for k in range(4):
            oof_predicted_densities[:, k + 1] = (
                oof_predicted_densities[:, k] * oof_predicted_densities[:, k + 1]
            )

        # Compute the SMAPE loss between predicted densities and target densities
        loss_fn = self._create_loss_fn()
        avg_smape_loss = loss_fn(
            y_true=self.target_raw_densities, y_pred=oof_predicted_densities
        ).numpy()

        self.logger.info(f"Best validation SMAPE over all folds: {avg_smape_loss}")

        return None


def main() -> int:
    # ---------------------------------- Set up ---------------------------------- #

    logger = get_logger("gru_hpo")

    # Hydra
    core.global_hydra.GlobalHydra.instance().clear()
    initialize(version_base="1.2", config_path="config", job_name="processing_job")
    config: Dict[str, Any] = cast(
        Dict[str, Any],
        OmegaConf.to_container(compose(config_name="main"), resolve=True),
    )

    additional_args = {
        # Architecture hyperparameters
        "gru_units_0": int,
        "gru_units_1": int,
        "gru_units_2": int,
        "gru_units_3": int,
        "gru_num_layers": int,
        "gru_kernel_l2_factor": float,
        "gru_recurrent_l2_factor": float,
        "gru_dropout": float,
        "gru_recurrent_dropout": float,
        # Optimization and fit hyperparameters
        "adam_initial_lr": float,
        "adam_beta_1": float,
        "adam_beta_2": float,
        "adam_epsilon": float,
        "adam_clipnorm": float,
        "fit_batch_size": int,
        "fit_epochs": int,
        "fit_sample_weights_window_size": int,
    }

    args = add_additional_args(parser_func=parser, additional_args=additional_args)()

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

        # Keep only first two counties for testing
        train_data = pl.read_csv(os.path.join(args.train, "train.csv"))[
            : 2 * config["gru"]["series_len"]
        ]

        # Target raw densities
        target_raw_densities = np.load(
            os.path.join(args.train, "densities_target.npy")
        )[: 2 * config["gru"]["series_len"]]
    else:
        logger.info("Running in SageMaker mode...")

        # Load training data
        train_data = pl.read_csv(os.path.join(args.train, "train.csv"))

        # Target raw densities
        target_raw_densities = np.load(os.path.join(args.train, "densities_target.npy"))

    # --------------------------------- Train model --------------------------------- #

    # In distributed mode, need the number of replicas to scale the batch size
    num_replicas_in_sync = strategy.num_replicas_in_sync if strategy else 1
    global_batch_size = args.fit_batch_size * num_replicas_in_sync

    trainer = GRUTrainer(
        hyperparameters={
            # Architecture hyperparameters
            "gru_units_0": args.gru_units_0,
            "gru_units_1": args.gru_units_1,
            "gru_units_2": args.gru_units_2,
            "gru_units_3": args.gru_units_3,
            "gru_num_layers": args.gru_num_layers,
            "gru_kernel_l2_factor": args.gru_kernel_l2_factor,
            "gru_recurrent_l2_factor": args.gru_recurrent_l2_factor,
            "gru_dropout": args.gru_dropout,
            "gru_recurrent_dropout": args.gru_recurrent_dropout,
            # Optimization and fit hyperparameters
            "adam_initial_lr": args.adam_initial_lr,
            "adam_beta_1": args.adam_beta_1,
            "adam_beta_2": args.adam_beta_2,
            "adam_epsilon": args.adam_epsilon,
            "adam_clipnorm": args.adam_clipnorm,
            "fit_batch_size": global_batch_size,
            "fit_epochs": args.fit_epochs,
            "fit_sample_weights_window_size": args.fit_sample_weights_window_size,
        },
        config=config,
        job_name=job_name,
        train_data=train_data,
        target_raw_densities=target_raw_densities,
        distributed=distributed,
        strategy=strategy,
        model_dir=args.model_dir,
        logger=logger,
    )

    trainer.cross_validate()

    return 0


if __name__ == "__main__":
    main()
