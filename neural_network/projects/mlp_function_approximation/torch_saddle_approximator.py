import argparse
import logging
import os
from copy import deepcopy
from typing import Callable, Dict, List, Tuple, Union

import numpy as np
import optuna
import torch
from IPython.display import display
from optuna.trial import TrialState
from torch.utils.data import DataLoader, TensorDataset

# --------------------------- Monkey saddle surface -------------------------- #


def monkey_saddle(
    x: np.ndarray,
    y: np.ndarray,
    theta: float = np.pi / 4,
    flip: list[bool] = [False, False, False],
    translate: list[float] = [0.0, 0.0, 0.0],
    scale: list[float] = [1.0, 1.0, 1.0],
) -> np.ndarray:
    """
    Generate a rotated or translated or scaled or all of the above monkey saddle function.

    Parameters
    ----------
    x : np.ndarray
        First input array.
    y : np.ndarray
        Second input array.
    theta : float, optional
        Rotation angle in radians, by default np.pi/4.
    flip : list[bool], optional
        If True, flip the saddle function along the corresponding axis, by default [False, False, False].
    translate : list[float], optional
        Value to translate the saddle function along the corresponding axis, by default [0.0, 0.0, 0.0].
    scale : list[float], optional
        Value to scale the saddle function along the corresponding axis, by default [1.0, 1.0, 1.0].

    Returns
    -------
    np.ndarray
        The output values of the rotated and transformed saddle function.
    """
    # Rotation matrix
    rotation_matrix = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    )

    # Flatten the input arrays and stack them into a 2 x n matrix so we can apply the rotation matrix
    inputs = np.vstack([x.flatten(), y.flatten()])
    x_rot, y_rot = np.dot(rotation_matrix, inputs)

    # Reshape the rotated arrays back to the original shape of the input arrays
    x_rot = x_rot.reshape(x.shape)
    y_rot = y_rot.reshape(y.shape)

    # Compute the z values
    saddle = x_rot**3 - 3 * x_rot * y_rot**2

    # Apply flip
    if flip[0]:
        x_rot = -x_rot
    if flip[1]:
        y_rot = -y_rot
    if flip[2]:
        saddle = -saddle

    # Apply translation
    x_rot = x_rot + translate[0]
    y_rot = y_rot + translate[1]
    saddle = saddle + translate[2]

    # Apply scale
    x_rot = x_rot * scale[0]
    y_rot = y_rot * scale[1]
    saddle = saddle * scale[2]

    return saddle


# ------------------------------ Data generator ------------------------------ #


def generate_monkey_saddle_data(
    start: float = -np.e, end: float = np.e, num_samples: int = 500, **kwargs
) -> TensorDataset:
    """
    Generate training data for sine wave.

    Parameters
    ----------
    start : float, optional
        Start of the interval, by default -np.pi.
    end : float, optional
        End of the interval, by default np.pi.
    num_samples : int, optional
        Number of samples to generate, by default 2000.
    **kwargs
        Additional keyword arguments to be passed to the monkey_saddle function.

    Returns
    -------
    TensorDataset
        A TensorDataset containing the input and target data.
    """
    # X and Y are each num_samples x num_samples matrices
    x, y = np.meshgrid(
        np.linspace(start, end, num_samples), np.linspace(start, end, num_samples)
    )
    # Z is the matrix where z[i, j] is the output of the monkey_saddle function for x[i, j] and y[i, j]
    z = monkey_saddle(x, y, **kwargs)

    # Input matrix (num_samples**2 x 2) and target (num_samples**2 x 1)
    X = torch.tensor(np.array([x.flatten(), y.flatten()]).T, dtype=torch.float32)
    y = torch.tensor(z.flatten(), dtype=torch.float32).reshape(-1, 1)

    data = TensorDataset(X, y)

    return data


# ----------------------------- Regression model ----------------------------- #


class MLPRegressor(torch.nn.Module):
    """
    A configurable Multi-Layer Perceptron (MLP) regression model.

    Attributes
    ----------
    activation : torch.nn.Module
        The activation function.
    initialization : function
        The weight initialization function.
    layers : torch.nn.ModuleList
        The layers of the MLP.
    """

    def __init__(
        self,
        num_layers: int,
        units_per_layer: List[int],
        initializer: str,
        input_shape: int = 2,
    ):
        """
        Instantiate

        Parameters
        ----------
        num_layers : int
            The number of layers in the MLP.
        units_per_layer : List[int]
            A list of integers specifying the number of units in each layer.
        initializer : str
            The weight initialization method to use. Options are 'xavier_normal' and 'kaiming_normal'.
        input_shape : int, optional
            The shape of the input tensor, by default 2.
        """
        super(MLPRegressor, self).__init__()

        # -------------------------------- Activation -------------------------------- #

        # Properly registered such all module methods can use this dictionary
        self.activations = torch.nn.ModuleDict(
            [
                ["relu", torch.nn.ReLU()],
                ["tanh", torch.nn.Tanh()],
                ["leaky_relu", torch.nn.LeakyReLU()],
            ]
        )
        # Do not use ModuleDict since the initialization functions are not subclasses of torch.nn.Module
        initializers = {
            "xavier_normal": torch.nn.init.xavier_normal_,
            "kaiming_normal": torch.nn.init.kaiming_normal_,
        }
        self.initializer = initializers[initializer]

        # ----------------------------- Number of layers ----------------------------- #

        self.layers = torch.nn.ModuleList()
        # Input layer
        self.layers.append(
            torch.nn.Linear(in_features=input_shape, out_features=units_per_layer[0])
        )
        # Hidden layers start at index 1 and index backwards to get the previous layer's number of units
        for i in range(1, num_layers):
            self.layers.append(
                torch.nn.Linear(
                    in_features=units_per_layer[i - 1], out_features=units_per_layer[i]
                )
            )
        # Output layer
        self.layers.append(
            torch.nn.Linear(in_features=units_per_layer[-1], out_features=1)
        )

        # ------------------------------ Initialization ------------------------------ #

        for layer in self.layers:
            self.initializer(layer.weight)
            torch.nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor, activation: str) -> torch.Tensor:
        """
        Forward pass of the network.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.
        activation : str
            The activation function to use. Options are 'relu', 'tanh', 'leaky_relu'.

        Returns
        -------
        torch.Tensor
            The output tensor.
        """
        for i, layer in enumerate(self.layers):
            x = layer(x)
            # Apply activation to all layers except for the output layer (linear)
            if i != len(self.layers) - 1:
                x = self.activations[activation](x)
        return x


# ------------------------------ Early stopping ------------------------------ #


class EarlyStopping:
    """
    This class implements early stopping for training the MLPRegressor model.
    """

    def __init__(
        self, patience: int = 3, min_delta: float = 0.0, restore_best_model: bool = True
    ):
        """
        Instantiate the EarlyStopping class.

        Parameters
        ----------
        patience : int, optional
            Number of epochs with no improvement after which training will be stopped, by default 3.
        min_delta : float, optional
            Minimum change in the monitored quantity to qualify as an improvement, i.e. an absolute change of
            less than min_delta, will count as no improvement., by default 0.0.
        restore_best_model : bool, optional
            Whether to restore the weights of the model from the epoch with the best loss, by default True.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = np.inf
        self.counter = 0
        self.epoch = 0
        self.restore_best_model = restore_best_model
        self.best_state_dict = None

    def __call__(
        self, model: MLPRegressor, val_loss: float, epoch: int
    ) -> Tuple[bool, float]:
        """
        Check if the training should stop.

        Parameters
        ----------
        model : MLPRegressor
            The model from which we can get the state dict.
        val_loss : float
            The validation loss.
        epoch : int
            The current epoch number, which will becomes the best epoch if the validation loss associated with this epoch is an improvement.

        Returns
        -------
        Tuple[bool, float]
            A tuple of a boolean indicating whether the training should stop and the best validation loss.
        """
        # If the improvement (self.best_loss - val_loss) is greater than minimum delta, update the best loss
        # This includes the initial case where self.best_loss is np.inf, where we carry out the first update
        if (self.best_loss == np.inf) or np.less(
            self.min_delta, (self.best_loss - val_loss)
        ):
            self.best_loss = val_loss
            self.best_epoch = epoch
            # If the loss improved greater than minimum delta, reset the counter since we stop only if consecutive epochs do not improve the loss
            # Training may occasionally be stuck at a local minima, so we want to give it a chance to get out of it
            self.counter = 0
            if self.restore_best_model:
                # OrderedDict is mutable, so we need to make a deep copy to prevent the best_state_dict from being modified
                self.best_state_dict = deepcopy(model.state_dict())
        # If the improvement (self.best_loss - val_loss) is less than minimum delta, increment the counter
        # This includes the case where the val_loss is the same as or greater than the best loss (since min_delta is non-negative)
        elif np.less((self.best_loss - val_loss), self.min_delta):
            self.counter += 1
            if self.counter == self.patience:
                return True, self.best_loss
        return False, self.best_loss


# ----------------------------- Training function ---------------------------- #


def trainer(
    train: TensorDataset,
    val: TensorDataset,
    num_layers: int,
    units_per_layer: List[int],
    activation: str,
    initializer: str,
    epochs: int,
    batch_size: int,
    optimizer: str,
    learning_rate: float,
    logger: logging.Logger,
    verbose: int = 1,
    patience: int = 3,
    min_delta: float = 1e-3,
    restore_best_model: bool = True,
    loss_fn: str = "MSELoss",
    gamma: float = 0.8,
    num_workers: int = 2,
    input_shape: int = 2,
) -> Tuple[MLPRegressor, float]:
    """
    Train the MLPRegressor model.

    Parameters
    ----------
    train: TensorDataset
        The training data.
    val: TensorDataset
        The validation data.
    num_layers : int
        The number of dense layers in the MLP.
    units_per_layer : List[int]
        A list of integers specifying the number of units in each layer.
    activation : str
        The activation function to use. Options are 'relu', 'tanh', 'leaky_relu'.
    initializer : str
        The weight initialization method to use. Options are 'xavier_normal' and 'kaiming_normal'.
    epochs : int
        The number of epochs to train the model.
    batch_size : int
        The batch size.
    optimizer : str
        The optimizer to use. Options are 'Adam', 'SGD', and 'RMSprop'.
    learning_rate : float
        The learning rate.
    logger : logging.Logger
        The logger.
    verbose : int, optional
        The verbosity level, by default 1.
    patience : int, optional
        Number of epochs with no improvement after which training will be stopped, by default 3.
    min_delta : float, optional
        Minimum change in the monitored quantity to qualify as an improvement, i.e. an absolute change of
        less than min_delta, will count as no improvement., by default 1e-3.
    restore_best_model : bool, optional
        Whether to restore the weights of the model from the epoch with the best loss, by default True.
    loss_fn : str, optional
        The loss function to use. Options are 'MSELoss', 'L1Loss', and 'HuberLoss', by default 'MSELoss'.
    gamma : float, optional
        The learning rate decay factor, by default 0.8.
    num_workers : int, optional
        The number of workers to use for loading the data, by default 2.
    input_shape : int, optional
        The shape of the input tensor, by default 2.

    Returns
    -------
    Tuple[MLPRegressor, float]
        A tuple of the trained model and the validation loss.
    """
    # ----------------------------------- Model ---------------------------------- #

    model = MLPRegressor(
        num_layers=num_layers,
        units_per_layer=units_per_layer,
        initializer=initializer,
        input_shape=input_shape,
    )
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    model.to(device)

    logger.info(f"Instantiated model and moved it to {device} device...")

    # ----------------------------------- Data ----------------------------------- #

    logger.info(
        f"Loading data with batch size {batch_size} and {num_workers} workers..."
    )

    train_loader = DataLoader(
        train, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    # ------------------------ Loss function and optimizer ----------------------- #

    logger.info(
        f"Using the {loss_fn} loss function and the {optimizer} optimizer with initial learning rate {learning_rate}..."
    )

    loss_fns = {
        "MSELoss": torch.nn.MSELoss(reduction="mean"),  # L2 loss
        "L1Loss": torch.nn.L1Loss(reduction="mean"),  # MAE loss
        "HuberLoss": torch.nn.HuberLoss(reduction="mean", delta=1.0),
    }
    optimizers = {
        "Adam": torch.optim.Adam(params=model.parameters(), lr=learning_rate),
        "SGD": torch.optim.SGD(params=model.parameters(), lr=learning_rate),
        "RMSprop": torch.optim.RMSprop(params=model.parameters(), lr=learning_rate),
    }
    loss_fn = loss_fns[loss_fn]
    optimizer = optimizers[optimizer]
    lr_schedulers = torch.optim.lr_scheduler.ExponentialLR(
        optimizer=optimizer, gamma=gamma
    )

    # ----------------------------- Early stopping ----------------------------- #

    early_stopper = EarlyStopping(
        patience=patience, min_delta=min_delta, restore_best_model=restore_best_model
    )

    # ------------------------------- Training loop ------------------------------ #

    for epoch in range(epochs):
        logger.info(f"Training for epoch number {epoch + 1}...")

        train_running_loss = 0.0
        # To track the number of batches in one epoch
        train_epoch_steps = 0

        # Train mode
        model.train()
        for train_batch_index, (X_train_batch, y_train_batch) in enumerate(
            train_loader
        ):
            # Move the data to the device
            X_train_batch = X_train_batch.to(device)
            y_train_batch = y_train_batch.to(device)

            # Zero parameter gradients
            optimizer.zero_grad()

            # Forward pass __call__ method of the model calls the forward method
            outputs = model(X_train_batch, activation=activation)
            # Compute the loss (a scalar tf.Tensor if reduction is 'mean' or 'sum')
            train_loss = loss_fn(outputs, y_train_batch)
            # Compute the gradient of the loss with respect to the model parameters
            train_loss.backward()
            # A single optimization step (parameter update)
            optimizer.step()

            train_running_loss += (
                train_loss.item()
            )  # Use loss.item() to return the scalar loss as a float
            train_epoch_steps += 1
            if train_batch_index % 300 == 299 and verbose > 0:
                logger.info(
                    f"Epoch {epoch + 1} | Batch {train_batch_index + 1} | Train Loss {train_running_loss / train_epoch_steps}"
                )
                # Reset running loss
                train_running_loss = 0.0

        # Scheduling should be applied after optimizer’s update
        lr_schedulers.step()

        # ------------------------------ Validation loop ----------------------------- #

        logger.info(f"Validating for epoch number {epoch + 1}...")

        val_running_loss = 0.0
        # To track the number of batches in one epoch
        val_epochs_steps = 0
        # To be used to compute the average validation loss over all batches for one epoch
        val_total_loss = 0.0

        # Switch to evaluation mode (e.g. batchnorm, dropout, etc. behave differently during training and evaluation)
        model.eval()
        for val_batch_index, (X_val_batch, y_val_batch) in enumerate(val_loader):
            # Inference mode used in conjunction with model.eval()
            with torch.no_grad():
                # Move the data to the device
                X_val_batch = X_val_batch.to(device)
                y_val_batch = y_val_batch.to(device)

                # Mini-batch predictions with shape (batch_size, 1)
                outputs = model(X_val_batch, activation=activation)
                val_loss = loss_fn(outputs, y_val_batch)

                val_running_loss += val_loss.item()
                # This is not reset to 0.0
                val_total_loss += val_loss.item()
                val_epochs_steps += 1
                if val_batch_index % 300 == 299 and verbose > 0:
                    logger.info(
                        f"Epoch {epoch + 1} | Batch {val_batch_index + 1} | Val Loss {val_running_loss / val_epochs_steps}"
                    )
                    val_running_loss = 0.0

        # ------------------------------ Early stopping ------------------------------ #

        # Computes the average validation loss for this epoch
        avg_val_loss = val_total_loss / val_epochs_steps
        logger.info(f"Epoch {epoch + 1} average validation loss: {avg_val_loss}")

        # Only start early stopping after the first epoch
        if epoch > 0:
            should_stop, best_val_loss = early_stopper(model, avg_val_loss, epoch)

            if should_stop:
                logger.info(f"Early stopping activated after {epoch + 1} epochs...")
                logger.info(
                    f"Best validation loss achieved at epoch {early_stopper.best_epoch + 1}: {best_val_loss}"
                )

                # If early stopping was activated and restore best model is True
                if restore_best_model:
                    matched_keys = model.load_state_dict(early_stopper.best_state_dict)
                return model, best_val_loss

    # If early stopping was not activate, simply return the model and the best validation loss (from the last epoch)
    return model, best_val_loss


# ------------------------- Optuna objective function ------------------------ #


def objective(
    trial: optuna.Trial,
    train: TensorDataset,
    val: TensorDataset,
    logger: logging.Logger,
    verbose: int = 1,
    patience: int = 3,
    min_delta: float = 1e-3,
    restore_best_model: bool = True,
    loss_fn: str = "MSELoss",
    gamma: float = 0.8,
    num_workers: int = 2,
    input_shape: int = 2,
) -> float:
    """
    Optuna surrogate objective function.

    Parameters
    ----------
    train: TensorDataset
        The training data.
    val: TensorDataset
        The validation data.
    logger : logging.Logger
        The logger.
    verbose : int, optional
        The verbosity level, by default 1.
    patience : int, optional
        Number of epochs with no improvement after which training will be stopped, by default 3.
    min_delta : float, optional
        Minimum change in the monitored quantity to qualify as an improvement, i.e. an absolute change of
        less than min_delta, will count as no improvement., by default 1e-3.
    restore_best_model : bool, optional
        Whether to restore the weights of the model from the epoch with the best loss, by default True.
    loss_fn : str, optional
        The loss function to use. Options are 'MSELoss', 'L1Loss', and 'HuberLoss', by default 'MSELoss'.
    gamma : float, optional
        The learning rate decay factor, by default 0.8.
    num_workers : int, optional
        The number of workers to use for loading the data, by default 2.
    input_shape : int, optional
        The shape of the input tensor, by default 2.

    Returns
    -------
    float
        The validation loss that is computed over all mini-batches for a given epoch.
    """
    hyperparameters = {
        "num_layers": trial.suggest_int("num_layers", 1, 3),
        "activation": trial.suggest_categorical("activation", ["relu", "tanh"]),
        "initializer": trial.suggest_categorical(
            "initializer", ["xavier_normal", "kaiming_normal"]
        ),
        "epochs": trial.suggest_int("epochs", 1, 10),
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
        "optimizer": trial.suggest_categorical("optimizer", ["Adam", "SGD", "RMSprop"]),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
    }
    hyperparameters["units_per_layer"] = [
        trial.suggest_int(f"num_units_layer_{i}", 64, 256, step=16)
        for i in range(hyperparameters["num_layers"])
    ]

    logger.info(f"Begin training for trial {trial.number}...")

    model, best_val_loss = trainer(
        train=train,
        val=val,
        logger=logger,
        verbose=verbose,
        patience=patience,
        min_delta=min_delta,
        restore_best_model=restore_best_model,
        loss_fn=loss_fn,
        gamma=gamma,
        num_workers=num_workers,
        input_shape=input_shape,
        **hyperparameters,
    )

    return best_val_loss


if __name__ == "__main__":
    from custom_utils import (create_study, get_logger, plot_2d_surfaces,
                              study_report)

    logger = get_logger(name="tf_monkey_saddle_approximator")

    # Create a directory to store optuna study results
    output_dir = os.path.join(__file__, "outputs/saddle_function_approximator_hpo")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # -------------------------- Command line arguments -------------------------- #

    parser = argparse.ArgumentParser()
    # Data generation
    parser.add_argument(
        "--num_samples",
        type=int,
        default=500,
        help="Number of training samples to generate",
    )
    parser.add_argument(
        "--start",
        type=float,
        default=-np.e,
        help="Start of the interval for data generation on the real line",
    )
    parser.add_argument(
        "--end",
        type=float,
        default=np.e,
        help="End of the interval for data generation on the real line",
    )
    parser.add_argument(
        "--theta", type=float, default=np.pi / 4, help="Rotation angle in radians"
    )
    parser.add_argument(
        "--flip",
        type=int,
        nargs="+",
        default=[1, 1, 1],
        help="Whether to flip the saddle function along the corresponding axis--- x, y, z",
    )
    parser.add_argument(
        "--translate",
        type=float,
        nargs="+",
        default=[0.0, 0.0, 0.0],
        help="Positve or negative floats to translate the saddle function along the corresponding axis--- x, y, z",
    )
    parser.add_argument(
        "--scale",
        type=float,
        nargs="+",
        default=[1.0, 1.0, 1.0],
        help="Values to scale the saddle function along the corresponding axis--- x, y, z",
    )
    # Training hyperparameters not tuned by optuna
    parser.add_argument("--verbose", type=int, default=1, help="The verbosity level")
    parser.add_argument(
        "--patience",
        type=int,
        default=3,
        help="Number of epochs with no improvement after which training will be stopped",
    )
    parser.add_argument(
        "--min_delta",
        type=float,
        default=1e-3,
        help="Minimum change in the monitored quantity to qualify as an improvement",
    )
    parser.add_argument(
        "--restore_best_model",
        type=int,
        default=1,
        help="Whether to restore the weights of the model from the epoch with the best loss",
    )
    parser.add_argument(
        "--loss_fn", type=str, default="MSELoss", help="The loss function to use"
    )
    parser.add_argument(
        "--gamma", type=float, default=0.8, help="The learning rate decay factor"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=2,
        help="The number of workers to use for loading the data",
    )
    parser.add_argument(
        "--input_shape", type=int, default=2, help="The shape of the input tensor"
    )
    # Optuna
    parser.add_argument(
        "--n_trials", type=int, default=5, help="Number of hyperparameter trials to run"
    )
    parser.add_argument(
        "--study_name",
        type=str,
        default="saddle_approximator",
        help="Name of the study",
    )

    args, _ = parser.parse_known_args()

    # ------------------------------- Simulate data ------------------------------ #

    logger.info(f"Generating training data with {args.num_samples} samples...")

    train_data = generate_monkey_saddle_data(
        start=args.start,
        end=args.end,
        num_samples=args.num_samples,
        theta=args.theta,
        flip=args.flip,
        translate=args.translate,
        scale=args.scale,
    )
    val_data = deepcopy(train_data)

    # ---------------------------------- Optuna ---------------------------------- #

    # Wrapper for objective
    def objective_wrapper(trial: optuna.Trial) -> Callable:
        return objective(
            trial=trial,
            train=train_data,
            val=val_data,
            logger=logger,
            verbose=args.verbose,
            patience=args.patience,
            min_delta=args.min_delta,
            restore_best_model=args.restore_best_model,
            loss_fn=args.loss_fn,
            gamma=args.gamma,
            num_workers=args.num_workers,
            input_shape=args.input_shape,
        )

    storage = "sqlite:///{}".format(
        os.path.join(output_dir, "saddle_approximator_hpo.db")
    )
    study = create_study(
        study_name=args.study_name, storage=storage, direction="minimize"
    )

    logger.info(f"Begin hyperparameter optimization with {args.n_trials} trials...")

    study.optimize(func=objective_wrapper, n_trials=args.n_trials, n_jobs=-1)

    study_report(study=study, logger=logger)

    # --------------------- Retrain with best hyperparameters -------------------- #

    logger.info(f"Retraining with best hyperparameters...")

    study = optuna.load_study(study_name="saddle_approximator", storage=storage)

    best_params = {
        param: value
        for param, value in study.best_params.items()
        if not param.startswith("num_units_layer")
    }

    model, best_val_loss = trainer(
        train=train_data,
        val=val_data,
        logger=logger,
        verbose=args.verbose,
        patience=args.patience,
        min_delta=args.min_delta,
        restore_best_model=args.restore_best_model,
        loss_fn=args.loss_fn,
        gamma=args.gamma,
        num_workers=args.num_workers,
        input_shape=args.input_shape,
        units_per_layer=[
            study.best_params[f"num_units_layer_{i}"]
            for i in range(study.best_params["num_layers"])
        ],
        **best_params,
    )

    # Predictions
    with torch.no_grad():
        X_train, y_train = train_data.tensors
        y_pred = model(X_train, activation=best_params["activation"])

    # ----------------------------------- Plot ----------------------------------- #

    logger.info("Plotting the results...")

    image = plot_2d_surfaces(
        y_true=y_train.numpy().reshape(args.num_samples, args.num_samples),
        y_pred=y_pred.numpy().reshape(args.num_samples, args.num_samples),
        X=X_train.numpy(),
        name="Monkey Saddle",
    )

    display(image)
