import logging
import sys
from typing import Any, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.nn import Module

# ---------------------------------- Logger ---------------------------------- #


def get_logger(name: str) -> logging.Logger:
    """
    Parameters
    ----------
    name : str
        A string that specifies the name of the logger.

    Returns
    -------
    logging.Logger
        A logger with the specified name.
    """
    logger = logging.getLogger(name)  # Return a logger with the specified name

    log_format = "%(asctime)s %(levelname)s %(name)s: %(message)s"
    formatter = logging.Formatter(log_format)
    # No matter how many processes we spawn, we only want one StreamHandler attached to the logger
    if not any(
        isinstance(handler, logging.StreamHandler) for handler in logger.handlers
    ):
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.setLevel(logging.INFO)

    return logger


# ---------------------------------- Classes --------------------------------- #


class LSTM(nn.Module):
    """
    This class implements the LSTM model with stacked layers. The hyperparameters are the
    number of layers, the input size, and the hidden size. Then, the output of the hidden
    states is passed to a fully connected layer to get the final regression (continuous)
    value.
    """

    def __init__(self, input_size: int = 1, hidden_size: int = 50, num_layers: int = 2):
        """
        Constructor for the LSTM class.
        """
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=False,
        )
        self.linear = nn.Linear(in_features=hidden_size, out_features=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the LSTM model.

        Parameters
        ----------
        x : torch.Tensor
            A 3D tensor of shape (sequence_length, batch_size, num_features).

        Returns
        -------
        torch.Tensor
            The single value prediction.
        """
        # The LSTM returns tuples of output, (h_n, c_n), i.e., final hidden state and final cell state
        out, _ = self.lstm(x)
        out = self.linear(out)
        return out


class GRU(nn.Module):
    """
    This class implements the GRU model with stacked layers. The hyperparameters are, again,
    the number of layers, the input size, and the hidden size.
    """

    def __init__(self, input_size: int = 1, hidden_size: int = 50, num_layers: int = 2):
        """
        Constructor for the GRU class.
        """
        super(GRU, self).__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=False,
        )
        self.linear = nn.Linear(in_features=hidden_size, out_features=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the GRU model.

        Parameters
        ----------
        x : torch.Tensor
            A 3D tensor of shape (sequence_length, batch_size, num_features).

        Returns
        -------
        torch.Tensor
            The single value prediction.
        """
        out, _ = self.gru(x)
        out = self.linear(out)
        return out


# ----------------------------- Count parameters ----------------------------- #


def count_parameters(model: nn.Module) -> int:
    """
    Count the number of trainable parameters in a model.

    Parameters
    ----------
    model : nn.Module
        The model.

    Returns
    -------
    int
        The number of trainable parameters in the model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# --------------- Function for reshaping a 2D grid to sequence --------------- #


def grid_to_seq(grid_2d: np.ndarray) -> torch.Tensor:
    """
    Format a 2D grid of values for use with sequence models in PyTorch.

    In Pytorch, LSTM and GRU expects all of its inputs to be 3D tensors. The first axis is the sequence itself,
    the second indexes instances in the mini-batch, and the third indexes number of features of in each input
    vector at each time step.

    We use a mini-batch size of 1. Each time step in our sequence only contains one feature (a single value from the 2D grid)
    so the third axis is also 1. The length of the sequence is the number of samples squared.

    We have effectively flattened the 2D grid into a sequence with batch and feature dimensions in order to feed the data
    into the LSTM or GRU. This is in contrast to approximating functions with MLP, in which case we first convert the data
    into X (matrix) and y (vector) before feeding it into the models. Essentially, we represent the data in different ways
    in order to feed it into different types of models.

    Finally, we can reshape the predictions back into a 2D grid for plotting after the LSTM or GRU has made its predictions.

    Parameters
    ----------
    grid_2d : np.ndarray
        A 2D grid of values with float32 data type.

    Returns
    -------
    torch.Tensor
        A 3D tensor of shape (sequence_length, batch_size, num_features).
    """
    tensor_3d = torch.from_numpy(np.copy(grid_2d))
    # 3D tensor with seq_len = samples * samples, batch = 1, and input_size = 1
    tensor_3d = tensor_3d.view(-1, 1, 1)

    return tensor_3d


# ----------------------------- Trainer function ----------------------------- #


def trainer(
    data: torch.Tensor,
    model_type: str,
    hyperparameters: Dict[str, Any],
    epochs: int,
    learning_rate: float,
    logger: logging.Logger,
) -> np.ndarray:
    """
    Train an LSTM or GRU model on the provided data.

    Parameters
    ----------
    data : torch.Tensor
        The input data, a 3D tensor of shape (sequence_length, batch_size, num_features).
    model_type : str
        The type of model to train. Either 'lstm' or 'gru'.
    hyperparameters : Dict[str, Any]
        A dictionary containing the hyperparameters for the model.
    epochs : int, optional
        The number of epochs to train for.
    learning_rate : float, optional
        The learning rate for the Adam optimizer.
    logger : logging.Logger
        A logger object for logging the training progress.

    Returns
    -------
    np.ndarray
        The predictions of the trained model, reshaped to match the original shape of the data.
    """
    model: Module
    # Initialize the model
    if model_type.lower() == "lstm":
        model = LSTM(**hyperparameters)
    elif model_type.lower() == "gru":
        model = GRU(**hyperparameters)
    else:
        raise ValueError("Invalid model_type; expected 'lstm' or 'gru'")

    # Define the loss function and the optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    for epoch in range(epochs):
        # The forward pass returns (seq_len, batch = 1, input_size = 1)
        # Squeeze removes all dimensions of size 1 from the tensor, so outputs has shape (seq_len,)
        outputs = model(data).squeeze()
        # The data has shape (seq_len, batch = 1, input_size = 1), so we need to squeeze it to (seq_len,) as well
        loss = criterion(outputs, data.squeeze())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(
                f"{model_type.upper()} | Epoch [{epoch  + 1}/{epochs}], Loss: {loss.item():.4f}"
            )

    # Flatten the predictions to a 1D array
    predictions = outputs.view(-1).detach().numpy()
    # Reshape to the original shape of the data
    original_shape = int(np.sqrt(predictions.shape[0]))
    predictions = predictions.reshape((original_shape, original_shape))

    return predictions


# ----------------------------- Plot predictions ----------------------------- #


def plot_predictions(
    original_data: np.ndarray,
    lstm_prediction: np.ndarray,
    gru_prediction: np.ndarray,
    figsize: Tuple[int, int] = (35, 25),
) -> None:
    """
    Plot the original data and the model's prediction in 3D.

    Parameters
    ----------
    original_data : np.ndarray
        The original data, a 2D array.
    lstm_prediction : np.ndarray
        The LSTM model's prediction, a 2D array.
    gru_prediction : np.ndarray
        The GRU model's prediction, a 2D array.
    figsize : Tuple[int, int], optional
        The size of the figure, by default (35, 25).
    """
    # Generate the x and y coordinates
    x = np.arange(original_data.shape[0])
    y = np.arange(original_data.shape[1])
    xv, yv = np.meshgrid(x, y)

    # Create a new figure for the plots
    fig = plt.figure(figsize=figsize)

    # Plot the original data
    ax1 = fig.add_subplot(131, projection="3d")
    ax1.plot_surface(xv, yv, original_data, cmap="viridis")  # type: ignore[attr-defined]
    ax1.set_title("Original Data")

    # Plot the LSTM prediction
    ax2 = fig.add_subplot(132, projection="3d")
    ax2.plot_surface(xv, yv, lstm_prediction, cmap="viridis")  # type: ignore[attr-defined]
    ax2.set_title("LSTM Prediction")

    # Plot the GRU prediction
    ax3 = fig.add_subplot(133, projection="3d")
    ax3.plot_surface(xv, yv, gru_prediction, cmap="viridis")  # type: ignore[attr-defined]
    ax3.set_title("GRU Prediction")

    # Show the plots
    plt.show()
