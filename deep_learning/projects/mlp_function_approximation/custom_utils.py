import logging
import os
import sys
from typing import Any, Dict, Tuple, Union

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Nopep8
import IPython
import keras_tuner as kt
import matplotlib.pyplot as plt
import numpy as np
import optuna
import plotly.graph_objects as go
from IPython.display import Image
from optuna.trial import TrialState
from plotly.subplots import make_subplots
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import PythonLexer

# ------------------------------ Logger function ----------------------------- #


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


# --------------------- Pretty print code in HTML format --------------------- #


def pretty_print_code(filename: str) -> Union[IPython.core.display.HTML, None]:
    """
    Function to pretty print Python code from a file.

    Parameters
    ----------
    filename : str
        The path to the Python file to be pretty printed.

    Returns
    -------
    IPython.core.display.HTML or None
        The HTML object containing the pretty printed code, or None if the file could not be read.
    """
    try:
        with open(filename, "r") as file:
            code = file.read()
    except OSError:
        return None

    formatter = HtmlFormatter(style="default")
    result = highlight(code, PythonLexer(), formatter)
    return IPython.display.HTML(
        '<style type="text/css">{}</style>{}'.format(
            formatter.get_style_defs(".highlight"), result
        )
    )  # type: ignore[no-untyped-call]


# ---------- Function to report best hyperparameters and best score ---------- #


def report_keras_hpo(
    tuner: kt.Tuner, X: np.ndarray, y: np.ndarray, logger: logging.Logger
) -> Tuple[Dict[str, Any], float]:
    """
    Function to report the best hyperparameters and the best score evaluated on the
    data set.

    Parameters
    ----------
    tuner : kt.Tuner
        The tuner object.
    X : np.ndarray
        The input data.
    y : np.ndarray
        The target data.
    logger : logging.Logger
        The logger object.

    Returns
    -------
    Tuple[Dict[str, Any], float]
        A tuple of the best hyperparameters and the best score.
    """
    best_hp = tuner.get_best_hyperparameters()[0].values
    best_model = tuner.get_best_models()[0]
    loss, metric = best_model.evaluate(X, y, verbose=0)

    logger.info(f"Best hyperparameters: {best_hp}")
    logger.info(f"Best metric score: {metric}")

    return best_hp, metric


# -------------- Function to plot function and its approximation ------------- #


def plot_1d_curve(
    y_true: np.ndarray, y_pred: np.ndarray, X: np.ndarray, name: str
) -> None:
    """
    Function to plot the true function and its approximation.

    Parameters
    ----------
    y_true : np.ndarray
        The true function.
    y_pred : np.ndarray
        The approximation of the true function.
    X : np.ndarray
        The input data.
    name : str
        The name of the function.

    Returns
    -------
    None
    """
    plt.plot(X, y_true, label=name)
    plt.plot(X, y_pred, label="Predictions")
    plt.legend()
    plt.show()


# ---------- Function to plot the hypersurface and its approximation --------- #


def plot_2d_surfaces(
    y_true: np.ndarray, y_pred: np.ndarray, X: np.ndarray, name: str, **kwargs
) -> Image:
    """
    Plot the original and predicted functions side by side.

    Parameters
    ----------
    y_true : np.ndarray
        Z-coordinates of the original function.
    y_pred : np.ndarray
        Z-coordinates of the predicted function.
    X : np.ndarray
        Matrix with columns x1 and x2.
    name : str
        Name of the function.
    **kwargs
        Additional keyword arguments to be passed to the update_layout function.


    Returns
    -------
    IPython.display.Image
        Static image of the plot.
    """
    # Extract x1 and x2 from X
    x1 = X[:, 0].reshape(y_true.shape)
    x2 = X[:, 1].reshape(y_true.shape)

    # Create a subplot with 1 row and 2 columns
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(f"{name} Original", f"{name} Predictions"),
        specs=[[{"type": "surface"}, {"type": "surface"}]],
    )

    fig.add_trace(
        go.Surface(x=x1, y=x2, z=y_true, name="Original", showscale=False), row=1, col=1
    )
    fig.add_trace(
        go.Surface(x=x1, y=x2, z=y_pred, name="Predictions", showscale=False),
        row=1,
        col=2,
    )

    # Update layout with additional arguments
    fig.update_layout(width=800, height=600, **kwargs)

    fig_bytes = fig.to_image(format="png")

    return Image(fig_bytes)  # type: ignore[no-untyped-call]


# ----------------------- Function to plot one surface ----------------------- #


def plot_2d_surface(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> Image:
    """
    Plot a 3D surface using Plotly.

    Parameters
    ----------
    x : np.ndarray
        X-coordinates.
    y : np.ndarray
        Y-coordinates.
    z : np.ndarray
        Z-coordinates (function values).

    Returns
    -------
    IPython.display.Image
        Static image of the plot.
    """
    fig = go.Figure(data=[go.Surface(z=z, x=x, y=y)])
    fig.update_layout(
        title="3D Surface Plot",
        autosize=False,
        width=500,
        height=500,
        margin=dict(l=65, r=50, b=65, t=90),
    )
    fig_bytes = fig.to_image(format="png")

    return Image(fig_bytes)  # type: ignore[no-untyped-call]


# -------------------------- Optuna report function -------------------------- #


def study_report(study: optuna.study.Study, logger: logging.Logger) -> None:
    """
    Report study results.

    Parameters
    ----------
    study : optuna.study.Study
        Optuna study instance.
    logger : logging.Logger
        The logger object.
    """
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    best_trial = study.best_trial

    logger.info(f"Number of pruned trials: {len(pruned_trials)}")
    logger.info(f"Number of complete trials: {len(complete_trials)}")
    logger.info(f"Best trial score: {best_trial.value}")
    logger.info(f"Best trial params: {best_trial.params}")

    return None


# ------------------------ Function for creating study ----------------------- #


def create_study(
    study_name: str, storage: str, direction: str = "maximize"
) -> optuna.study.Study:
    """
    Create Optuna study instance.

    Parameters
    ----------
    study_name : str
        Name of the study.
    storage : str
        Database url.
    direction: str
        Direction of the metric--- maximize or minimize.

    Returns
    -------
    optuna.study.Study
        Optuna study instance.
    """
    study = optuna.create_study(
        storage=storage,
        sampler=optuna.samplers.TPESampler(),
        study_name=study_name,
        direction=direction,
        load_if_exists=True,
    )

    return study
