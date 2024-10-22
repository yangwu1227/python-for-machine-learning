import argparse
import json
import logging
import os
import sys
from typing import Callable, Dict, List, Tuple

from IPython.display import Image

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Nopep8
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import tensorflow as tf
from imblearn.metrics import classification_report_imbalanced
from tensorflow.keras import backend as K
from tensorflow.keras import layers

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


# ------------------- Parse arguments from the command line ------------------ #


def parser() -> argparse.ArgumentParser:
    """
    Parse arguments from the command line.

    Returns
    -------
    argparse.ArgumentParser
        The parser object.
    """
    parser = argparse.ArgumentParser()

    # Data, model, and output directories
    parser.add_argument("--model_dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--train", type=str, default=os.environ["SM_CHANNEL_TRAIN"])
    parser.add_argument("--val", type=str, default=os.environ["SM_CHANNEL_VAL"])
    parser.add_argument(
        "--training_env", type=str, default=json.loads(os.environ["SM_TRAINING_ENV"])
    )

    # Other
    parser.add_argument("--test_mode", type=int, default=0)

    return parser


# --------- Decorator for adding additional arguments to base parser --------- #


def add_additional_args(parser_func: Callable, additional_args: Dict[str, type]):
    """
    Add additional arguments to the parser function, which returns the parser, that are specific to each script.
    This function decorator returns a callable parser function that can be called to parse additional arguments,
    and finaly returning the namespace object containging those arguments.

    Parameters
    ----------
    parser_func : Callable
        The base parser function.
    additional_args : Dict[str, type]
        A dictionary with the additional arguments to add to the parser function. Each key is the name of the argument and the value is the type of the argument, e.g. {'arg1': str, 'arg2': int}.

    Returns
    -------
    Callable
        The parser function with additional arguments.
    """

    def wrapper():
        # Call the original parser function to get the parser object
        parser = parser_func()

        for arg_name, arg_type in additional_args.items():
            parser.add_argument(f"--{arg_name}", type=arg_type)

        args, _ = parser.parse_known_args()

        return args

    return wrapper


# ------------------------- Function for loading data ------------------------ #


def load_dataset(dir: str, batch_size: int) -> tf.data.Dataset:
    """
    Read in the dataset from the specified directory and return a tf.data.Dataset object.

    Parameters
    ----------
    dir : str
        The directory where the dataset is located, which can conveinently be S3 with tensorflow-io.
    batch_size : int
        The batch size.

    Returns
    -------
    tf.data.Dataset
        The dataset with the specified batch size.
    """
    # Load data as tensorflow dataset
    dataset = tf.data.Dataset.load(dir).batch(batch_size)

    return dataset


# --------------------------- Precision and recall --------------------------- #


def recall(y_true: tf.Tensor, y_pred: tf.Tensor) -> float:
    """
    Function for computing the recall score. Only computes a batch-wise average of precision.
    This is taken from the Keras github repository, which is removed in the latest versions.

    Parameters
    ----------
    y_true : tf.Tensor
        The true labels.
    y_pred : tf.Tensor
        The predicted values.

    Returns
    -------
    float
        The recall score.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))

    # K.epsilon() is a small number to avoid division by zero
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision(y_true: tf.Tensor, y_pred: tf.Tensor) -> float:
    """
    Function for computing the precision score. Only computes a batch-wise average of precision

    Parameters
    ----------
    y_true : tf.Tensor
        The true labels.
    y_pred : tf.Tensor
        The predicted values.

    Returns
    -------
    float
        The precision score.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))

    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


# --------------------------- Classification report -------------------------- #


def classification_report(
    y_true: np.ndarray, y_pred: np.ndarray, labels: List[str]
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Function for computing the classification report.

    Parameters
    ----------
    y_true : np.ndarray
        The true labels.
    y_pred : np.ndarray
        The predicted labels.
    labels : List[str]
        The list of labels for the classes, ordered alphabetically.

    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, float]]
        A tuple of the classification report and the class-wise accuracy.

    Raises
    ------
    ValueError
        If the number of labels is not equal to the number of classes.
    """
    if labels != sorted(labels):
        raise ValueError("Labels must be ordered alphabetically")

    clf_report_dict = classification_report_imbalanced(y_true, y_pred, output_dict=True)

    agg_metrics_keys = [
        "avg_pre",
        "avg_rec",
        "avg_spe",
        "avg_f1",
        "avg_geo",
        "avg_iba",
        "total_support",
    ]

    agg_metrics = {k: clf_report_dict[k].round(4) for k in agg_metrics_keys}

    # Obtain the class indices
    class_indices = list(set(clf_report_dict.keys()) - set(agg_metrics_keys))
    clf_report_frame = (
        pd.DataFrame(clf_report_dict).transpose().loc[class_indices].round(4)
    )

    # Add the class names
    clf_report_frame.index = labels

    # Change column names to be more readable
    clf_report_frame.columns = [
        "precision",
        "recall",
        "specificity",
        "f1",
        "geometric_mean",
        "index_balanced_accuracy",
        "support",
    ]

    return clf_report_frame, agg_metrics


# ------------------- Parametrized data augmentation layer ------------------- #


class AugmentationModel(object):
    """
    A class for creating a parametrized data augmentation layers, which is essentially a sequetial model. This class can be extended to include more data augmentation layers, which the user can specify using 'layer_name' and **kwargs pairs.
    """

    def __init__(self, aug_params):
        """
        Instantiate the augmentation model. The augmentation parameters are passed as a dictionary.
        The format of the dictionary is must be 'layer_name': {'param1': value1, 'param2': value2, ...}.
        For example, for random flip, the dictionary is {'RandomFlip': {'mode': 'horizontal'}}

        Parameters
        ----------
        aug_params : Dict[str, Dict[str, Any]]
            The augmentation parameters.
        """
        # Base model is a sequential model
        self.base_model = tf.keras.Sequential()
        # Augmentation layer: parameters are passed as a dictionary
        self.aug_params = aug_params

    @property
    def aug_params(self):
        return self._aug_params

    # Validate aug_params input
    @aug_params.setter
    def aug_params(self, aug_params):
        if not isinstance(aug_params, dict):
            raise TypeError(
                "The augmentation parameters must be supplied as a dictionary of str -> dict"
            )
        self._aug_params = aug_params

    def _add_augmentation_layer(self, layer_name, **kwargs):
        """
        Private method for adding a single augmentation layer to the model.

        Parameters
        ----------
        layer_name : str
            The name of the augmentation layer.
        **kwargs : Dict[str, Any]
            The parameters for the augmentation layer as a dictionary. The keys are the parameter names
            and the values are the parameter values.
        """
        # Intantiate a layer from a config dictionary
        layer = tf.keras.layers.deserialize(
            config={"class_name": layer_name, "config": kwargs}
        )
        # Add the layer to the base model
        self.base_model.add(layer)

    def build_augmented_model(self):
        """
        Build the augmented model with the specified data augmentation layers.

        Returns
        -------
        tf.keras.Model
            The augmented model.
        """
        for layer_name, args in self.aug_params.items():
            if not isinstance(args, dict):
                raise ValueError(
                    f"Augmentation layer arguments should be provided as a dictionary for layer: {layer_name}"
                )
            self._add_augmentation_layer(layer_name, **args)

        model = tf.keras.Sequential([self.base_model], name="data_augmentation_layers")

        return model


# --------------------- Hyperparameter tuning visualizer --------------------- #


class TuningVisualizer(object):
    """
    This class implements visualizations to support interpreting
    hyperparameter optimization results.
    """

    def __init__(
        self, tune_data: pd.DataFrame, num_params: List[str], cat_params: List[str]
    ):
        """
        Instantiate the tuning visualizer.

        Parameters
        ----------
        tune_data : pd.DataFrame
            The hyperparameter tuning data.
        num_params : List[str]
            The list of numerical parameters.
        cat_params : List[str]
            The list of categorical parameters.
        """
        self.tune_data = tune_data
        self.num_params = num_params
        self.cat_params = cat_params

    @property
    def tune_data(self):
        return self._tune_data

    @tune_data.setter
    def tune_data(self, tune_data):
        if not isinstance(tune_data, pd.DataFrame):
            raise TypeError("The tuning data must be supplied as a pandas DataFrame")
        self._tune_data = tune_data

    @property
    def num_params(self):
        return self._num_params

    @num_params.setter
    def num_params(self, num_params):
        if not isinstance(num_params, list) or not all(
            isinstance(param, str) for param in num_params
        ):
            raise TypeError(
                "The numerical parameters must be supplied as a list of strings"
            )
        self._num_params = num_params

    @property
    def cat_params(self):
        return self._cat_params

    @cat_params.setter
    def cat_params(self, cat_params):
        if not isinstance(cat_params, list) or not all(
            isinstance(param, str) for param in cat_params
        ):
            raise TypeError(
                "The categorical parameters must be supplied as a list of strings"
            )
        self._cat_params = cat_params

    def _cat_mapping(self, cat_params: List[str]) -> List[Dict[str, int]]:
        """
        Create mappings for categorical parameters from str -> int.

        Parameters
        ----------
        cat_params : List[str]
            The list of categorical parameters to generate mappings for.

        Returns
        -------
        List[Dict[str, int]]
            A list of mappings for each categorical parameter.
        """
        cat_mappings = []
        for param in cat_params:
            if self.tune_data[param].dtype != "object":
                raise TypeError(f"The parameter {param} is not categorical")
            cat_mappings.append(
                {
                    categorical_key: integer_encoding
                    for integer_encoding, categorical_key in enumerate(
                        self.tune_data[param].unique()
                    )
                }
            )

        return cat_mappings

    def _static_plot(self, fig: go.Figure) -> Image:
        """
        Create a static plot.

        Parameters
        ----------
        fig : go.Figure
            The plotly figure.

        Returns
        -------
        Image
            The static plot.
        """
        fig_bytes = fig.to_image(format="png")

        return Image(fig_bytes)

    def plot_parallel_coordinate(
        self,
        columns: List[str],
        target: str = "FinalObjectiveValue",
        static: bool = False,
        figsize: Tuple[float, float] = (1000, 800),
        **kwargs,
    ) -> go.Figure:
        """
        Function for plotting the parallel coordinate plot given the
        list of hyperparameters.

        Parameters
        ----------
        columns : List[str]
            The list of hyperparameters to plot.
        target : str, optional
            The target column to use to color the marks, by default 'FinalObjectiveValue'.
        static : bool, optional
            Whether to show the plot in static mode, by default False.
        figsize : Tuple[float, float], optional
            The figure size, by default (1000, 800).
        **kwargs: Dict[str, Any]
            Additional arguments to pass to the plotly Parcoords plot.

        Returns
        -------
        go.Figure
            The parallel coordinates plot.
        """
        # Create a copy of the data
        tune_data = self.tune_data.copy()

        cat_params = [col for col in columns if col in self.cat_params]
        num_params = [col for col in columns if col in self.num_params]

        cat_dicts = []
        if len(cat_params) > 0:
            cat_mappings = self._cat_mapping(cat_params=cat_params)
            for i, col in enumerate(cat_params):
                # Encode categorical to integer
                tune_data[col] = tune_data[col].map(cat_mappings[i])
                # List of dictionaries for categorical hyperparameters
                cat_dicts.append(
                    dict(
                        label=col,
                        values=tune_data[col],
                        tickvals=list(cat_mappings[i].values()),
                        ticktext=list(cat_mappings[i].keys()),
                    )
                )

        num_dicts = []
        if len(num_params) > 0:
            num_dicts = [dict(label=col, values=tune_data[col]) for col in num_params]

        # Create the parallel coordinates plot
        fig = go.Figure(
            go.Parcoords(
                line=dict(
                    color=tune_data[target],
                    colorscale=(
                        kwargs["colorscale"]
                        if "colorscale" in kwargs.keys()
                        else "viridis"
                    ),
                    showscale=True,
                ),
                dimensions=cat_dicts + num_dicts,
            )
        )

        fig.update_layout(width=figsize[0], height=figsize[1])

        if static:
            return self._static_plot(fig)

        return fig


# ------------------------- Class for error analysis ------------------------- #


class ErrorAnalyzer(object):
    """
    This class implements methods for analyzing model errors.
    """

    def __init__(
        self,
        y_true: np.ndarray,
        images: np.ndarray,
        y_pred: np.ndarray,
        label_mapping: Dict[str, int],
    ):
        """
        Instantiate the error analyzer.

        Parameters
        ----------
        y_true : np.ndarray
            The true labels with shape (num_samples,).
        images : np.ndarray
            The images with shape (num_samples, height, width, channels).
        y_pred : np.ndarray
            The predicted probabilies matrix with shape (num_samples, num_classes).
        label_mapping : Dict[str, int]
            The label mapping.
        """
        self.y_true = y_true
        self.images = images
        self.y_pred = y_pred
        # Computed upon instantiation
        self.class_label = list(label_mapping.keys())
        self.class_encode = list(label_mapping.values())
        self.mis_clf_indices = np.where(y_true != np.argmax(self.y_pred, axis=1))[0]
        # Misclassified and correctly classified indices for each class
        self.clf_by_class = self._clf_by_class()

    def _clf_by_class(self) -> Dict[str, np.ndarray]:
        """
        Report the misclassified and correctly classified indices by class.

        Returns
        -------
        Dict[str, np.ndarray]
            A dictionary mapping each class to its misclassified and correctly classified indices.
        """
        clf_by_class = {}
        for encode, label in zip(self.class_encode, self.class_label):
            # Indices of correctly classified samples
            correct_indices = np.where(
                (self.y_true == encode)
                & (self.y_true == np.argmax(self.y_pred, axis=1))
            )[0]
            # Indices of misclassified samples
            mis_clf_indices = np.where(
                (self.y_true == encode)
                & (self.y_true != np.argmax(self.y_pred, axis=1))
            )[0]
            clf_by_class[label] = {
                "correctly_classified": correct_indices,
                "misclassified": mis_clf_indices,
            }

        return clf_by_class

    def plot_mis_clf(
        self,
        class_label: str,
        sample_mis_clf: int = 10,
        sample_correct_clf: int = 5,
        figsize: Tuple[int, int] = (12, 10),
        **kwargs,
    ) -> None:
        """
        Plot the misclassified images for a given class. Note that this method first samples
        10 images from the misclassified images for the given class. To adequately conduct
        error analysis, it is recommended to call this method multiple times for each class.

        Parameters
        ----------
        class_label : str
            The class label.
        sample_mis_clf : int, optional
            The number of misclassified images to sample, by default 10. Must be less than or equal to 10.
        sample_correct_clf : int, optional
            The number of correctly classified images to sample, by default 5. Must be less than or equal to 5.
        figsize : Tuple[int, int], optional
            The figure size, by default (12, 10).
        **kwargs: Dict[str, Any]
            Additional arguments to pass to the plt.imshow() function.

        Returns
        -------
        plt.Figure
            The matplotlib figure.
        """
        if sample_mis_clf > 10 or sample_correct_clf > 5:
            raise ValueError(
                "The arguments `sample_mis_clf` must be less than or equal to 10 and `sample_correct_clf` must be less than or equal to 5."
            )

        # Get the correctly classified and misclassified indices for the class
        mis_clf_indices = self.clf_by_class[class_label]["misclassified"]
        correct_clf_indices = self.clf_by_class[class_label]["correctly_classified"]

        # If the number of misclassified images is less than sample_mis_clf, then sample all misclassified images (including 0)
        if len(mis_clf_indices) < sample_mis_clf:
            sample_mis_clf = len(mis_clf_indices)
        # If number of correctly classified images is less than sample_correct_clf, then sample all correctly classified images (including 0)
        if len(correct_clf_indices) < sample_correct_clf:
            sample_correct_clf = len(correct_clf_indices)

        mis_clf_indices = np.random.choice(
            mis_clf_indices, size=sample_mis_clf, replace=False
        )
        correct_indices = np.random.choice(
            correct_clf_indices, size=sample_correct_clf, replace=False
        )

        fig, axs = plt.subplots(3, 5, figsize=figsize)

        # Iterate over the sample indices and plot the images
        for i, idx in enumerate(mis_clf_indices):
            ax = axs[i // 5, i % 5]
            ax.imshow(self.images[idx].astype(np.int16), **kwargs)
            true_label = self.class_label[self.class_encode.index(self.y_true[idx])]
            pred_label = self.class_label[
                self.class_encode.index(np.argmax(self.y_pred[idx]))
            ]
            ax.set_title(f"True: {true_label}\nPred: {pred_label}")
            ax.axis("off")

        for i, idx in enumerate(correct_indices):
            ax = axs[2, i]
            ax.imshow(self.images[idx].astype(np.int16), **kwargs)
            ax.set_title(f"Correctly classified: {class_label}")
            ax.title.set_size(10)
            ax.axis("off")

        fig.tight_layout()

        return plt
