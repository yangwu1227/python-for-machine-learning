import os
import json
import logging
import sys
import argparse
from typing import Dict, Callable, Tuple, List, Union

from sagemaker.predictor import Predictor
import matplotlib.patches as patches
from matplotlib import pyplot as plt
from PIL import Image
from PIL import ImageColor
import numpy as np

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

    # Other
    parser.add_argument("--local_test_mode", type=int, default=0)

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


# ------------------------------ Inference tools ----------------------------- #


class InferenceHandler(object):
    """
    A class for performing inference on an object detection model.
    """

    @classmethod
    def plot_predictions(
        cls,
        image_file_path: str,
        normalized_boxes: List[List[float]],
        class_names: List[Union[str, int]],
        confidences: List[float],
        **kwargs,
    ) -> None:
        """
        Plot the predictions on the image.

        Parameters
        ----------
        image_file_path : str
            The path to the image file.
        normalized_boxes : List[List[float]]
            A list of bounding boxes. Each bounding box is a list of 4 floats, which are the normalized coordinates of the bounding box.
        class_names : List[Union[str, int]]
            A list of class names or IDs.
        confidences : List[float]
            A list of confidences for each bounding box.
        kwargs : Dict[str, type]
            Additional arguments to pass to plt.figure().

        Returns
        -------
        None
        """
        colors = list(ImageColor.colormap.values())
        image_np = np.array(Image.open(image_file_path))

        plt.figure(**kwargs)
        ax = plt.axes()
        ax.imshow(image_np)

        for idx in range(len(normalized_boxes)):
            left, bot, right, top = normalized_boxes[idx]
            x, w = [val * image_np.shape[1] for val in [left, right - left]]
            y, h = [val * image_np.shape[0] for val in [bot, top - bot]]
            color = colors[hash(class_names[idx]) % len(colors)]
            rect = patches.Rectangle(
                (x, y), w, h, linewidth=3, edgecolor=color, facecolor="none"
            )
            ax.add_patch(rect)

            text_label = class_names[idx], confidences[idx] * 100
            ax.text(
                x,
                y,
                f"{class_names[idx]} {confidences[idx] * 100:.0f}%",
                bbox=dict(facecolor="white", alpha=0.5),
            )

        return None

    def __init__(self, model_predictor: Predictor, class_label_map: Dict[str, str]):
        """
        Constructor method.

        Parameters
        ----------
        model_predictor : Predictor
            A Predictor object that is used to perform inference.
        class_label_map : Dict[str, str]
            A dictionary that maps class IDs to class names.

        Returns
        -------
        self
        """
        self.model_predictor = model_predictor
        self.class_label_map = class_label_map

    def _query(self, image_file_path: str, num_boxes: int = 5) -> bytes:
        """
        Query the model for inference. This function reads an image file, converts it to bytes, and then sends it to the model for inference.
        The response of the endpoint is a set of bounding boxes, class names, and scores for the bounding boxes, which are also returned as
        bytes.

        Parameters
        ----------
        image_file_path : str
            The path to the image file.
        num_boxes : int, optional
            The number of bounding boxes to return, by default 5.

        Returns
        -------
        bytes
            The response of the endpoint.
        """
        with open(image_file_path, "rb") as f:
            image_bytes = f.read()

        response = self.model_predictor.predict(
            data=image_bytes,
            # These are arguments for the boto3 invoke_endpoint() method
            initial_args={
                # MIME type of the input data
                "ContentType": "application/x-image",
                # Desired MIME type of the inference in the response
                "Accept": f"application/json;verbose;n_predictions={num_boxes}",
            },
        )

        return response

    def predict(
        self, image_file_path: str, num_boxes: int = 5
    ) -> Tuple[List[str], List[float], List[List[float]]]:
        """
        Parse the response of the endpoint. This function parses the response of the endpoint, which is a set of bounding boxes, class names,
        and scores for the bounding boxes.

        Parameters
        ----------
        image_file_path : str
            The path to the image file.
        num_boxes : int, optional
            The number of bounding boxes to return, by default 5.

        Returns
        -------
        Tuple[List[str], List[float], List[List[float]]]
            A tuple containing the normalized bounding boxes, class names, and scores for the bounding boxes.
        """
        response = self._query(image_file_path, num_boxes)
        # This is a dictionary with the bounding boxes, class names, scores, and tensorflow_model_output
        model_predictions = json.loads(response)

        normalized_boxes, classes, scores, labels = (
            model_predictions["normalized_boxes"],
            model_predictions["classes"],
            model_predictions["scores"],
            model_predictions["labels"],
        )

        # Substitute the class indices with the classes labels
        class_label_int = [labels[int(idx)] for idx in classes]
        # Then, map the integer labels to the class names
        class_label_str = [self.class_label_map[str(idx)] for idx in class_label_int]

        return normalized_boxes, class_label_str, scores
