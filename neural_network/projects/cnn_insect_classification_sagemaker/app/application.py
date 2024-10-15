import os
from typing import List, Union, Tuple

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Nopep8
import tensorflow as tf
import matplotlib

matplotlib.use("Agg")  # Nopep8
import matplotlib.pyplot as plt
from flask import Flask, request, render_template, send_from_directory
import numpy as np

from PIL import Image
from werkzeug.datastructures import FileStorage

# ---------------------------------- Set up ---------------------------------- #

application = Flask(__name__)

# --------------------------------- Functions -------------------------------- #


def load_class_names() -> List[str]:
    """
    Load the class names from the classes.txt file into a list.

    Returns
    -------
    List[str]
        List of class names.
    """
    with open(os.path.join(os.path.dirname(__file__), "models/classes.txt"), "r") as f:
        classes = [line.strip().split(maxsplit=1)[1] for line in f]
    return classes


def select_model(model_name: str) -> tf.keras.models.Model:
    """
    Select the model based on the model name.

    Parameters
    ----------
    model_name : str
        Name of the model.

    Returns
    -------
    tf.keras.models.Model
        The trained model.
    """
    model = baseline_model if model_name == "baseline" else fine_tune_model
    return model


def process_image(image_file: FileStorage) -> np.ndarray:
    """
    Process the image file for prediction--- resize and add batch dimension.

    Parameters
    ----------
    image_file : FileStorage
        The image file to process.

    Returns
    -------
    np.ndarray
        The processed image.
    """
    image = Image.open(image_file.stream).convert("RGB")

    # Resize the image to match the model's input size (256 x 256)
    input_size = (256, 256)
    image = image.resize(input_size)

    # Add batch dimension
    image_np = np.expand_dims(np.array(image), axis=0).astype(np.int16)

    return image_np


def predict_class(model: tf.keras.models.Model, image_np: np.ndarray) -> str:
    """
    Use the model to predict the class of the image.

    Parameters
    ----------
    model : tf.keras.models.Model
        The trained model.
    image_np : np.ndarray
        The processed image.

    Returns
    -------
    str
        The name of the predicted class.
    """
    # Prediction
    predicted_label = tf.nn.softmax(model.predict(image_np)).numpy().argmax()
    class_name = classes[predicted_label]

    return class_name


def plot_image(image_np: np.ndarray, class_name: str) -> str:
    """
    Plot the image and save it to the static folder for display on the web page.

    Parameters
    ----------
    image_np : np.ndarray
        The processed image.
    class_name : str
        The name of the predicted class.

    Returns
    -------
    str
        The name of the image plot.
    """
    # Plot image
    plt.imshow(image_np[0])
    plt.axis("off")
    image_plot = "image_plot.png"
    plt.savefig(os.path.join(os.path.dirname(__file__), "static", image_plot))
    plt.clf()

    return image_plot


# ------------------------ Load model and class names ------------------------ #

classes = load_class_names()
baseline_model = tf.keras.models.load_model(
    os.path.join(os.path.dirname(__file__), "models/baseline")
)
fine_tune_model = tf.keras.models.load_model(
    os.path.join(os.path.dirname(__file__), "models/fine_tune")
)

# -------------------------------- Application ------------------------------- #


@application.route("/", methods=["GET", "POST"])
def upload_and_predict():
    if request.method == "POST":
        model = select_model(request.form["model"])
        image_file = request.files["image"]

        if image_file:
            image_np = process_image(image_file)
            class_name = predict_class(model, image_np)
            image_plot = plot_image(image_np, class_name)

            return render_template(
                "index.html", predicted_class=class_name, image_plot=image_plot
            )

    return render_template("index.html")


if __name__ == "__main__":
    # This is only for running locally since AWS Elastic Beanstalk imports application.py as a module and runs it on its own server
    application.run()
