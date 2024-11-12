import logging
import os
from functools import partial
from typing import Any, Dict, List, Tuple

import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Nopep8
import tensorflow as tf
from custom_utils import get_logger, load_data, parser

# ------------------------- Function for fine tuning ------------------------- #


def fine_tune_cnn(
    logger: logging.Logger,
    train_data: List[np.ndarray],
    val_data: List[np.ndarray],
    dense_params: Dict[str, Any],
    aug_params: Dict[str, Any],
    opt_params: Dict[str, Any],
    fit_params: Dict[str, Any],
    dropout_rate: float = 0.5,
    input_shape: Tuple[int] = (256, 256, 3),
    verbose: int = 2,
) -> Tuple[tf.keras.models.Sequential, tf.keras.callbacks.EarlyStopping]:
    """
    Function to fine tune the last three convolutional layers of a pretrained
    VGG16 model.

    Parameters
    ----------
    logger: logging.Logger
        Logger to log information to CloudWatch.
    train_data : List[np.ndarray]
        Training data.
    val_data : List[np.ndarray]
        Validation data.
    dense_params : Dict[str, Any]
        Hyperparameters for dense layers.
    aug_params : Dict[str, Any]
        Hyperparameters for data augmentation.
    opt_params : Dict[str, Any]
        Hyperparameters for optimizer.
    fit_params: Dict[str, Any]
        Hyperparameters for training.
    dropout_rate : float, optional
        Dropout rate, by default 0.5.
    input_shape : Tuple[int], optional
        Dimension of the input feature vector, by default (256, 256, 3).
    verbose : int, optional
        Verbosity mode, by default 2.

    Returns
    -------
    Tuple[tf.keras.models.Sequential, tf.keras.callbacks.EarlyStopping]
        Fine tuned model and early stopping callback.
    """
    conv_base = tf.keras.applications.vgg16.VGG16(
        include_top=False, weights="imagenet", input_shape=input_shape
    )

    # ------------------------ Train classifier at the top ----------------------- #

    logger.info("Begin training classifer on top of frozen base...")

    # Freeze all layers for the base
    conv_base.trainable = False

    # Default dense layer
    DefaultDense = partial(
        tf.keras.layers.Dense,
        activation="relu",
        kernel_initializer="he_normal",
        kernel_regularizer=tf.keras.regularizers.l2(dense_params["dense_weight_decay"]),
    )
    # Data augmentation layers
    data_augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomContrast(factor=aug_params["random_contrast_factor"]),
            tf.keras.layers.RandomFlip(mode=aug_params["random_flip_mode"]),
            tf.keras.layers.RandomRotation(factor=aug_params["random_rotation_factor"]),
            tf.keras.layers.RandomZoom(height_factor=aug_params["random_zoom_factor"]),
        ]
    )

    inputs = tf.keras.Input(shape=input_shape)
    x = data_augmentation(inputs)
    x = tf.keras.applications.vgg16.preprocess_input(x)
    x = conv_base(x, training=False)
    x = tf.keras.layers.Flatten()(x)
    x = DefaultDense(units=dense_params["dense_units"])(x)
    x = tf.keras.layers.Dropout(rate=dropout_rate)(x)
    outputs = tf.keras.layers.Dense(units=4, activation="softmax")(x)
    cnn_model = tf.keras.Model(inputs, outputs)
    cnn_model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=opt_params["learning_rate"], clipnorm=opt_params["clipnorm"]
        ),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )

    early_stopper_clf = tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy", patience=3, restore_best_weights=True
    )

    cnn_model.fit(
        x=train_data[0],
        y=train_data[1],
        batch_size=fit_params["batch_size"],
        epochs=fit_params["epochs"],
        validation_data=(val_data[0], val_data[1]),
        callbacks=[early_stopper_clf],
        verbose=verbose,
    )

    logger.info(
        f"Best validation accuracy after training classifier: {early_stopper_clf.best}"
    )

    # ----------------- Fine-tune last three convolutional layers ---------------- #

    logger.info("Begin fine-tuning...")

    # Unfreeze the last three convolutional layers
    conv_base.trainable = True
    for layer in conv_base.layers[:-4]:
        layer.trainable = False

    # Re-compile the model
    cnn_model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=opt_params["learning_rate"], clipnorm=opt_params["clipnorm"]
        ),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )

    early_stopper_fine_tune = tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy", patience=3, restore_best_weights=True
    )

    cnn_model.fit(
        x=train_data[0],
        y=train_data[1],
        batch_size=fit_params["batch_size"],
        epochs=fit_params["epochs"],
        validation_data=(val_data[0], val_data[1]),
        callbacks=[early_stopper_fine_tune],
        verbose=verbose,
    )

    return cnn_model, early_stopper_fine_tune


if __name__ == "__main__":
    args = parser()

    logger = get_logger(name=__name__)

    X_train, y_train, X_val, y_val = load_data(
        paths={"train": args.train, "val": args.val}, test_mode=False
    )

    # --------------------------- Build and train model -------------------------- #

    cnn_model, early_stopper = fine_tune_cnn(
        logger=logger,
        train_data=[X_train, y_train],
        val_data=[X_val, y_val],
        dense_params={
            "dense_units": args.dense_units,
            "dense_weight_decay": args.dense_weight_decay,
        },
        aug_params={
            "random_contrast_factor": args.random_contrast_factor,
            "random_flip_mode": args.random_flip_mode,
            "random_rotation_factor": args.random_rotation_factor,
            "random_zoom_factor": args.random_zoom_factor,
        },
        opt_params={"learning_rate": args.learning_rate, "clipnorm": args.clipnorm},
        fit_params={"batch_size": args.batch_size, "epochs": args.epochs},
        dropout_rate=args.dropout_rate,
    )

    logger.info(f"Best validation accuracy after fine-tuning: {early_stopper.best}")

    # Save model, a version number is needed for the TF serving container to load the model
    cnn_model.save(os.path.join(args.model_dir, "00000000"))
