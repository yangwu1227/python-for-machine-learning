import logging
import os
from typing import Any, Dict, Tuple

import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Nopep8
import tensorflow as tf
from segmentation_models import Unet
from segmentation_models.losses import DiceLoss

from model_utils import get_logger, load_data, parser

logger = get_logger(name=__name__)

# ------------------------- Function for fine tuning ------------------------- #


def fine_tune_unet(
    logger: logging.Logger,
    back_bone: str,
    train_data: Tuple[np.ndarray, np.ndarray],
    val_data: Tuple[np.ndarray, np.ndarray],
    aug_params: Dict[str, Any],
    opt_params: Dict[str, Any],
    fit_params: Dict[str, Any],
    input_shape: Tuple[int, int, int] = (256, 256, 1),
    verbose: int = 2,
) -> tf.keras.Model:
    """
    Function to fine tune a pre-trained Unet model.

    Parameters
    ----------
    logger: logging.Logger
        Logger to log information to CloudWatch.
    back_bone : str
        Backbone of the Unet model.
    train_data : Tuple[np.ndarray, np.ndarray]
        Training data.
    val_data : Tuple[np.ndarray, np.ndarray]
        Validation data.
    aug_params : Dict[str, Any]
        Hyperparameters for data augmentation.
    opt_params : Dict[str, Any]
        Hyperparameters for optimizer.
    fit_params: Dict[str, Any]
        Hyperparameters for training.
    input_shape : Tuple[int, int, int], optional
        Dimension of the input feature vector, by default (256, 256, 1).
    verbose : int, optional
        Verbosity mode, by default 2.

    Returns
    -------
    tf.keras.Model
        Fine tuned model.
    """

    conv_base = Unet(
        backbone_name=back_bone,
        # Use pre-trained weights
        encoder_weights="imagenet",
        # Freeze the encoder blocks
        encoder_freeze=True,
    )

    # -------------- Train decoder blocks with frozen encoder blocks ------------- #

    logger.info("Begin training decoder blocks with frozen encoder blocks...")

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
    x = tf.keras.layers.Rescaling(scale=(1.0 / 255.0))(x)
    # Map 1 channel to 3 channels for ResNet
    x = tf.keras.layers.Conv2D(
        filters=3, kernel_size=(1, 1), kernel_initializer="he_normal"
    )(x)  # Map 1 channel to 3 channels
    outputs = conv_base(x, training=False)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=opt_params["learning_rate"], clipnorm=opt_params["clipnorm"]
        ),
        loss=DiceLoss(),
        metrics=[
            tf.keras.metrics.MeanIoU(num_classes=2, name="mean_iou"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )

    trainable_params_count = np.sum(
        [tf.keras.backend.count_params(w) for w in model.trainable_weights]
    )
    logger.info(
        f"Number of trainable parameters for decoder training: {trainable_params_count}"
    )
    del trainable_params_count

    early_stopper_decoder = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )

    model.fit(
        x=train_data[0],
        y=train_data[1],
        batch_size=fit_params["batch_size"],
        epochs=fit_params["epochs"],
        validation_data=(val_data[0], val_data[1]),
        callbacks=[early_stopper_decoder],
        verbose=verbose,
    )

    logger.info(
        f"Best validation dice loss after training decoder blocks with frozen encoder blocks: {early_stopper_decoder.best}"
    )

    # --------------- Train entire model with reduced learning rate -------------- #

    logger.info("Begin fine-tuning entire model with reduced learning rate...")

    # Release all layers for training
    for layer in conv_base.layers:
        layer.trainable = True

    # Re-compile with reduced learning rate by a factor of 10
    reduced_learning_rate = opt_params["learning_rate"] / 10
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=reduced_learning_rate, clipnorm=opt_params["clipnorm"]
        ),
        loss=DiceLoss(),
        metrics=[
            tf.keras.metrics.MeanIoU(num_classes=2, name="mean_iou"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )

    trainable_params_count = np.sum(
        [tf.keras.backend.count_params(w) for w in model.trainable_weights]
    )
    logger.info(
        f"Number of trainable parameters for fine-tuning: {trainable_params_count}"
    )
    del trainable_params_count

    early_stopper_fine_tune = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )

    model.fit(
        x=train_data[0],
        y=train_data[1],
        batch_size=fit_params["batch_size"],
        epochs=fit_params["epochs"],
        validation_data=(val_data[0], val_data[1]),
        callbacks=[early_stopper_fine_tune],
        verbose=verbose,
    )

    logger.info(
        f"Best validation dice loss after fine-tuning: {early_stopper_fine_tune.best}"
    )

    return model


def main() -> int:
    args = parser()

    train_images, train_masks, val_images, val_masks = load_data(
        {"train": args.train, "val": args.val}, test_mode=False
    )

    # ------------------------------ Fine-tune model ----------------------------- #

    unet_model = fine_tune_unet(
        logger=logger,
        back_bone=args.back_bone,
        train_data=(train_images, train_masks),
        val_data=(val_images, val_masks),
        aug_params={
            "random_contrast_factor": args.random_contrast_factor,
            "random_flip_mode": args.random_flip_mode,
            "random_rotation_factor": args.random_rotation_factor,
            "random_zoom_factor": args.random_zoom_factor,
        },
        opt_params={"learning_rate": args.learning_rate, "clipnorm": args.clipnorm},
        fit_params={"batch_size": args.batch_size, "epochs": args.epochs},
    )

    # Save model, a version number is needed for the TF serving container to load the model
    unet_model.save(os.path.join(args.model_dir, "00000000"))

    return 0


if __name__ == "__main__":
    main()
