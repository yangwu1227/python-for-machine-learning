import os
from functools import partial
from typing import List, Tuple

import tensorflow as tf
from model_utils import DataHandler, get_logger, parser

logger = get_logger(__name__)

# ------------------------- Function for building cnn ------------------------ #


def build_cnn(
    filters_list: List[int],
    conv2d_regularizer_decay: float,
    dense_units_list: List[int],
    dense_regularizer_decay: float,
    kernel_size: int,
    dropout_rate: float,
    batch_norm_momentum: float,
    learning_rate: float,
    clipnorm: float,
    input_shape: Tuple[int, int, int] = (28, 28, 1),
) -> tf.keras.models.Sequential:
    """
    Build and compile a convolutional neural network with the following architecture:
    - 5 convolutional layers with 3 x 3 kernel size and ReLU activation
    - 3 max pooling layers with 2 x 2 pool size
    - 2 dense layers with ReLU activation
    - 1 output layer with softmax activation

    Parameters
    ----------
    filters_list : List[int]
        A list of integers representing the filter dimensions outputted by each convolutional layer
    conv2d_regularizer_decay : float
        L2 regularization decay for convolutional layers
    dense_units_list : List[int]
        A list of integers representing the number of units in each dense layer
    dense_regularizer_decay : float
        L2 regularization decay for dense layers
    kernel_size : int
        Size of the kernel for the first convolutional layer
    dropout_rate : float
        Dropout rate for the dropout layers
    batch_norm_momentum : float
        Momentum for the batch normalization layers
    learning_rate : float
        Learning rate for the Adam optimizer
    clipnorm : float
        Clipnorm for the Adam optimizer
    input_shape : Tuple[int, int, int], optional
        Dimension of the input feature vector, by default (28, 28, 1)

    Returns
    -------
    tf.keras.models.Sequential
        A compiled convolutional neural network
    """
    # Default convolutional layer
    DefaultConv2D = partial(
        tf.keras.layers.Conv2D,
        kernel_size=3,
        padding="same",
        activation="relu",
        kernel_initializer="he_normal",
        kernel_regularizer=tf.keras.regularizers.l2(conv2d_regularizer_decay),
    )
    # Default dense layer
    DefaultDense = partial(
        tf.keras.layers.Dense,
        activation="relu",
        kernel_initializer="he_normal",
        kernel_regularizer=tf.keras.regularizers.l2(dense_regularizer_decay),
    )

    # Model architecture
    cnn_model = tf.keras.Sequential(
        [
            # First convolutional layer can have larger kernel size (more than 3 x 3)
            DefaultConv2D(
                filters=filters_list[0],
                kernel_size=kernel_size,
                input_shape=input_shape,
            ),
            tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
            DefaultConv2D(filters=filters_list[1]),
            DefaultConv2D(filters=filters_list[2]),
            tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
            DefaultConv2D(filters=filters_list[3]),
            DefaultConv2D(filters=filters_list[4]),
            tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
            # The Dense layer expects a 1D array of features for each instance, so we need to flatten its inputs
            tf.keras.layers.Flatten(),
            DefaultDense(units=dense_units_list[0]),
            tf.keras.layers.BatchNormalization(momentum=batch_norm_momentum),
            tf.keras.layers.Dropout(dropout_rate),
            DefaultDense(units=dense_units_list[1]),
            tf.keras.layers.BatchNormalization(momentum=batch_norm_momentum),
            tf.keras.layers.Dropout(dropout_rate),
            DefaultDense(units=10, activation="softmax"),
        ]
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=clipnorm)
    cnn_model.compile(
        # Used when labels are a 1D integer vector rather than one-hotted
        loss=tf.keras.losses.sparse_categorical_crossentropy,
        optimizer=optimizer,
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )

    return cnn_model


def main() -> int:
    args = parser()

    # ------------------------------ Data ingestion ------------------------------ #

    data_handler = DataHandler(args.s3_bucket, args.s3_key)
    X_train, y_train = data_handler.load_data(mode="train")
    X_val, y_val = data_handler.load_data(mode="val")

    logger.info(
        f"Successfully load training set with shapes {X_train.shape} and validation set with shapes {X_val.shape}"
    )

    # --------------------------- Build and train model -------------------------- #

    cnn_model = build_cnn(
        filters_list=[
            args.filter_dim_1,
            args.filter_dim_2,
            args.filter_dim_3,
            args.filter_dim_4,
            args.filter_dim_5,
        ],
        conv2d_regularizer_decay=args.conv2d_regularizer_decay,
        dense_units_list=[args.dense_units_1, args.dense_units_2],
        dense_regularizer_decay=args.dense_regularizer_decay,
        kernel_size=args.kernel_size,
        dropout_rate=args.dropout_rate,
        batch_norm_momentum=args.batch_norm_momentum,
        learning_rate=args.learning_rate,
        clipnorm=args.clipnorm,
    )

    early_stopper = tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy", patience=10, restore_best_weights=True
    )
    cnn_model.fit(
        x=X_train,
        y=y_train,
        batch_size=args.batch_size,
        epochs=args.epochs,
        callbacks=[early_stopper],
        validation_data=(X_val, y_val),
        verbose=2,
    )

    logger.info(f"Best validation accuracy: {early_stopper.best}")

    # Save model, a version number is needed for the TF serving container to load the model
    cnn_model.save(os.path.join(args.model_dir, "00000000"))

    return 0


if __name__ == "__main__":
    main()
