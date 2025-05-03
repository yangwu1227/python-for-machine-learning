import argparse
import os
from datetime import datetime
from typing import Tuple

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Nopep8
import keras_tuner as kt
import numpy as np
import tensorflow as tf
from model_utils import plot_1d_curve, report_keras_hpo, setup_logger

logger = setup_logger("tf_sine_apprixmator")

# ------------------------------ Data generator ------------------------------ #


def generate_sine_wave_data(
    start: float = -np.pi, end: float = np.pi, num_samples: int = 2000
) -> Tuple[np.ndarray, np.ndarray]:
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

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Tuple of X and y.
    """
    X = np.linspace(start, end, num_samples)
    y = np.sin(X)

    return X, y


# ------------------------------ Tuner subclass ------------------------------ #


class SineHyperModel(kt.HyperModel):
    """
    This class inherits from the `keras_tuner` `HyperModel` class, which defines a search space of models.
    A search space is a collection of models. The build function will build one of the models from the
    space using the given HyperParameters object. We subclass the `HyperModel` class to define our search
    spaces by overriding the `build()` method, which creates and returns the Keras model. Optionally, we
    can also override the `fit()` method to customize the training process of the model.
    """

    def build(self, hp: kt.HyperParameters) -> tf.keras.Model:
        """
        This method builds a Keras model using the given HyperParameters object.

        Parameters
        ----------
        hp : kt.HyperParameters
            Container for both the hyperparameter space and the current values. This object
            has two attributes--- space (a list of HyperParameter objects) and values (a dict
            mapping hyperparameter names to current values).

        Returns
        -------
        tf.keras.Model
            A Keras model.
        """
        activation = hp.Choice("activation", values=["relu", "tanh"])
        kernel_initializer = hp.Choice(
            "kernel_initializer", values=["glorot_normal", "he_normal", "lecun_normal"]
        )
        learning_rate = hp.Float(
            "learning_rate", min_value=1e-4, max_value=1e-2, sampling="log"
        )
        optimizer = getattr(
            tf.keras.optimizers,
            hp.Choice("optimizer", values=["Adam", "SGD", "RMSprop"]),
        )(learning_rate=learning_rate)

        # The sine wave we approximate is one-dimensional
        inputs = tf.keras.Input(shape=(1,), name="input_layer")
        for i in range(hp.Int("n_layers", 1, 3)):
            if i == 0:
                x = tf.keras.layers.Dense(
                    units=hp.Int(
                        f"layer_{i}_n_units", min_value=64, max_value=256, step=16
                    ),
                    activation=activation,
                    name=f"layer_{i}",
                )(inputs)
            else:
                units = hp.Int(
                    f"layer_{i}_n_units", min_value=64, max_value=256, step=16
                )
                x = tf.keras.layers.Dense(
                    units, activation=activation, name=f"layer_{i}"
                )(x)
        outputs = tf.keras.layers.Dense(
            units=1, activation="linear", name="output_layer"
        )(x)
        model = tf.keras.Model(inputs=inputs, outputs=outputs, name="sine_approximator")
        model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.MeanAbsoluteError(name="mse_loss"),
            metrics=[tf.keras.metrics.MeanSquaredError(name="mse_metric")],
        )

        return model

    def fit(
        self, hp: kt.HyperParameters, model: tf.keras.Model, *args, **kwargs
    ) -> float:
        """
        This method fits the Keras model using the given HyperParameters object. The *args and
        **kwargs are passed from `tuner.search(*args, **kwargs)`.

        Parameters
        ----------
        hp : kt.HyperParameters
            HyperParameters object.
        model : tf.keras.Model
            Keras model returned by the `build()` method.

        Returns
        -------
        float
            The best loss of the model.
        """
        history = model.fit(
            *args,
            batch_size=hp.Choice("batch_size", values=[16, 32, 64, 128]),
            epochs=hp.Int("epochs", min_value=10, max_value=20, step=2),
            **kwargs,
        )

        return history


def main() -> int:
    # Suppress TensorFlow warnings and info, only show errors
    tf.setup_logger().setLevel("ERROR")

    # Create working directory if it does not already exist
    output_dir = os.path.join(__file__, "outputs/sine_approximator_hpo")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # -------------------------- Command line arguments -------------------------- #

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_sample",
        type=int,
        default=2000,
        help="Number of training samples to generate",
    )
    parser.add_argument(
        "--start", type=float, default=-np.pi, help="Start of the interval"
    )
    parser.add_argument("--end", type=float, default=np.pi, help="End of the interval")
    parser.add_argument(
        "--max_trials",
        type=int,
        default=20,
        help="Number of hyperparameter trials to run",
    )
    parser.add_argument(
        "--project_name",
        type=str,
        default="sine_approximator",
        help="Name of the project",
    )
    parser.add_argument(
        "--overwrite",
        type=bool,
        default=True,
        help="Whether to overwrite the existing project",
    )
    parser.add_argument(
        "--hypermodel_name",
        type=str,
        default="sine_hypermodel",
        help="Name of the hypermodel",
    )
    parser.add_argument("--verbose", type=int, default=0, help="Verbosity level")
    parser.add_argument(
        "--tensorboard", type=int, default=0, help="Whether to use TensorBoard"
    )
    parser.add_argument(
        "--plot_name",
        type=str,
        default="Sine Wave",
        help="Name of function we are approximating",
    )
    args, _ = parser.parse_known_args()

    # ------------------------------- Simulate data ------------------------------ #

    logger.info("Simulating training data...")

    X, y = generate_sine_wave_data(
        start=args.start, end=args.end, num_samples=args.num_sample
    )

    # -------------------------------- Keras tuner ------------------------------- #

    tuner = kt.BayesianOptimization(
        hypermodel=SineHyperModel(name=args.hypermodel_name),
        objective=kt.Objective("val_mse_metric", direction="min"),
        max_trials=args.max_trials,
        # Arguments passed to the Tuner class
        directory=output_dir,
        project_name=args.project_name,
        overwrite=args.overwrite,
        logger=logger,
    )

    # --------------------------------- Callbacks -------------------------------- #

    early_stopper = tf.keras.callbacks.EarlyStopping(
        monitor="val_mse_metric",
        patience=3,
        restore_best_weights=True,
        mode="min",
        min_delta=1e-4,
    )

    if args.tensorboard:
        tensorboard = tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join(
                output_dir, "tb-logs-v" + datetime.now().strftime("%y-%m-%d-%H-%M-%S")
            )
        )

    # --------------------------- Bayesian optimization -------------------------- #

    logger.info("Running Bayesian optimization...")

    tuner.search(
        x=X,
        y=y,
        verbose=args.verbose,
        validation_data=(X, y),
        callbacks=[early_stopper, tensorboard] if args.tensorboard else [early_stopper],
    )

    report_keras_hpo(tuner, X, y, logger)

    # ------------------ Retrain model with best hyperparameters ----------------- #

    logger.info("Retraining model with best hyperparameters...")

    best_hp = tuner.get_best_hyperparameters()[0]
    best_model = tuner.hypermodel.build(best_hp)
    best_model.fit(
        x=X,
        y=y,
        batch_size=best_hp.get("batch_size"),
        epochs=best_hp.get("epochs"),
        verbose=args.verbose,
        validation_data=(X, y),
        callbacks=[early_stopper],
    )

    # ------------------------ Plot approximated function ------------------------ #

    logger.info("Predicting and plotting...")

    # Flatten predictions to 1D array
    y_pred = best_model(X).numpy().flatten()

    plot_1d_curve(y_true=y, y_pred=y_pred, X=X, name=args.plot_name)

    return 0


if __name__ == "__main__":
    main()
