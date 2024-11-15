import argparse
import sys
from multiprocessing import Pool

import numpy as np

# ------------------------------- Generate data ------------------------------ #


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

    return saddle.astype(np.float32)


# ------------------------------ Data generator ------------------------------ #


def generate_monkey_saddle_data(
    start: float = -np.e, end: float = np.e, num_samples: int = 500, **kwargs
) -> np.ndarray:
    """
    Generate training data for monkey saddle function.

    Parameters
    ----------
    start : float, optional
        Start of the interval, by default -np.e.
    end : float, optional
        End of the interval, by default np.e.
    num_samples : int, optional
        Number of samples to generate, by default 500.
    **kwargs
        Additional keyword arguments to be passed to the monkey_saddle function.

    Returns
    -------
    np.ndarray
        A 2D grid of values representing the output of the monkey saddle function.
    """
    # X and Y are each num_samples x num_samples matrices
    x, y = np.meshgrid(
        np.linspace(start, end, num_samples), np.linspace(start, end, num_samples)
    )
    # Z is the matrix where z[i, j] is the output of the monkey_saddle function for x[i, j] and y[i, j]
    z = monkey_saddle(x, y, **kwargs)

    return z.astype(np.float32)


# ------------------------------ Main function ------------------------------- #


def main() -> int:
    logger = get_logger("torch_monkey_saddle")

    # ------------------- Parse arguments from the command line ------------------ #

    parser = argparse.ArgumentParser(
        description="PyTorch LSTM and GRU for Monkey Saddle"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=100,
        help="Number of samples to generate on the real line",
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
    parser.add_argument(
        "--input_size",
        type=int,
        default=1,
        help="The number of expected features in the input x",
    )
    parser.add_argument(
        "--hidden_size",
        type=int,
        default=20,
        help="The number of features in the hidden state h",
    )
    parser.add_argument(
        "--num_layers", type=int, default=3, help="The number of recurrent layers"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.01, help="Learning rate"
    )
    parser.add_argument(
        "--epochs", type=int, default=30, help="Number of epochs to train"
    )
    args, _ = parser.parse_known_args()

    # ------------------------------- Generate data ------------------------------ #

    logger.info("Generating monkey saddle...")

    data = generate_monkey_saddle_data(
        start=args.start,
        end=args.end,
        num_samples=args.samples,
        theta=args.theta,
        flip=args.flip,
        translate=args.translate,
        scale=args.scale,
    )
    tensor_data = grid_to_seq(data)

    # ---------------------- Train LSTM and GRU in parallel ---------------------- #

    logger.info("Training LSTM and GRU in parallel...")

    hyperparameters = {
        "input_size": args.input_size,
        "hidden_size": args.hidden_size,
        "num_layers": args.num_layers,
    }

    with Pool(processes=2) as p:
        lstm_predictions, gru_predictions = p.starmap(
            trainer,
            [
                (
                    tensor_data,
                    "lstm",
                    hyperparameters,
                    args.epochs,
                    args.learning_rate,
                    logger,
                ),
                (
                    tensor_data,
                    "gru",
                    hyperparameters,
                    args.epochs,
                    args.learning_rate,
                    logger,
                ),
            ],
        )

    # ------------------------------- Plot results ------------------------------- #

    logger.info("Plotting results...")

    plot_predictions(data, lstm_predictions, gru_predictions)

    return 0


if __name__ == "__main__":
    from custom_utils import get_logger, grid_to_seq, plot_predictions, trainer

    sys.exit(main())
