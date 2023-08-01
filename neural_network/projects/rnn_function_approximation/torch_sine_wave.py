import argparse
import os
import sys
import logging
from typing import List, Tuple, Union
from multiprocessing import Pool

import numpy as np

# ------------------------------- Generate data ------------------------------ #

def generate_2d_sine_wave(samples: int, freq_x: float = 1, freq_y: float = 1, noise: float = 0.1) -> np.ndarray:
    """
    Generate a 2D sine wave.

    Parameters
    ----------
    samples : int
        The number of samples to generate.
    freq_x : float, optional
        The frequency (number of times a wave repeats a cycle) of the sine wave in the x direction, by default 1.
    freq_y : float, optional
        The frequency (number of times a wave repeats a cycle) of the sine wave in the y direction, by default 1.
    noise : float, optional
        The amount of noise (scalar factor) to add to the sine wave, by default 0.1.

    Returns 
    -------
    np.ndarray
        A 2D sine wave with float32 data type.
    """
    x = np.linspace(0, 2 * np.pi, samples)
    y = np.linspace(0, 2 * np.pi, samples)
    # Each (i, j) is computed as as np.sin(freq_x * x[i]) + np.sin(freq_y * y[j]) plus some noise
    sinewave_2d = np.sin(freq_x * x)[:, None] + np.sin(freq_y * y)[None, :] + noise * np.random.randn(samples, samples)
    
    return sinewave_2d.astype(np.float32)
    
# ------------------------------ Main function ------------------------------- #

def main() -> int:
    
    logger = get_logger('torch_2d_sine_wave')
    
    # ------------------- Parse arguments from the command line ------------------ #
    
    parser = argparse.ArgumentParser(description='PyTorch LSTM and GRU for 2D Sine Wave')
    parser.add_argument('--samples', type=int, default=100, help='The number of samples to generate')
    parser.add_argument('--freq_x', type=float, default=1, help='The frequency (number of times a wave repeats a cycle) of the sine wave in the x direction')
    parser.add_argument('--freq_y', type=float, default=1, help='The frequency (number of times a wave repeats a cycle) of the sine wave in the y direction')
    parser.add_argument('--noise', type=float, default=0.1, help='The amount of noise (scalar factor) to add to the sine wave')
    parser.add_argument('--input_size', type=int, default=1, help='The number of expected features in the input x')
    parser.add_argument('--hidden_size', type=int, default=50, help='The number of features in the hidden state h')
    parser.add_argument('--num_layers', type=int, default=2, help='The number of recurrent layers')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs to train')
    args, _ = parser.parse_known_args()
    
    # ------------------------------- Generate data ------------------------------ #
    
    logger.info('Generating 2D sine wave...')
    
    data = generate_2d_sine_wave(args.samples, args.freq_x, args.freq_y, args.noise)
    tensor_data = grid_to_seq(data)
    
    # ---------------------- Train LSTM and GRU in parallel ---------------------- #
    
    logger.info('Training LSTM and GRU in parallel...')
    
    hyperparameters = {'input_size': args.input_size, 'hidden_size': args.hidden_size, 'num_layers': args.num_layers}
    
    with Pool(processes=2) as p:
        lstm_predictions, gru_predictions = p.starmap(
            trainer, 
            [(tensor_data, 'lstm', hyperparameters, args.epochs, args.learning_rate, logger), 
             (tensor_data, 'gru', hyperparameters, args.epochs, args.learning_rate, logger)]
        )
    
    # ------------------------------- Plot results ------------------------------- #
    
    logger.info('Plotting results...')
    
    plot_predictions(data, lstm_predictions, gru_predictions)

    return 0

if __name__ == '__main__':
    
    from custom_utils import grid_to_seq, LSTM, GRU, trainer, plot_predictions, get_logger
    
    sys.exit(main())