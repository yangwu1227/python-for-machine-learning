# ---------------------------------- Imports --------------------------------- #

import pandas as pd
import numpy as np
from sklearn.metrics import balanced_accuracy_score
import matplotlib.pyplot as plt

# ------------------------------- Plot function ------------------------------ #

def plot_histograms(var, seq_data, path=None):
  """
  Plot histograms for each variable in the dataset for all five folds.
  """
  fig = plt.figure(figsize=(16, 9))
  plt.rcParams.update({'font.size': 8})
  plt.subplots_adjust(hspace=0.2, wspace=0.5)

  # Fold 1
  ax1 = plt.subplot2grid(shape=(2,6), loc=(0,0), colspan=2)
  ax1.hist(seq_data[0][var])
  ax1.set_title(f'Fold 1: Sample of {len(seq_data[0]):,.0f}')

  # Fold 2
  ax2 = plt.subplot2grid(shape=(2,6), loc=(0,2), colspan=2)
  ax2.hist(seq_data[1][var])
  ax2.set_title(f'Fold 2: Sample of {len(seq_data[1]):,.0f}')
      
  # Fold 3
  ax3 = plt.subplot2grid(shape=(2,6), loc=(0,4), colspan=2)
  ax3.hist(seq_data[2][var])
  ax3.set_title(f'Fold 3: Sample of {len(seq_data[2]):,.0f}')

  # Fold 4
  ax4 = plt.subplot2grid(shape=(2,6), loc=(1,1), colspan=2)
  ax4.hist(seq_data[3][var])
  ax4.set_title(f'Fold 4: Sample of {len(seq_data[3]):,.0f}')
  
  # Fold 5
  ax5 = plt.subplot2grid(shape=(2,6), loc=(1,3), colspan=2)
  ax5.hist(seq_data[4][var])
  ax5.set_title(f'Fold 5: Sample of {len(seq_data[4]):,.0f}')
      
  if path:
    plt.savefig(path)
  
  plt.show()

# ------------------- Function to compute average accuracy ------------------- #

def compute_average_accuracy(seq_data):
  """
  Compute the average accuracy of the model on five folds.
  """
  # Compute accuracy for each fold
  accuracy = np.empty(shape=(5,))
  for i in range(5):
    accuracy[i] = balanced_accuracy_score(y_true=seq_data[i]['target'], y_pred=seq_data[i]['predictions'] > 0.5)

  # Compute average accuracy
  avg_accuracy = np.mean(accuracy)
  return avg_accuracy
