# ---------------------------------- Imports --------------------------------- #

import pandas as pd
import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    balanced_accuracy_score,
)
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------- Plot function ------------------------------ #


def plot_histograms(var, seq_data, path=None):
    """
    Plot histograms for each variable in the dataset for all five folds.
    """
    fig = plt.figure(figsize=(16, 9))
    plt.rcParams.update({"font.size": 8})
    plt.subplots_adjust(hspace=0.2, wspace=0.5)

    # Fold 1
    ax1 = plt.subplot2grid(shape=(2, 6), loc=(0, 0), colspan=2)
    ax1.hist(seq_data[0][var])
    ax1.set_title(f"Fold 1: Sample of {len(seq_data[0]):,.0f}")

    # Fold 2
    ax2 = plt.subplot2grid(shape=(2, 6), loc=(0, 2), colspan=2)
    ax2.hist(seq_data[1][var])
    ax2.set_title(f"Fold 2: Sample of {len(seq_data[1]):,.0f}")

    # Fold 3
    ax3 = plt.subplot2grid(shape=(2, 6), loc=(0, 4), colspan=2)
    ax3.hist(seq_data[2][var])
    ax3.set_title(f"Fold 3: Sample of {len(seq_data[2]):,.0f}")

    # Fold 4
    ax4 = plt.subplot2grid(shape=(2, 6), loc=(1, 1), colspan=2)
    ax4.hist(seq_data[3][var])
    ax4.set_title(f"Fold 4: Sample of {len(seq_data[3]):,.0f}")

    # Fold 5
    ax5 = plt.subplot2grid(shape=(2, 6), loc=(1, 3), colspan=2)
    ax5.hist(seq_data[4][var])
    ax5.set_title(f"Fold 5: Sample of {len(seq_data[4]):,.0f}")

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
        accuracy[i] = balanced_accuracy_score(
            y_true=seq_data[i]["target"], y_pred=seq_data[i]["predictions"] > 0.5
        )

    # Compute average accuracy
    avg_accuracy = np.mean(accuracy)
    return avg_accuracy


# --------------------------- Plot confusion matrix -------------------------- #


def plot_cm(labels, predictions, p=0.5):
    cm = confusion_matrix(labels, predictions > p)
    plt.rcParams["figure.figsize"] = (17, 15)
    plt.figure(figsize=(5, 5))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title("Confusion matrix @{:.2f}".format(p))
    plt.ylabel("Actual label")
    plt.xlabel("Predicted label")

    print("True Negatives: ", cm[0][0])
    print("False Positives:", cm[0][1])
    print("False Negatives: ", cm[1][0])
    print("True Positives: ", cm[1][1])
    print("Total Positive Case: ", np.sum(cm[1]))


# --------------------------------- Plot ROC --------------------------------- #


def plot_roc(name, labels, predictions, **kwargs):
    fp, tp, _ = roc_curve(labels, predictions)
    plt.rcParams["figure.figsize"] = (17, 15)
    plt.plot(100 * fp, 100 * tp, label=name, linewidth=2, **kwargs)
    plt.plot([0, 100], [0, 100], "k--", label="chance level (AUC = 0.5)")
    plt.xlabel("False positives [%]")
    plt.ylabel("True positives [%]")
    plt.grid(True)
    ax = plt.gca()
    ax.set_aspect("equal")


# --------------------------------- Plot PRC --------------------------------- #


def plot_prc(name, labels, predictions, **kwargs):
    precision, recall, _ = precision_recall_curve(labels, predictions)

    plt.plot(precision, recall, label=name, linewidth=2, **kwargs)
    plt.xlabel("Precision")
    plt.ylabel("Recall")
    plt.grid(True)
    ax = plt.gca()
    ax.set_aspect("equal")
