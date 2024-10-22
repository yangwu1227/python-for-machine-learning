import os
from typing import List, Tuple

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import auc, confusion_matrix, precision_recall_curve, roc_curve


def get_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Function to get train, validation, test data.
    """
    data_prefix = "preprocessed_data/"

    # Check if directory containing preprocessed data exists
    if not os.path.exists(data_prefix):
        print(
            f"""Expected the following folder {data_prefix} to contain the preprocessed data. 
                 Run data processing first in main notebook before running baselines comparisons"""
        )
        return None

    # Read in data
    features = pd.read_csv(data_prefix + "features_xgboost.csv", header=None)
    labels = pd.read_csv(data_prefix + "tags.csv").set_index("TransactionID")

    # Read in train and validation 'TransactionID's, see data processing script for more details
    valid_users = pd.read_csv(data_prefix + "validation.csv", header=None)
    test_users = pd.read_csv(data_prefix + "test.csv", header=None)

    # Obtain train, validation data using inner joins
    valid_X = features.merge(valid_users, on=[0], how="inner")
    test_X = features.merge(test_users, on=[0], how="inner")

    # Obtain train data by using the complement of the set of validation and test TransactionID's
    train_index = ~(
        features[0].isin(test_users[0].values)
        | (features[0].isin(valid_users[0].values))
    )

    # Obtain train, validation, test data
    train_X = features[train_index]
    valid_y = labels.loc[valid_X[0]]
    test_y = labels.loc[test_X[0]]
    train_y = labels.loc[train_X[0]]
    # Set index to TransactionID
    train_X.set_index([0], inplace=True)
    valid_X.set_index([0], inplace=True)
    test_X.set_index([0], inplace=True)

    # Join labels to data as the first column
    train_data = train_y.join(train_X)  # First column is the label 'isFraud'
    valid_data = valid_y.join(valid_X)
    test_data = test_y.join(test_X)

    return train_data, valid_data, test_data


def print_metrics(y_true: np.ndarray, y_predicted: np.ndarray) -> List[np.float]:
    """
    Print model performance metrics.

    Parameters
    ----------
    y_true : np.ndarray
    y_predicted : np.ndarray

    Returns
    -------
    List[np.float]
        Performance metrics--- f1, precision, recall, accuracy.
    """
    cm = confusion_matrix(y_true, y_predicted)
    true_neg, false_pos, false_neg, true_pos = cm.ravel()
    cm = pd.DataFrame(
        np.array([[true_pos, false_pos], [false_neg, true_neg]]),
        columns=["labels positive", "labels negative"],
        index=["predicted positive", "predicted negative"],
    )

    acc = (true_pos + true_neg) / (true_pos + true_neg + false_pos + false_neg)
    precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
    recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )
    return [f1, precision, recall, acc]


def plot_cm(y_true: np.ndarray, y_predicted: np.ndarray, p: float = 0.5) -> None:
    """
    Plot the confusion matrix.

    Parameters
    ----------
    y_true : np.ndarray
    y_predicted : np.ndarray
    p : float, optional
        Prediction threshold, by default 0.5
    """

    # Normalize the rows of the confusion matrix
    cm = confusion_matrix(y_true, y_predicted > p, normalize="true")
    plt.rcParams["figure.figsize"] = (17, 15)
    plt.figure(figsize=(5, 5))
    sns.heatmap(cm, annot=True, fmt="f")
    plt.title("Confusion matrix @{:.2f}".format(p))
    plt.ylabel("Actual label")
    plt.xlabel("Predicted label")

    print("Legitimate Transactions Detected (True Negatives): ", cm[0][0])
    print("Legitimate Transactions Incorrectly Detected (False Positives): ", cm[0][1])
    print("Fraudulent Transactions Missed (False Negatives): ", cm[1][0])
    print("Fraudulent Transactions Detected (True Positives): ", cm[1][1])
    print("Total Fraudulent Transactions: ", np.sum(cm[1]))


def plot_prc(y_true: np.ndarray, y_predicted: np.ndarray, **kwargs) -> None:
    """
    Plot the precision-recall curve.

    Parameters
    ----------
    y_true : np.ndarray
    y_predicted : np.ndarray
    """

    precision, recall, _ = precision_recall_curve(y_true, y_predicted)

    plt.plot(precision, recall, linewidth=2, **kwargs)
    plt.title("Precision-Recall Curve")
    plt.xlabel("Precision")
    plt.ylabel("Recall")
    plt.grid(True)
    ax = plt.gca()
    ax.set_aspect("equal")
