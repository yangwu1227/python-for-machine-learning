import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve

# ------------------------------- Plot metrics ------------------------------- #


def plot_metrics(history):
    metrics = ["loss", "prc", "precision", "recall"]
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    plt.rcParams["figure.figsize"] = (17, 15)
    for n, metric in enumerate(metrics):
        name = metric.replace("_", " ").capitalize()
        plt.subplot(2, 2, n + 1)
        plt.plot(history.epoch, history.history[metric], color=colors[0], label="Train")
        plt.plot(
            history.epoch,
            history.history["val_" + metric],
            color=colors[0],
            linestyle="--",
            label="Val",
        )
        plt.xlabel("Epoch")
        plt.ylabel(name)
        if metric == "loss":
            plt.ylim([0, plt.ylim()[1]])
        elif metric == "auc":
            plt.ylim([0.8, 1])
        else:
            plt.ylim([0, 1])

        plt.legend()


# --------------------------- Plot confusion matrix -------------------------- #


def plot_cm(labels, predictions, p=0.5):
    cm = confusion_matrix(labels, predictions > p)
    plt.rcParams["figure.figsize"] = (17, 15)
    plt.figure(figsize=(5, 5))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title("Confusion matrix @{:.2f}".format(p))
    plt.ylabel("Actual label")
    plt.xlabel("Predicted label")

    print(
        "Non-subscriber correctly identified as non-subscriber (True Negatives): ",
        cm[0][0],
    )
    print(
        "Non-subscriber incorrectly identified as subscriber (False Positives): ",
        cm[0][1],
    )
    print(
        "Subscriber incorrectly identified as non-subscriber (False Negatives): ",
        cm[1][0],
    )
    print("Subscriber correctly identified as subscribers (True Positives): ", cm[1][1])
    print("Total subscribers identified: ", np.sum(cm[1]))


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
