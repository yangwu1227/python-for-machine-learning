from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import HistGradientBoostingClassifier


def simulate_missing_prediction(
    n_classes: int,
    n_samples: int = 1000,
    n_features: int = 10,
    max_depth: int = 5,
    random_state_train: Optional[int] = 42,
    random_state_test: Optional[int] = 43,
) -> pd.DataFrame:
    """
    Train a single-tree HistGradientBoostingClassifier on fully-observed data,
    then evaluate prediction behavior under varying feature-missingness levels.

    Parameters
    ----------
    n_classes : int
        Number of target classes for classification.
    n_samples : int, default=1000
        Number of samples to generate for both training and test datasets.
    n_features : int, default=10
        Total number of features to generate.
    max_depth : int, default=5
        Maximum depth of the single tree in the HistGradientBoostingClassifier.
    random_state_train : int or None, default=42
        Random seed for training data generation.
    random_state_test : int or None, default=43
        Random seed for test data generation.

    Returns
    -------
    pd.DataFrame
        Simulation results:

        For multiclass (n_classes > 2):
        - missing_prob : float
            Fraction of features set to NaN.
        - top_class_fraction : float
            Fraction of samples predicted as the most frequent class.
        - entropy : float
            Shannon entropy (in bits) of the predicted class distribution.
        - class_{i}_fraction : float
            Fraction of predictions for class i, for i in [0, n_classes).

        For binary (n_classes == 2):
        - missing_prob : float
            Fraction of features set to NaN.
        - mean_pos_prob : float
            Mean predicted probability for the positive class.
        - std_pos_prob : float
            Standard deviation of predicted probabilities for the positive class.
    """
    # 1) Generate training and test sets
    X_train, y_train = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_features // 2,
        n_redundant=0,
        n_classes=n_classes,
        random_state=random_state_train,
    )
    X_test, y_test = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_features // 2,
        n_redundant=0,
        n_classes=n_classes,
        random_state=random_state_test,
    )

    # 2) Train a 1-tree HistGradientBoostingClassifier
    clf = HistGradientBoostingClassifier(
        max_iter=1, learning_rate=1.0, max_depth=max_depth, random_state=0
    )
    clf.fit(X_train, y_train)

    # 3) Vary missingness from 0% up to 90%
    missing_probs = np.linspace(0.0, 0.9, 10)
    records = []

    for p in missing_probs:
        X_missing = X_test.copy()
        mask = np.random.rand(*X_missing.shape) < p
        X_missing[mask] = np.nan

        preds = clf.predict(X_missing)
        probs = clf.predict_proba(X_missing)

        if n_classes > 2:
            counts = np.bincount(preds, minlength=n_classes) / len(preds)
            top_frac = counts.max()
            entropy = -(counts[counts > 0] * np.log2(counts[counts > 0])).sum()
            rec = {
                "missing_prob": p,
                "top_class_fraction": top_frac,
                "entropy": entropy,
            }
            for i in range(n_classes):
                rec[f"class_{i}_fraction"] = counts[i]
            records.append(rec)
        else:
            pos_probs = probs[:, 1]
            records.append(
                {
                    "missing_prob": p,
                    "mean_pos_prob": float(pos_probs.mean()),
                    "std_pos_prob": float(pos_probs.std()),
                }
            )

    return pd.DataFrame(records)


def main() -> int:
    # Run simulations
    multi_df = simulate_missing_prediction(n_classes=3)
    binary_df = simulate_missing_prediction(n_classes=2)

    # Display tables
    print("=== Multiclass Simulation Results ===")
    print(multi_df.to_string(index=False))
    print("\n=== Binary Simulation Results ===")
    print(binary_df.to_string(index=False))

    # Plot: Multiclass
    plt.figure()
    plt.plot(multi_df["missing_prob"], multi_df["top_class_fraction"], marker="o")
    plt.title("Multiclass: Top-Class Fraction vs Missing Probability")
    plt.xlabel("Missing Probability")
    plt.ylabel("Fraction of Most-Predicted Class")
    plt.grid(True)
    plt.show()

    plt.figure()
    plt.plot(multi_df["missing_prob"], multi_df["entropy"], marker="o")
    plt.title("Multiclass: Prediction Entropy vs Missing Probability")
    plt.xlabel("Missing Probability")
    plt.ylabel("Entropy (bits)")
    plt.grid(True)
    plt.show()

    # Plot: Binary
    plt.figure()
    plt.plot(binary_df["missing_prob"], binary_df["mean_pos_prob"], marker="o")
    plt.title("Binary: Mean Positive-Class Probability vs Missing Probability")
    plt.xlabel("Missing Probability")
    plt.ylabel("Mean Predicted Positive Probability")
    plt.grid(True)
    plt.show()

    plt.figure()
    plt.plot(binary_df["missing_prob"], binary_df["std_pos_prob"], marker="o")
    plt.title("Binary: STD of Positive-Class Probability vs Missing Probability")
    plt.xlabel("Missing Probability")
    plt.ylabel("STD of Predicted Positive Probability")
    plt.grid(True)
    plt.show()

    return 0


if __name__ == "__main__":
    main()
