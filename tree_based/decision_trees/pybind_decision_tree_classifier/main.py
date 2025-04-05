import argparse
import time

import numpy as np
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from decision_tree_classifier import DecisionTreeClassifier


def main() -> int:
    parser = argparse.ArgumentParser(description="Decision tree classifier benchmark")
    parser.add_argument(
        "--n_samples",
        type=int,
        default=20_000,
        help="Number of samples for the synthetic dataset (default: 20000)",
    )
    parser.add_argument(
        "--n_features",
        type=int,
        default=100,
        help="Number of features for the synthetic dataset (default: 120)",
    )
    parser.add_argument(
        "--n_classes",
        type=int,
        default=4,
        help="Number of classes for the synthetic dataset (default: 4)",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=27,
        help="Random seed for reproducibility (default: 27)",
    )
    parser.add_argument(
        "--max_depth",
        type=int,
        default=5,
        help="Maximum depth of the decision tree (default: 5)",
    )
    parser.add_argument(
        "--min_samples_split",
        type=int,
        default=2,
        help="Minimum number of samples required to split an internal node (default: 2)",
    )
    parser.add_argument(
        "--criterion",
        type=str,
        default="entropy",
        choices=["gini", "entropy"],
        help="Function to measure the quality of a split (default: 'entropy')",
    )
    parser.add_argument(
        "--min_impurity_decrease",
        type=float,
        default=1e-3,
        help="A node will be split if this split induces a decrease of the impurity greater than or equal to this value (default: 1e-3)",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Proportion of the dataset to include in the test split (default: 0.2)",
    )

    args, _ = parser.parse_known_args()

    rng = np.random.default_rng(args.random_seed)
    random_state = np.random.RandomState(args.random_seed)
    # Randomly determine the number of informative features between 2 and n_features (inclusive)
    n_informative = rng.integers(low=2, high=args.n_features + 1)
    print(f"Randomly chosen n_informative: {n_informative}")

    X, y = make_classification(
        n_samples=args.n_samples,
        n_features=args.n_features,
        n_informative=n_informative,
        n_redundant=0,
        n_clusters_per_class=1,
        n_classes=args.n_classes,
        random_state=random_state,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=random_state
    )

    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_train class distribution: {np.bincount(y_train) / len(y_train)}")
    print(f"y_test class distribution: {np.bincount(y_test) / len(y_test)}")

    X_train_np = X_train.astype(np.float64)
    y_train_np = y_train.astype(np.int32)
    X_test_np = X_test.astype(np.float64)

    # Test with NumPy arrays (zero-copy approach)
    print("\nTesting with NumPy arrays (zero-copy):")
    clf = DecisionTreeClassifier(
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        criterion=args.criterion,
        min_impurity_decrease=args.min_impurity_decrease,
    )

    # Time the fit with NumPy arrays
    start_time = time.time()
    clf.fit(X_train_np, y_train_np)
    fit_time = time.time() - start_time

    # Time the predict with NumPy arrays
    start_time = time.time()
    y_preds = clf.predict(X_test_np)
    predict_time = time.time() - start_time

    acc_score = accuracy_score(y_test, y_preds)
    print(f"Accuracy: {acc_score:.4f}")
    print(f"Fit time: {fit_time:.4f} seconds")
    print(f"Predict time: {predict_time:.4f} seconds")

    # Test with Python lists for comparison
    print("\nTesting with Python lists (copying):")
    clf_lists = DecisionTreeClassifier(
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        criterion=args.criterion,
        min_impurity_decrease=args.min_impurity_decrease,
    )

    # Time the fit with lists
    start_time = time.time()
    clf_lists.fit(X_train.tolist(), y_train.tolist())
    list_fit_time = time.time() - start_time

    # Time the predict with lists
    start_time = time.time()
    y_preds_lists = clf_lists.predict(X_test.tolist())
    list_predict_time = time.time() - start_time

    acc_score_lists = accuracy_score(y_test, y_preds_lists)
    print(f"Accuracy: {acc_score_lists:.4f}")
    print(f"Fit time: {list_fit_time:.4f} seconds")
    print(f"Predict time: {list_predict_time:.4f} seconds")

    # Calculate speedup
    fit_speedup = list_fit_time / fit_time if fit_time > 0 else float("inf")
    predict_speedup = (
        list_predict_time / predict_time if predict_time > 0 else float("inf")
    )

    print("\nPerformance comparison:")
    print(f"Fit speedup with Eigen: {fit_speedup:.2f}x")
    print(f"Predict speedup with Eigen: {predict_speedup:.2f}x")

    return 0


if __name__ == "__main__":
    main()
