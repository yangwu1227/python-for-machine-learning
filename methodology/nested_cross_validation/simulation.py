import argparse
from typing import Any, List, Tuple

import numpy as np
import polars as pl
from sklearn.model_selection import KFold, StratifiedGroupKFold, StratifiedKFold


class NonUniqueFoldsError(ValueError):
    """
    Exception raised when cross-validation folds are not unique.

    This error occurs when validation sets in cross-validation contain
    the same samples across different folds, which violates the
    principle of cross-validation.
    """

    pass


def simulate_nested_cv(
    n: int,
    outer_folds: int,
    inner_folds: int,
    outer_seed: int,
    inner_seed: int,
    n_groups: int = 25,
) -> pl.DataFrame:
    """
    Simulate nested cross-validation using different scikit-learn splitters.

    This function creates synthetic binary classification data and simulates a nested
    cross-validation procedure with three different cross-validation splitters:
    `KFold`, `StratifiedKFold`, and `StratifiedGroupKFold`. It collects and returns statistics
    about the training and validation set sizes in both inner and outer loops.

    Parameters
    ----------
    n : int
        Number of samples in the synthetic dataset.
    outer_folds : int
        Number of folds for the outer cross-validation loop.
    inner_folds : int
        Number of folds for the inner cross-validation loop.
    outer_seed : int
        Random seed for the outer loop data generation and splitting.
    inner_seed : int
        Random seed for generating the inner loop splitter seeds.
    n_groups : int
        Number of groups for StratifiedGroupKFold.

    Returns
    -------
    pl.DataFrame
        A DataFrame containing statistics for each splitter with columns:
        - splitter: name of the cross-validation splitter
        - total_n: total number of samples
        - avg_outer_train: average size of training sets in outer CV
        - avg_outer_val: average size of validation sets in outer CV
        - avg_inner_train: average size of training sets in inner CV
        - avg_inner_val: average size of validation sets in inner CV
    """
    outer_rng = np.random.RandomState(outer_seed)
    inner_rng = np.random.RandomState(inner_seed)

    y = outer_rng.randint(0, 2, size=n)
    group_probs = outer_rng.dirichlet(alpha=np.ones(n_groups))
    groups = outer_rng.choice(n_groups, size=n, p=group_probs)

    results = []
    splitters: List[Tuple[str, Any]] = [
        ("KFold", KFold),
        ("StratifiedKFold", StratifiedKFold),
        ("StratifiedGroupKFold", StratifiedGroupKFold),
    ]

    for name, splitter in splitters:
        # Draw a fresh stream of inner seeds for this splitter
        inner_seed_stream = inner_rng.randint(0, 2**31 - 1, size=outer_folds)

        outer = splitter(
            n_splits=outer_folds,
            shuffle=True,
            random_state=outer_seed,
        )

        outer_train_sizes = []
        outer_val_sizes = []
        outer_val_sets = []
        inner_train_sizes = []
        inner_val_sizes = []

        # Use a 2D dummy X for scikit-learn
        X = np.zeros((n, 1))

        for fold_index_outer, (train_indices_outer, val_indices_outer) in enumerate(
            outer.split(
                X=X,
                y=y if "Stratified" in name else None,
                groups=groups if name == "StratifiedGroupKFold" else None,
            )
        ):
            outer_train_sizes.append(len(train_indices_outer))
            outer_val_sizes.append(len(val_indices_outer))
            outer_val_sets.append(frozenset(val_indices_outer))

            inner_seed_value = int(inner_seed_stream[fold_index_outer])
            if name == "KFold":
                inner = KFold(
                    n_splits=inner_folds, shuffle=True, random_state=inner_seed_value
                )
                split_args: Tuple[Any, ...] = ()
            elif name == "StratifiedKFold":
                inner = StratifiedKFold(
                    n_splits=inner_folds, shuffle=True, random_state=inner_seed_value
                )
                split_args = (y[train_indices_outer],)
            else:
                inner = StratifiedGroupKFold(
                    n_splits=inner_folds, shuffle=True, random_state=inner_seed_value
                )
                split_args = (y[train_indices_outer], groups[train_indices_outer])

            # Dummy X for inner splits
            X_inner = np.zeros((len(train_indices_outer), 1))
            inner_val_sets = []
            for train_indices_inner, val_indices_inner in inner.split(
                X_inner,
                *split_args,
            ):
                inner_train_sizes.append(len(train_indices_inner))
                inner_val_sizes.append(len(val_indices_inner))
                inner_val_sets.append(frozenset(train_indices_outer[val_indices_inner]))

            # Check inner-fold uniqueness
            if len(inner_val_sets) != len(set(inner_val_sets)):
                raise NonUniqueFoldsError(
                    f"{name} inner folds in outer fold {fold_index_outer} are not unique"
                )

        # Check outer-fold uniqueness
        if len(outer_val_sets) != len(set(outer_val_sets)):
            raise NonUniqueFoldsError(f"{name} outer folds are not unique")

        results.append(
            {
                "splitter": name,
                "total_n": n,
                "avg_outer_train": float(np.mean(outer_train_sizes)),
                "avg_outer_val": float(np.mean(outer_val_sizes)),
                "avg_inner_train": float(np.mean(inner_train_sizes)),
                "avg_inner_val": float(np.mean(inner_val_sizes)),
            }
        )

    return pl.DataFrame(results)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Simulate nested CV with different splitters"
    )
    parser.add_argument("--n", type=int, default=20_000, help="Number of samples")
    parser.add_argument(
        "--outer-folds", type=int, default=5, help="Number of outer folds"
    )
    parser.add_argument(
        "--inner-folds", type=int, default=10, help="Number of inner folds"
    )
    parser.add_argument("--outer-seed", type=int, default=27, help="Seed for outer RNG")
    parser.add_argument("--inner-seed", type=int, default=17, help="Seed for inner RNG")
    parser.add_argument(
        "--n-groups",
        type=int,
        default=25,
        help="Number of groups for StratifiedGroupKFold",
    )
    args = parser.parse_args()

    data = simulate_nested_cv(
        n=args.n,
        outer_folds=args.outer_folds,
        inner_folds=args.inner_folds,
        outer_seed=args.outer_seed,
        inner_seed=args.inner_seed,
        n_groups=args.n_groups,
    )
    print(data)
    return 0


if __name__ == "__main__":
    main()
