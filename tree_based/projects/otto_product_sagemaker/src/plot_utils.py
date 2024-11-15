from typing import List

import cupy as cp


def misclassification_cost_matrix(target: cp.ndarray, n: int) -> List[float]:
    """
    Function that creates a misclassification cost matrix based on class
    frequencies in the target.

    Parameters
    ----------
    target : cp.ndarray
        The target.
    n : int
        The number of classes.

    Returns
    -------
    List[float]
        The misclassification cost matrix as a flattened list where the
        i * n + j-th element of the list corresponds to the cost of
        misclassifying a sample from class j as class i.
    """
    class_prevalence = (target.value_counts() / len(target)).sort_index().values
    # Initialize with equal weights for all classes
    auc_mu_weights = cp.ones(shape=(n, n))
    # Set diagonal to zero (no error for correct classification)
    cp.fill_diagonal(auc_mu_weights, 0)
    # Row-major order (rows are consecutively placed in memory) when flattened
    auc_mu_weights = cp.ravel(auc_mu_weights, order="C").get().tolist()

    # Modify the weights based on class prevalence
    for i in range(n):
        for j in range(n):
            # Cost of misclassifying class j (true label) as class i (incorrect prediction)
            # If numerator class is more prevalent, the cost is higher
            auc_mu_weights[i * n + j] = float(class_prevalence[i] / class_prevalence[j])

    return auc_mu_weights
