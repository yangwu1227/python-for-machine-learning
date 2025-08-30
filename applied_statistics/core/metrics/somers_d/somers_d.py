import time
from typing import List, Sequence, Tuple, Union

import numba
import numpy as np
import numpy.typing as npt
from sklearn.metrics import roc_auc_score

rng: np.random.Generator = np.random.default_rng(12)


def _validate_and_preprocess_inputs(
    scores: Union[Sequence[float], npt.NDArray[np.floating]],
    labels: Union[Sequence[int], npt.NDArray[np.integer]],
) -> Tuple[npt.NDArray[np.floating], npt.NDArray[np.integer], int, int]:
    """
    Validate and preprocess input scores and labels for Somers' D calculation.

    This function performs common input validation and preprocessing steps
    shared across all Somers' D implementations (except the reference AUC-based one).

    Parameters
    ----------
    scores : array-like of float
        Predicted scores or probabilities for each sample.
    labels : array-like of int
        True binary labels (0 or 1) for each sample.

    Returns
    -------
    scores_arr : ndarray of float
        Validated and converted scores array.
    labels_arr : ndarray of int
        Validated and converted labels array.
    n_pos : int
        Number of positive samples (label = 1).
    n_neg : int
        Number of negative samples (label = 0).

    Raises
    ------
    ValueError
        If scores and labels have different lengths, or if labels contain
        values other than 0 and 1.
    ZeroDivisionError
        If only one class is present (n_pos = 0 or n_neg = 0).
    """
    scores_arr = np.asarray(scores, dtype=float)
    labels_arr = np.asarray(labels, dtype=int)

    if len(scores_arr) != len(labels_arr):
        raise ValueError("scores and labels must have the same length")

    if not np.all(np.isin(labels_arr, [0, 1])):
        raise ValueError("labels must only contain 0 and 1 values")

    n_pos = int(np.sum(labels_arr == 1))
    n_neg = int(np.sum(labels_arr == 0))

    if n_pos == 0 or n_neg == 0:
        raise ZeroDivisionError("Cannot compute Somers' D with only one class present")

    return scores_arr, labels_arr, n_pos, n_neg


def somers_d_auc_reference(
    scores: Union[Sequence[float], npt.NDArray[np.floating]],
    labels: Union[Sequence[int], npt.NDArray[np.integer]],
) -> float:
    """
    Reference implementation using ROC AUC relationship.

    This is the ground truth for binary classification Somers' D. Somers' D is
    a measure of the strength and direction of association between two ranked
    variables. For binary classification, it measures how well the model scores
    rank orders between positive and negative classes.

    Parameters
    ----------
    scores : array-like of float
        Predicted scores or probabilities for each sample. Higher scores should
        correspond to higher likelihood of positive class.
    labels : array-like of int
        True binary labels (0 or 1) for each sample. Must have the same length
        as scores.

    Returns
    -------
    float
        Somers' D statistic in range [-1, 1]. Values closer to 1 indicate
        perfect positive association (model perfectly ranks positives higher),
        values closer to -1 indicate perfect negative association, and 0
        indicates no association.

    Notes
    -----
    This implementation uses the relationship between Somers' D and ROC AUC:

    D = 2 * AUC - 1

    Where AUC is the Area Under the ROC Curve. This provides a reliable
    reference implementation for validation of other methods.

    Raises
    ------
    ValueError
        If only one class is present in labels, returns 0.0 instead of raising.
    """
    try:
        auc = roc_auc_score(labels, scores)
        return 2 * auc - 1
    except ValueError:
        # Handle case where only one class is present
        return 0.0


@numba.jit(nopython=True, nogil=True)
def _vectorized(
    scores_arr: npt.NDArray[np.floating],
    labels_arr: npt.NDArray[np.integer],
    n_pos: int,
    n_neg: int,
) -> float:
    """
    This is the core logic for the vectorized implementation, taken out so it can be
    decorated with numba for performance.

    Parameters
    ----------
    scores_arr : ndarray of float
        Array of predicted scores for each sample.
    labels_arr : ndarray of int
        Array of true binary labels (0 or 1) for each sample.
    n_pos : int
        Number of positive samples (label = 1).
    n_neg : int
        Number of negative samples (label = 0).

    Returns
    -------
    float
        Somers' D statistic in range [-1, 1]. Calculated as:
        (concordant_pairs - discordant_pairs) / total_pairs
    """
    # Extract scores for positive and negative samples
    pos_scores = scores_arr[labels_arr == 1]
    neg_scores = scores_arr[labels_arr == 0]

    # Use broadcasting to compare all pairs efficiently
    concordant = np.sum(pos_scores[:, None] > neg_scores[None, :])
    discordant = np.sum(pos_scores[:, None] < neg_scores[None, :])

    return (concordant - discordant) / (n_pos * n_neg)


def somers_d_vectorized(
    scores: Union[Sequence[float], npt.NDArray[np.floating]],
    labels: Union[Sequence[int], npt.NDArray[np.integer]],
) -> float:
    """
    Optimized vectorized implementation using NumPy broadcasting.

    This implementation uses NumPy's broadcasting capabilities to efficiently
    compute all pairwise comparisons between positive and negative samples
    without explicit loops.

    Parameters
    ----------
    scores : array-like of float
        Predicted scores or probabilities for each sample. Higher scores should
        correspond to higher likelihood of positive class.
    labels : array-like of int
        True binary labels (0 or 1) for each sample. Must have the same length
        as scores.

    Returns
    -------
    float
        Somers' D statistic in range [-1, 1]. Calculated as:
        (concordant_pairs - discordant_pairs) / total_pairs

    Raises
    ------
    ValueError
        If scores and labels have different lengths, or if labels contain
        values other than 0 and 1.
    ZeroDivisionError
        If only one class is present.

    Notes
    -----
    The algorithm separates positive and negative samples, then uses NumPy
    broadcasting to compare all positive scores against all negative scores
    simultaneously:

    - Concordant pairs: positive score > negative score
    - Discordant pairs: positive score < negative score
    - Tied pairs: positive score = negative score (ignored)

    Time complexity: O(n_pos * n_neg)
    Space complexity: O(n_pos * n_neg) for the broadcasting operation

    Examples
    --------
    >>> scores = [0.2, 0.6, 0.4, 0.9, 0.75]
    >>> labels = [0, 1, 0, 1, 1]
    >>> somers_d_vectorized(scores, labels)
    0.5
    """
    try:
        scores_arr, labels_arr, n_pos, n_neg = _validate_and_preprocess_inputs(
            scores, labels
        )
    except ZeroDivisionError:
        return 0.0

    return _vectorized(scores_arr, labels_arr, n_pos, n_neg)


@numba.jit(nopython=True, nogil=True)
def _sorting(
    scores_arr: npt.NDArray[np.floating],
    labels_arr: npt.NDArray[np.integer],
    n_pos: int,
    n_neg: int,
) -> float:
    """
    This is the core logic for the sorting-based implementation, taken out so it can be
    decorated with numba for performance.

    Parameters
    ----------
    scores_arr : ndarray of float
        Array of predicted scores for each sample.
    labels_arr : ndarray of int
        Array of true binary labels (0 or 1) for each sample.
    n_pos : int
        Number of positive samples (label = 1).
    n_neg : int
        Number of negative samples (label = 0).

    Returns
    -------
    float
        Somers' D statistic in range [-1, 1]. Calculated as:
        (concordant_pairs - discordant_pairs) / total_pairs
    """
    # Sort by scores (ascending)
    sorted_indices = np.argsort(scores_arr)
    sorted_scores = scores_arr[sorted_indices]
    sorted_labels = labels_arr[sorted_indices]

    concordant = 0
    discordant = 0

    # For each positive sample, count how many negative samples have lower/higher scores
    for i, (score, label) in enumerate(zip(sorted_scores, sorted_labels)):
        if label == 1:  # Positive sample
            # Count negative samples with lower scores (concordant)
            neg_lower = np.sum((sorted_scores[:i] < score) & (sorted_labels[:i] == 0))
            # Count negative samples with higher scores (discordant)
            neg_higher = np.sum(
                (sorted_scores[i + 1 :] > score) & (sorted_labels[i + 1 :] == 0)
            )

            concordant += neg_lower
            discordant += neg_higher

    return (concordant - discordant) / (n_pos * n_neg)


def somers_d_sorting(
    scores: Union[Sequence[float], npt.NDArray[np.floating]],
    labels: Union[Sequence[int], npt.NDArray[np.integer]],
) -> float:
    """
    Correct sorting-based approach with proper two-directional counting.

    This implementation sorts samples by score and counts concordant/discordant
    pairs by examining the relative positions of positive and negative samples
    in the sorted order.

    Parameters
    ----------
    scores : array-like of float
        Predicted scores or probabilities for each sample. Higher scores should
        correspond to higher likelihood of positive class.
    labels : array-like of int
        True binary labels (0 or 1) for each sample. Must have the same length
        as scores.

    Returns
    -------
    float
        Somers' D statistic in range [-1, 1]. Calculated as:
        (concordant_pairs - discordant_pairs) / total_pairs

    Raises
    ------
    ValueError
        If scores and labels have different lengths, or if labels contain
        values other than 0 and 1.
    ZeroDivisionError
        If only one class is present.

    Notes
    -----
    The algorithm works by:

    1. Sorting all samples by score in ascending order
    2. For each positive sample, counting:
       - Negative samples with lower scores (concordant pairs)
       - Negative samples with higher scores (discordant pairs)

    This approach is more memory-efficient than vectorized methods but has
    higher time complexity due to the nested loops.

    Time complexity: O(n log n + n_pos * n)
    Space complexity: O(n)

    Examples
    --------
    >>> scores = [0.2, 0.6, 0.4, 0.9, 0.75]
    >>> labels = [0, 1, 0, 1, 1]
    >>> somers_d_sorting(scores, labels)
    0.5
    """
    try:
        scores_arr, labels_arr, n_pos, n_neg = _validate_and_preprocess_inputs(
            scores, labels
        )
    except ZeroDivisionError:
        return 0.0

    return _sorting(scores_arr, labels_arr, n_pos, n_neg)


@numba.jit(nopython=True, nogil=True)
def _naive_pairwise(
    scores_arr: npt.NDArray[np.floating],
    labels_arr: npt.NDArray[np.integer],
    n_pos: int,
    n_neg: int,
) -> float:
    """
    This is the core logic for the naive pairwise implementation, taken out so it can be
    decorated with numba for performance.

    Parameters
    ----------
    scores_arr : ndarray of float
        Array of predicted scores for each sample.
    labels_arr : ndarray of int
        Array of true binary labels (0 or 1) for each sample.
    n_pos : int
        Number of positive samples (label = 1).
    n_neg : int
        Number of negative samples (label = 0).

    Returns
    -------
    float
        Somers' D statistic in range [-1, 1]. Calculated as:
        (concordant_pairs - discordant_pairs) / total_pairs
    """
    concordant = 0
    discordant = 0

    # Compare all positive-negative pairs
    for i in range(len(scores_arr)):
        if labels_arr[i] == 1:  # Positive sample
            for j in range(len(scores_arr)):
                if labels_arr[j] == 0:  # Negative sample
                    if scores_arr[i] > scores_arr[j]:
                        concordant += 1
                    elif scores_arr[i] < scores_arr[j]:
                        discordant += 1
                    # Equal scores contribute neither concordant nor discordant

    return (concordant - discordant) / (n_pos * n_neg)


def somers_d_naive_pairwise(
    scores: Union[Sequence[float], npt.NDArray[np.floating]],
    labels: Union[Sequence[int], npt.NDArray[np.integer]],
) -> float:
    """
    Naive O(n^2) implementation for verification (use only on small datasets).

    This implementation explicitly compares every positive sample against every
    negative sample using nested loops. While inefficient for large datasets,
    it provides a clear, easy-to-understand reference for building intuition.

    Parameters
    ----------
    scores : array-like of float
        Predicted scores or probabilities for each sample. Higher scores should
        correspond to higher likelihood of positive class.
    labels : array-like of int
        True binary labels (0 or 1) for each sample. Must have the same length
        as scores.

    Returns
    -------
    float
        Somers' D statistic in range [-1, 1]. Calculated as:
        (concordant_pairs - discordant_pairs) / total_pairs

    Raises
    ------
    ValueError
        If scores and labels have different lengths.
    ZeroDivisionError
        If only one class is present.

    Notes
    -----
    The algorithm uses a straightforward double loop to examine all pairs:

    - For each positive sample, compare against all negative samples
    - Count concordant pairs (positive score > negative score)
    - Count discordant pairs (positive score < negative score)
    - Ignore tied pairs (positive score = negative score)

    **Warning**: This implementation has O(n^2) time complexity and should only
    be used for verification on small datasets (n < 1000).

    Time complexity: O(n^2)
    Space complexity: O(1)

    Examples
    --------
    >>> scores = [0.2, 0.6, 0.4, 0.9, 0.75]
    >>> labels = [0, 1, 0, 1, 1]
    >>> somers_d_naive_pairwise(scores, labels)
    0.5
    """
    try:
        scores_arr, labels_arr, n_pos, n_neg = _validate_and_preprocess_inputs(
            scores, labels
        )
    except ZeroDivisionError:
        return 0.0

    # Use numba-optimized core logic for performance
    return _naive_pairwise(scores_arr, labels_arr, n_pos, n_neg)


@numba.jit(nopython=True, nogil=True)
def _loop(
    scores_arr: npt.NDArray[np.floating],
    labels_arr: npt.NDArray[np.integer],
    n_pos: int,
    n_neg: int,
) -> float:
    """
    This is the core logic for the loop-based implementation, taken out so it can be
    decorated with numba for performance.

    Parameters
    ----------
    scores_arr : ndarray of float
        Array of predicted scores for each sample.
    labels_arr : ndarray of int
        Array of true binary labels (0 or 1) for each sample.
    n_pos : int
        Number of positive samples (label = 1).
    n_neg : int
        Number of negative samples (label = 0).

    Returns
    -------
    float
        Somers' D statistic in range [-1, 1]. Calculated as:
        (concordant_pairs - discordant_pairs) / total_pairs
    """
    # Separate positive and negative samples
    pos_indices = np.where(labels_arr == 1)[0]
    neg_indices = np.where(labels_arr == 0)[0]

    concordant = 0
    discordant = 0

    # Compare only positive vs negative pairs
    for pos_idx in pos_indices:
        pos_score = scores_arr[pos_idx]
        for neg_idx in neg_indices:
            neg_score = scores_arr[neg_idx]
            if pos_score > neg_score:
                concordant += 1
            elif pos_score < neg_score:
                discordant += 1

    return (concordant - discordant) / (n_pos * n_neg)


def somers_d_loop(
    scores: Union[Sequence[float], npt.NDArray[np.floating]],
    labels: Union[Sequence[int], npt.NDArray[np.integer]],
) -> float:
    """
    Loop-based implementation avoiding redundant computations.

    This implementation pre-separates positive and negative samples to avoid
    redundant label checks in the inner loop, making it more efficient than
    the naive pairwise approach while maintaining good memory efficiency.

    Parameters
    ----------
    scores : array-like of float
        Predicted scores or probabilities for each sample. Higher scores should
        correspond to higher likelihood of positive class.
    labels : array-like of int
        True binary labels (0 or 1) for each sample. Must have the same length
        as scores.

    Returns
    -------
    float
        Somers' D statistic in range [-1, 1]. Calculated as:
        (concordant_pairs - discordant_pairs) / total_pairs

    Raises
    ------
    ValueError
        If scores and labels have different lengths, or if labels contain
        values other than 0 and 1.
    ZeroDivisionError
        If only one class is present.

    Notes
    -----
    The algorithm optimizes the naive pairwise approach by:

    1. Pre-identifying indices of positive and negative samples
    2. Only comparing positive vs negative pairs (avoiding positive-positive and negative-negative comparisons)
    3. Using direct array indexing instead of repeated label checks

    This provides a good balance between simplicity and performance, especially
    for memory-constrained environments where the vectorized approach's
    memory usage might be prohibitive.

    Time complexity: O(n_pos * n_neg)
    Space complexity: O(n_pos + n_neg)

    Examples
    --------
    >>> scores = [0.2, 0.6, 0.4, 0.9, 0.75]
    >>> labels = [0, 1, 0, 1, 1]
    >>> somers_d_loop(scores, labels)
    0.5
    """
    try:
        scores_arr, labels_arr, n_pos, n_neg = _validate_and_preprocess_inputs(
            scores, labels
        )
    except ZeroDivisionError:
        return 0.0

    return _loop(scores_arr, labels_arr, n_pos, n_neg)


def generate_test_cases() -> List[Tuple[List[float], List[int], str]]:
    """
    Generate diverse test cases for comprehensive validation.

    Creates a comprehensive suite of test cases covering various scenarios
    including edge cases, different data distributions, and challenging
    conditions to thoroughly validate Somers' D implementations.

    Returns
    -------
    List[Tuple[List[float], List[int], str]]
        List of test cases, where each tuple contains:
        - scores: List of predicted scores/probabilities
        - labels: List of true binary labels (0 or 1)
        - description: String describing the test case

    Notes
    -----
    The test cases include:

    - Perfect separation scenarios
    - Random data with various sizes and distributions
    - Imbalanced class scenarios
    - Edge cases with tied scores
    - Single-class scenarios
    - Various data patterns (alternating, etc.)

    Examples
    --------
    >>> test_cases = generate_test_cases()
    >>> len(test_cases) > 10
    True
    >>> all(len(case) == 3 for case in test_cases)
    True
    """
    test_cases = [
        # Simple example
        ([0.2, 0.6, 0.4, 0.9, 0.75], [0, 1, 0, 1, 1], "Simple example"),
        # Perfect separation
        ([0.1, 0.2, 0.3, 0.7, 0.8, 0.9], [0, 0, 0, 1, 1, 1], "Perfect separation"),
        # Perfect anti-separation
        ([0.7, 0.8, 0.9, 0.1, 0.2, 0.3], [0, 0, 0, 1, 1, 1], "Perfect anti-separation"),
        # No discrimination (all same scores)
        ([0.5, 0.5, 0.5, 0.5], [0, 1, 0, 1], "No discrimination - tied scores"),
        # Random scores
        (
            rng.random(20).tolist(),
            rng.choice([0, 1], 20).tolist(),
            "Random 20 samples",
        ),
        # Imbalanced classes (mostly negative)
        (
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            [0, 0, 0, 0, 0, 0, 0, 1, 1],
            "Imbalanced - mostly negative",
        ),
        # Imbalanced classes (mostly positive)
        (
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            [1, 1, 1, 1, 1, 1, 1, 0, 0],
            "Imbalanced - mostly positive",
        ),
        # With ties in scores
        ([0.1, 0.1, 0.5, 0.5, 0.9, 0.9], [0, 1, 0, 1, 0, 1], "Tied scores"),
        # Single positive
        ([0.1, 0.2, 0.3, 0.9], [0, 0, 0, 1], "Single positive"),
        # Single negative
        ([0.1, 0.7, 0.8, 0.9], [0, 1, 1, 1], "Single negative"),
        # Larger random dataset
        (
            rng.random(100).tolist(),
            rng.choice([0, 1], 100).tolist(),
            "Random 100 samples",
        ),
        # Near-perfect but not quite
        (
            [0.1, 0.15, 0.25, 0.7, 0.85, 0.95],
            [0, 0, 0, 1, 1, 1],
            "Near-perfect separation",
        ),
        # Alternating pattern
        ([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], [0, 1, 0, 1, 0, 1], "Alternating pattern"),
        # Edge case: identical scores, different labels
        ([0.5, 0.5, 0.5], [0, 1, 0], "Identical scores"),
        # Large tied groups
        ([0.1, 0.1, 0.1, 0.9, 0.9, 0.9], [0, 0, 1, 1, 1, 0], "Large tied groups"),
    ]

    return test_cases


def validate_all_implementations() -> bool:
    """
    Test all implementations against the reference (AUC-based) method.

    Runs comprehensive validation of all Somers' D implementations by comparing
    their results against the reference AUC-based implementation across diverse
    test cases. This ensures correctness and identifies any bugs or edge cases.

    Returns
    -------
    bool
        True if all implementations pass validation (results match reference
        within tolerance), False otherwise.

    Notes
    -----
    The validation process:

    1. Generates diverse test cases using `generate_test_cases()`
    2. Runs each implementation on every test case
    3. Compares results against the reference implementation
    4. Reports discrepancies and overall pass/fail status

    Uses a tolerance of 1e-10 for floating-point comparisons to account for
    numerical precision differences between implementations.

    Test cases that contain only one class are skipped as Somers' D is
    undefined in such scenarios.

    Examples
    --------
    >>> validation_passed = validate_all_implementations()
    >>> isinstance(validation_passed, bool)
    True
    """
    implementations = [
        ("Reference (AUC)", somers_d_auc_reference),
        ("Vectorized", somers_d_vectorized),
        ("Sorting", somers_d_sorting),
        ("Loop", somers_d_loop),
        ("Naive Pairwise", somers_d_naive_pairwise),
    ]

    test_cases = generate_test_cases()
    tolerance = 1e-10
    all_passed = True

    for i, (scores, labels, description) in enumerate(test_cases):
        print(f"Test {i + 1:2d}: {description}")

        # Skip edge cases for certain implementations
        n_pos = sum(labels)
        n_neg = len(labels) - n_pos

        if n_pos == 0 or n_neg == 0:
            print("  -> Skipped (only one class present)")
            continue

        results = []
        for name, func in implementations:
            try:
                result = func(scores, labels)
                results.append((name, result))
            except Exception as error:
                results.append((name, -999.0))
                all_passed = False

        # Check if all results match the reference
        reference_result = results[0][1] if isinstance(results[0][1], float) else None

        print(
            f"  Reference result: {reference_result:.8f}"
            if reference_result is not None
            else "  Reference failed"
        )

        for name, result in results[1:]:
            diff = abs(result - reference_result)
            status = "correct" if diff < tolerance else "incorrect"
            print(f"  {name:15}: {result:.8f} (diff: {diff:.2e}) {status}")
            if diff >= tolerance:
                all_passed = False

        print()

    print(f"Overall validation: {'✓ PASSED' if all_passed else '✗ FAILED'}")
    return all_passed


def benchmark_implementations() -> None:
    """
    Benchmark all implementations across different dataset sizes.

    Conducts comprehensive performance benchmarking of Somers' D implementations
    across various dataset sizes to analyze their scalability and identify the
    most efficient approaches for different use cases.

    Notes
    -----
    The benchmarking process:

    1. Tests implementations on datasets of sizes: 100, 500, 1000, 5000, 10000
    2. Runs multiple trials (default: 10) to get reliable timing statistics
    3. Reports average execution time and standard deviation
    4. Excludes naive pairwise implementation for large datasets due to O(n²) complexity
    5. Includes special testing for naive pairwise on small datasets only

    All timing measurements use `time.perf_counter()` for high precision.
    Random data is generated with fixed seed (42) for reproducibility.

    The output includes:
    - Average execution time in milliseconds
    - Standard deviation of execution times
    - Average result value for verification

    Examples
    --------
    >>> benchmark_implementations()  # doctest: +SKIP
    === Performance Benchmark ===
    ...
    """
    print("=== Performance Benchmark ===\n")

    implementations = [
        ("Reference (AUC)", somers_d_auc_reference),
        ("Vectorized", somers_d_vectorized),
        ("Sorting", somers_d_sorting),
        ("Loop", somers_d_loop),
        ("Naive Pairwise", somers_d_naive_pairwise),
    ]

    dataset_sizes = [100, 500, 1000, 5000, 10000]
    n_trials = 10

    print(
        f"{'Size':<8} {'Method':<15} {'Time (ms)':<12} {'Std (ms)':<12} {'Result':<12}"
    )
    print("-" * 65)

    for size in dataset_sizes:
        print(f"\nDataset size: {size}")

        # Generate test data
        scores = rng.random(size)
        labels = rng.choice([0, 1], size)

        for name, func in implementations:
            times = []
            results = []

            for trial in range(n_trials):
                start_time = time.perf_counter()
                result = func(scores, labels)
                end_time = time.perf_counter()

                times.append((end_time - start_time) * 1000)  # Convert to ms
                results.append(result)

            avg_time = np.mean(times)
            std_time = np.std(times)
            avg_result = np.mean(results)

            print(
                f"{'':8} {name:<15} {avg_time:8.2f}     {std_time:8.2f}     {avg_result:8.4f}"
            )

    print("\n" + "=" * 65)


def complexity_analysis() -> None:
    """
    Analyze time complexity empirically.

    Conducts empirical time complexity analysis by measuring execution times
    across different dataset sizes and estimating the apparent algorithmic
    complexity based on timing ratios.

    Notes
    -----
    The analysis process:

    1. Tests each implementation on datasets of increasing size
    2. Measures average execution time across multiple runs
    3. Calculates timing ratios between consecutive dataset sizes
    4. Estimates apparent complexity using logarithmic scaling

    The apparent complexity is calculated as:

    complexity = (log(time_ratio) / log(size_ratio))

    This provides insight into how each implementation scales with data size,
    helping to verify theoretical complexity expectations and identify the
    most scalable approaches.

    Expected complexities:
    - Vectorized: O(n_pos * n_neg) ≈ O(n^2) worst case, O(n) best case
    - Sorting: O(n log n + n_pos * n)
    - Optimized Loop: O(n_pos * n_neg) ≈ O(n^2) worst case
    """
    print("=== Time Complexity Analysis ===\n")

    implementations = [
        ("Vectorized", somers_d_vectorized),
        ("Sorting", somers_d_sorting),
        ("Optimized Loop", somers_d_loop),
    ]

    sizes = [100, 200, 500, 1000, 2000, 5000]

    for name, func in implementations:
        print(f"{name} Implementation:")
        times = []

        for size in sizes:
            np.random.seed(42)
            scores = np.random.random(size)
            labels = np.random.choice([0, 1], size)

            # Time multiple runs
            run_times = []
            for _ in range(5):
                start = time.perf_counter()
                func(scores, labels)
                end = time.perf_counter()
                run_times.append(end - start)

            avg_time = np.mean(run_times)
            times.append(avg_time)

            # Estimate complexity
            if len(times) > 1:
                ratio = times[-1] / times[-2]
                size_ratio = sizes[len(times) - 1] / sizes[len(times) - 2]
                apparent_complexity = np.log(ratio) / np.log(size_ratio)
                print(
                    f"  Size {size:4d}: {avg_time * 1000:6.2f}ms (complexity ≈ O(n^{apparent_complexity:.1f}))"
                )
            else:
                print(f"  Size {size:4d}: {avg_time * 1000:6.2f}ms")

    return None


def main() -> int:
    print("Comprehensive Somers' D Implementation Analysis")
    print("=" * 55)
    print()

    validation_passed = validate_all_implementations()

    if not validation_passed:
        print("Validation failed! Stopping analysis")
        return 1

    print("\n")

    benchmark_implementations()

    print("\n")

    complexity_analysis()

    return 0


if __name__ == "__main__":
    main()
