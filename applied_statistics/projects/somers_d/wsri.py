import logging
from sys import stdout
from typing import Protocol, Sequence, Tuple, TypeAlias, Union, runtime_checkable

import numba
import numpy as np
import numpy.typing as npt
import polars as pl
from sklearn.metrics import roc_auc_score

logger = logging.getLogger("wSRI Metric:")
log_format = "%(asctime)s %(levelname)s %(name)s: %(message)s"
formatter = logging.Formatter(log_format)
handler = logging.StreamHandler(stdout)
handler.setFormatter(formatter)
logger.addHandler(handler)

FloatArrayLike: TypeAlias = Union[Sequence[float], npt.NDArray[np.floating]]
IntegerArrayLike: TypeAlias = Union[Sequence[int], npt.NDArray[np.integer]]


@runtime_checkable
class SomersDCallback(Protocol):
    def __call__(
        self,
        *,  # Keyword-only arguments
        scores: FloatArrayLike,
        labels: IntegerArrayLike,
        weights: FloatArrayLike,
    ) -> float: ...


def generate_score_bins(
    data: Union[pl.DataFrame, pl.LazyFrame],
    score_column: str,
    quantiles: Union[Sequence[float], int],
) -> npt.NDArray[np.float64]:
    """
    Generate score bins for the given `score_column` data using quantiles.

    Parameters
    ----------
    data : pl.DataFrame
        The input data frame containing the score column.
    score_column : str
        The name of the score column to be binned.
    quantiles : Union[Sequence[float], int]
        Either a list of quantile probabilities between 0 and 1 or a positive integer determining the number of bins with uniform probability.

    Returns
    -------
    npt.NDArray[np.float64]
        The generated score bins as an array.
    """
    data_lazy: pl.LazyFrame = data.lazy()

    return (
        data_lazy.select(pl.col(score_column).drop_nulls())
        .select(
            pl.col(score_column)
            .qcut(quantiles=quantiles, left_closed=False, include_breaks=True)
            .alias("bins")
        )
        .unnest("bins")
        .select(pl.col("breakpoint").cast(pl.Float64))
        .filter(pl.col("breakpoint").is_finite())
        .unique()
        .sort(by="breakpoint", descending=False)
        .collect(engine="streaming")
        .get_column(name="breakpoint")
        .to_numpy()
    )


def preprocess_data(
    benchmark: Union[pl.DataFrame, pl.LazyFrame],
    monitoring: Union[pl.DataFrame, pl.LazyFrame],
    score_column: str,
    score_bins: FloatArrayLike,
) -> Tuple[pl.LazyFrame, pl.LazyFrame]:
    """
    Preprocess the benchmark and monitoring data:

    1. Drop any rows with missing values in the score column and cast to float64.
    2. Bin the score column using the provided score bins.
    3. Count the number of observations in each bin for both benchmark and monitoring data.
    4. Perform sanity checks to ensure that:

        - All monitoring bins should be present in benchmark, so (monitoring counts / benchmark counts = 0) != inf weights
        - Some overlap between monitoring and benchmark bins, or else all weights would be (monitoring counts = 0 / benchmark counts) = 0

    5. Compute the weights for benchmark bins using (monitoring counts / benchmark counts).
    6. For any benchmark bins not present in monitoring, their weights will be set to zero.
    7. Monitoring observations will have weights as 1.

    Parameters
    ----------
    benchmark : Union[pl.DataFrame, pl.LazyFrame]
        The benchmark data.
    monitoring : Union[pl.DataFrame, pl.LazyFrame]
        The monitoring data.
    score_column : str
        The name of the score column.
    score_bins : Union[Sequence[float], npt.NDArray[np.float64]]
        The score bins to use for binning; passed to `pl.cut(breaks=score_bins)`.

    Returns
    -------
    Tuple[pl.LazyFrame, pl.LazyFrame]
        The preprocessed benchmark and monitoring data.
    """
    benchmark_lazy: pl.LazyFrame = benchmark.lazy()
    monitoring_lazy: pl.LazyFrame = monitoring.lazy()

    # Cast score column dtype, sort, and cut continuous scores into (cat -> int via to_physical) bins
    benchmark_intermediate: pl.LazyFrame = benchmark_lazy.drop_nans(
        subset=[score_column]
    ).with_columns(
        [
            pl.col(score_column).cast(pl.Float64),
            pl.col(score_column)
            .cast(pl.Float64)
            .cut(breaks=score_bins)
            .to_physical()
            .alias("bins"),  # type: ignore[arg-type]
        ]
    )
    monitoring_intermediate: pl.LazyFrame = monitoring_lazy.drop_nans(
        subset=[score_column]
    ).with_columns(
        [
            pl.col(score_column).cast(pl.Float64),
            pl.col(score_column)
            .cast(pl.Float64)
            .cut(breaks=score_bins)
            .to_physical()
            .alias("bins"),  # type: ignore[arg-type]
        ]
    )

    # Create bins counts given the score bins
    benchmark_bin_counts: pl.LazyFrame = (
        benchmark_intermediate.group_by(pl.col("bins"))
        .len()
        .rename({"len": "benchmark_count"})
    )
    monitoring_bin_counts: pl.LazyFrame = (
        monitoring_intermediate.group_by(pl.col("bins"))
        .len()
        .rename({"len": "monitoring_count"})
    )

    base_message: str = f"Weighting benchmark data failed, "

    monitoring_minus_benchmark: pl.LazyFrame = monitoring_bin_counts.join(
        other=benchmark_bin_counts, on="bins", how="anti"
    )
    if monitoring_minus_benchmark.select(pl.len()).collect().item() > 0:
        raise ValueError(
            f"{base_message}some monitoring bins not present in benchmark; the weighting formula would result in (monitoring count / 0) = inf"
        )

    overlap: pl.LazyFrame = benchmark_bin_counts.join(
        other=monitoring_bin_counts, on="bins", how="inner"
    )
    # There must be some overlap between monitoring and benchmark bins, or all benchmark weights would be zeros
    if overlap.select(pl.len()).collect().item() == 0:
        raise ValueError(
            f"{base_message}no overlap between monitoring and benchmark bins, and all benchmark weights would be zeros"
        )

    # Acceptable for some benchmark bins to not be in monitoring, but warn that those counts will be set to zeros in monitoring bin counts data
    benchmark_minus_monitoring: pl.LazyFrame = benchmark_bin_counts.join(
        other=monitoring_bin_counts, on="bins", how="anti"
    )
    if benchmark_minus_monitoring.select(pl.len()).collect().item() > 0:
        logger.warning(
            f"Some benchmark bins not present in monitoring; benchmark observations with scores in those bins will have zero weights"
        )

    # Compute weights for benchmark data
    benchmark_weights: pl.LazyFrame = (
        benchmark_bin_counts.join(
            other=monitoring_bin_counts,
            on="bins",
            how="left",
        )
        .with_columns(pl.col("monitoring_count").fill_null(value=0).cast(pl.UInt64))
        .with_columns(
            (pl.col("monitoring_count") / pl.col("benchmark_count"))
            .cast(pl.Float64)
            .alias("weights")
        )
        .sort(pl.col("bins"))
    )

    benchmark_preprocessed: pl.LazyFrame = benchmark_intermediate.join(
        other=benchmark_weights,
        on="bins",
        how="left",
    ).drop(["bins", "benchmark_count", "monitoring_count"])

    monitoring_preprocessed: pl.LazyFrame = monitoring_intermediate.with_columns(
        pl.lit(1, dtype=pl.Float64).alias("weights")
    ).drop("bins")

    return benchmark_preprocessed, monitoring_preprocessed


@numba.njit(parallel=True, fastmath=True)
def somers_d_two_pointers(
    scores: FloatArrayLike,
    labels: IntegerArrayLike,
    weights: FloatArrayLike,
) -> float:
    """
    Compute weighted Somers' D using the two-pointer technique.

    Parameters
    ----------
    scores : Union[Sequence[float], npt.NDArray[np.floating]]
        The model scores.
    labels : Union[Sequence[int], npt.NDArray[np.integer]]
        The ground-truth binary labels (0 or 1).
    weights : Union[Sequence[float], npt.NDArray[np.floating]]
        The weights for each observation.

    Returns
    -------
    float
        The weighted Somers' D statistic.
    """
    n_total: int = len(scores)

    # The argsort sorts in ascending order, which is what we need for the two-pointer technique
    sorted_indices: npt.NDArray[np.integer] = np.argsort(scores)
    scores: npt.NDArray[np.floating] = np.ascontiguousarray(scores[sorted_indices])
    labels: npt.NDArray[np.integer] = np.ascontiguousarray(labels[sorted_indices])
    weights: npt.NDArray[np.floating] = np.ascontiguousarray(weights[sorted_indices])

    # Sum of weights for all positive and negative samples
    n_pos: float = np.sum(weights[labels == 1])
    n_neg: float = np.sum(weights[labels == 0])

    if n_pos == 0 or n_neg == 0:
        return 0.0

    # This accumulates "discordant mass" -> tracks how many positives ranked lower than a negative
    discordant_contrib: float = 0.0
    # Initialized as total positives (i.e., assume all positives pairs are concordant), then reduced as we find discordant pairs
    concordant_contrib: float = n_pos
    numerator: float = 0.0
    discordant_index: int = 0
    concordant_index: int = 0

    for i in range(n_total):
        # Treat each negative sample as an "anchor" to compare against positives
        if labels[i] == 0:
            current_false_score: float = scores[i]

            # Move the discordant pointer up until scores are strictly less
            # Every time we encounter a positive with a score < current negative
            # we add its weight to discordant_contrib because this is a discordant pair
            # Example: negative = 0.7, positive = 0.6 -> model misranks this pair
            while (
                discordant_index < n_total
                and scores[discordant_index] < current_false_score
            ):
                if labels[discordant_index] == 1:
                    discordant_contrib += weights[discordant_index]
                discordant_index += 1

            # Move the concordant pointer up until scores are <= current negative
            # Every time we encounter a positive with a score <= current negative
            # we subtract its weight from concordant_contrib because it is no longer concordant against the current negative
            # Example: negative = 0.7, positive = 0.7 -> ties count against concordant mass
            while (
                concordant_index < n_total
                and scores[concordant_index] <= current_false_score
            ):
                if labels[concordant_index] == 1:
                    concordant_contrib -= weights[concordant_index]
                concordant_index += 1

            # For the current negative, its weighted contribution is (positives above it - positives below it)
            # Multiplying by weight[i] scales the effect by the current negative's weight
            numerator += weights[i] * (concordant_contrib - discordant_contrib)

    # Denominator is total number of positive-negative weighted pairs
    return numerator / (n_neg * n_pos)


def somers_d_roc_auc(
    scores: FloatArrayLike,
    labels: IntegerArrayLike,
    weights: FloatArrayLike,
) -> float:
    """
    Compute weighted Somers' D using the relationship between
    Somers' D and the ROC AUC.

    Parameters
    ----------
    scores : Union[Sequence[float], npt.NDArray[np.floating]]
        The model scores.
    labels : Union[Sequence[int], npt.NDArray[np.integer]]
        The ground-truth binary labels (0 or 1).
    weights : Union[Sequence[float], npt.NDArray[np.floating]]
        The weights for each observation.

    Returns
    -------
    float
        The weighted Somers' D statistic.
    """
    roc_auc: float = roc_auc_score(y_true=labels, y_score=scores, sample_weight=weights)
    return 2.0 * roc_auc - 1.0


def wsri(
    benchmark: pl.LazyFrame,
    monitoring: pl.LazyFrame,
    score_column: str,
    label_column: str,
    quantiles: Union[Sequence[float], int],
    callback: SomersDCallback,
) -> float:
    """
    Compute the weighted Somers' D metric between benchmark and monitoring data.

    Parameters
    ----------
    benchmark : pl.LazyFrame
        The benchmark data.
    monitoring : pl.LazyFrame
        The monitoring data.
    score_column : str
        The name of the score column.
    label_column : str
        The name of the label column.
    quantiles : Union[Sequence[float], int]
        Either a list of quantile probabilities between 0 and 1 or a positive integer determining the number of bins with uniform probability.
    callback : SomersDCallback
        A callback function to compute Somers' D given the scores, labels, and weights.

    Returns
    -------
    float
        The weighted Somers' D metric.
    """
    score_bins: npt.NDArray[np.float64] = generate_score_bins(
        data=benchmark,
        score_column=score_column,
        quantiles=quantiles,
    )

    benchmark_preprocessed, monitoring_preprocessed = preprocess_data(
        benchmark=benchmark,
        monitoring=monitoring,
        score_column=score_column,
        score_bins=score_bins,
    )

    # Call collect to compute results
    benchmark_collected: pl.DataFrame = benchmark_preprocessed.collect(
        engine="streaming"
    )
    monitoring_collected: pl.DataFrame = monitoring_preprocessed.collect(
        engine="streaming"
    )

    benchmark_somers_d: float = callback(
        scores=benchmark_collected[score_column].to_numpy(),
        labels=benchmark_collected[label_column].to_numpy(),
        weights=benchmark_collected["weights"].to_numpy(),
    )
    monitoring_somers_d: float = callback(
        scores=monitoring_collected[score_column].to_numpy(),
        labels=monitoring_collected[label_column].to_numpy(),
        weights=monitoring_collected["weights"].to_numpy(),
    )

    try:
        return monitoring_somers_d / benchmark_somers_d
    except ZeroDivisionError:
        return np.nan
