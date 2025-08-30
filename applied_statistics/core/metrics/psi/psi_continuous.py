from collections.abc import Sequence
from typing import Dict, Final, List, Literal, Optional, Tuple, TypeAlias, Union

import matplotlib.pyplot as plt
import numba
import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from pydantic import BaseModel, ConfigDict, Field

FloatArrayLike: TypeAlias = npt.NDArray[np.floating]
IntArrayLike: TypeAlias = npt.NDArray[np.integer]
BinRule = Literal["auto", "fd", "doane", "scott", "stone", "rice", "sturges", "sqrt"]
BinsParam = Union[int, Sequence[float], BinRule]

ALLOWED_BIN_RULES: Final[Tuple[BinRule, ...]] = (
    "auto",
    "fd",
    "doane",
    "scott",
    "stone",
    "rice",
    "sturges",
    "sqrt",
)


class PSIResult(BaseModel):
    """
    Immutable result holder for PSI computations.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    psi: float = Field(..., description="PSI value")
    n_benchmark: int = Field(
        ..., description="Total number of data points in benchmark"
    )
    n_monitoring: int = Field(
        ..., description="Total number of data points in monitoring"
    )
    bin_rule: Optional[BinRule] = Field(
        default=None,
        description="Histogram rule if a string rule was used; None if int or explicit edges",
    )
    open_ended: bool = Field(
        ..., description="Whether outer bins were extended to [-inf, inf]"
    )
    eps: float = Field(..., description="Smoothing floor used for probabilities")


@numba.njit(parallel=True, fastmath=True)
def _psi_numba(
    benchmark_array: npt.NDArray[np.floating],
    monitoring_array: npt.NDArray[np.floating],
    bins: npt.NDArray[np.floating],
    eps: float,
) -> float:
    """
    Compute the Population Stability Index (PSI) between two univariate distributions.

    Parameters
    ----------
    benchmark_array: npt.NDArray[np.floating]
        Benchmark distribution (reference).
    monitoring_array: npt.NDArray[np.floating]
        Monitoring distribution.
    bins: npt.NDArray[np.floating]
        Bin edges computed from benchmark array, `e_0 < e_1 < ... < e_k` (length `k + 1`).
    eps: float
        Small positive floor to avoid log/zero issues.

    Returns
    -------
    float
        The computed PSI value.
    """
    # Histogram counts per bin for i = 0, ..., k
    # Benchmark b_i = |{x in benchmark_array : e_i <= x < e_{i +  1}}|
    # Monitoring m_i = |{x in monitoring_array : e_i <= x < e_{i +  1}}|
    benchmark_hist, _ = np.histogram(benchmark_array, bins=bins)
    monitoring_hist, _ = np.histogram(monitoring_array, bins=bins)

    # Totals (denominators) n_b = Σ_i b_i, n_m = Σ_i m_i, used to form empirical pmfs
    benchmark_sum: np.floating = np.sum(benchmark_hist)
    monitoring_sum: np.floating = np.sum(monitoring_hist)

    # Both histograms must contain at least one observation; otherwise pmfs are undefined
    if benchmark_sum == 0.0 or monitoring_sum == 0.0:
        raise ValueError(
            "Invalid histogram (zero counts), which means no data was found in one or both distributions based on the provided bins"
        )

    # Empirical proportions (pmfs on the shared partition):
    # Where q_i = b_i / n_b   (benchmark pmf)
    # Where p_i = m_i / n_m   (monitoring pmf)
    # Note: Σ_i q_i = Σ_i p_i = 1 in exact arithmetic before smoothing
    benchmark_prop: FloatArrayLike = benchmark_hist / benchmark_sum
    monitoring_prop: FloatArrayLike = monitoring_hist / monitoring_sum

    # Apply ε-smoothing to enforce strict positivity (p_i, q_i > 0) required by log(p_i/q_i)
    # This creates interim vectors: q'_i = max(q_i, ε) & p'_i = max(p_i, ε)
    # After this step, Σ_i q'_i ≥ 1 and Σ_i p'_i ≥ 1 in general
    benchmark_prop_processed: FloatArrayLike = np.where(
        benchmark_prop == 0.0, eps, benchmark_prop
    )
    monitoring_prop_processed: FloatArrayLike = np.where(
        monitoring_prop == 0.0, eps, monitoring_prop
    )

    # Renormalize to probability vectors: q''_i = q'_i / Σ_j q'_j,   p''_i = p'_i / Σ_j p'_j
    # So Σ_i q''_i = Σ_i p''_i = 1 exactly (up to floating error)
    benchmark_norm: np.floating = np.sum(benchmark_prop_processed)
    monitoring_norm: np.floating = np.sum(monitoring_prop_processed)
    benchmark_prop_processed = benchmark_prop_processed / benchmark_norm
    monitoring_prop_processed = monitoring_prop_processed / monitoring_norm

    # Jeffrey divergence: J(p'', q'') = Σ_i (p''_i - q''_i) * log(p''_i / q''_i)
    psi_value: float = float(
        np.sum(
            (monitoring_prop_processed - benchmark_prop_processed)
            * np.log(monitoring_prop_processed / benchmark_prop_processed)
        )
    )

    return psi_value


def psi(
    benchmark: Union[Sequence[float], FloatArrayLike],
    monitoring: Union[Sequence[float], FloatArrayLike],
    *,
    bins: BinsParam = "doane",
    data_range: Optional[Tuple[float, float]] = None,
    open_ended: bool = True,
    eps: float = 1e-6,
) -> PSIResult:
    """
    Population Stability Index on a fixed partition derived from `benchmark` only.

    Parameters
    ----------
    benchmark: Union[Sequence[float], NDArray[np.floating]]
        Benchmark distribution (reference).
    monitoring: Union[Sequence[float], NDArray[np.floating]]
        Monitoring distribution.
    bins
        - int: number of equal-width bins (over `benchmark` or within `data_range`).
        - Sequence[float]: explicit bin edges (including rightmost edge).
        - One of {'auto','fd','doane','scott','stone','rice','sturges','sqrt'}: estimator for `numpy.histogram_bin_edges`.
    data_range
        Forwarded to `numpy.histogram_bin_edges(range=...)`; defaults to `(min(benchmark), max(benchmark))`.
    open_ended
        If True, extend outer edges to [-inf, inf] to capture tail drift.
    eps
        Small positive floor to avoid log/zero issues.

    Returns
    -------
    PSIResult
        Holds the PSI value and metadata.
    """
    benchmark_array: FloatArrayLike = np.asarray(benchmark, dtype=float).ravel()
    monitoring_array: FloatArrayLike = np.asarray(monitoring, dtype=float).ravel()

    if benchmark_array.size == 0 or monitoring_array.size == 0:
        raise ValueError("Arrays `benchmark` and `monitoring` must be non-empty")

    edges: FloatArrayLike
    rule_used: Optional[BinRule] = None

    if isinstance(bins, (list, tuple, np.ndarray)):
        edges = np.asarray(bins, dtype=float)
    elif isinstance(bins, int):
        edges = np.histogram_bin_edges(benchmark_array, bins=bins, range=data_range)
    elif isinstance(bins, str):
        rule: BinRule = bins
        if rule not in ALLOWED_BIN_RULES:
            raise ValueError(
                f"Bin rule '{rule}' is invalid; allowed: {ALLOWED_BIN_RULES}"
            )
        edges = np.histogram_bin_edges(benchmark_array, bins=rule, range=data_range)
        rule_used = rule

    if edges.ndim != 1 or edges.size < 2:
        raise ValueError(
            "Invalid bin edges produced; must be 1-dimensional and have at least 2 elements"
        )
    if np.any(np.diff(edges) <= 0):
        raise ValueError("Bin edges must be strictly increasing")

    if open_ended:
        edges[0] = float(-np.inf)
        edges[-1] = float(np.inf)

    psi_value: float = _psi_numba(
        benchmark_array=benchmark_array,
        monitoring_array=monitoring_array,
        bins=edges,
        eps=eps,
    )

    return PSIResult(
        psi=psi_value,
        n_benchmark=int(benchmark_array.size),
        n_monitoring=int(monitoring_array.size),
        bin_rule=rule_used,
        open_ended=open_ended,
        eps=float(eps),
    )


def main() -> int:
    rng: np.random.Generator = np.random.default_rng(seed=12)
    n_samples: int = 20_000
    shifts: List[float] = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5]
    rule: BinRule = "doane"
    open_ended_flag: bool = True
    eps_val: float = 1e-6

    psi_vals: List[float] = []
    data_sets: Dict[float, Tuple[FloatArrayLike, FloatArrayLike]] = {}

    for shift in shifts:
        benchmark_sample: FloatArrayLike = rng.normal(
            loc=0.0, scale=1.0, size=n_samples
        ).astype(float)
        monitoring_sample: FloatArrayLike = rng.normal(
            loc=shift, scale=1.0, size=n_samples
        ).astype(float)
        data_sets[shift] = (benchmark_sample, monitoring_sample)

        res: PSIResult = psi(
            benchmark_sample,
            monitoring_sample,
            bins=rule,
            open_ended=open_ended_flag,
            eps=eps_val,
        )
        psi_vals.append(res.psi)

    is_monotone: bool = all(
        psi_vals[i + 1] >= psi_vals[i] for i in range(len(psi_vals) - 1)
    )

    if not is_monotone:
        for i in range(len(shifts)):
            print(f"shift = {shifts[i]:.2f}, PSI = {psi_vals[i]:.6f}")
        raise AssertionError("PSI is not approximately monotone increasing with shift")

    n_plots: int = len(shifts)
    n_cols: int = 3
    n_rows: int = (n_plots + n_cols - 1) // n_cols

    fig: Figure
    fig, axes_arr = plt.subplots(n_rows, n_cols, figsize=(4.8 * n_cols, 3.6 * n_rows))
    axes: List[Axes] = (
        [axes_arr] if isinstance(axes_arr, Axes) else list(axes_arr.ravel().tolist())
    )

    for idx, shift in enumerate(shifts):
        ax: Axes = axes[idx]

        ax.hist(
            data_sets[shift][0], bins=50, density=True, alpha=0.5, label="benchmark"
        )
        ax.hist(
            data_sets[shift][1], bins=50, density=True, alpha=0.5, label="monitoring"
        )
        ax.set_title(f"shift = {shift:.2f} | PSI ~ {psi_vals[idx]:.4f}")
        ax.set_xlabel("value")
        ax.set_ylabel("density")
        if idx == 0:
            ax.legend(loc="best")

    # Hide any unused subplots
    for j in range(n_plots, len(axes)):
        axes[j].axis("off")

    fig.tight_layout()
    plt.show()

    return 0


if __name__ == "__main__":
    main()
