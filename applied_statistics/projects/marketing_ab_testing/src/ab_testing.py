from typing import Dict, List, Tuple

import numpy as np
import polars as pl
from great_tables import loc, style
from scipy.stats import brunnermunzel, rankdata
from great_tables import GT


def rankdata_2samp(x1: np.ndarray, x2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute midranks for two samples.

    Parameters
    ----------
    x1, x2 : array_like
        Original data for two samples that will be converted to midranks.

    Returns
    -------
    rank1 : ndarray
        Midranks of the first sample in the pooled sample.
    rank2 : ndarray
        Midranks of the second sample in the pooled sample.
    """
    x1 = np.asarray(x1)
    x2 = np.asarray(x2)

    nobs1 = len(x1)
    nobs2 = len(x2)
    if nobs1 == 0 or nobs2 == 0:
        raise ValueError("One sample has zero length")

    x_combined = np.concatenate((x1, x2))
    rank = rankdata(
        x_combined, method="average"
    )  # Compute midranks for the pooled data
    rank1 = rank[:nobs1]
    rank2 = rank[nobs1:]

    return rank1, rank2


def compute_treatment_effect(x1: np.ndarray, x2: np.ndarray) -> float:
    """
    Compute the estimated relative treatment effect between two samples using rank-based statistics.

    The relative treatment effect is defined as the difference in ranks between two samples. It estimates the
    probability that a randomly chosen observation from the second sample is greater than a randomly chosen observation
    from the first sample.

    Parameters
    ----------
    x1: array_like
        Observed values for the first sample.
    x2: array_like
        Observed values for the second sample.

    Returns
    -------
    treatment_effect : float
        The unbiased and consistent estimator for the relative treatment effect p hat.
    """
    x1 = x1.copy()[~np.isnan(x1)]
    x2 = x2.copy()[~np.isnan(x2)]

    # Get ranks of control and treatment in the combined sample
    _, rank2 = rankdata_2samp(x1, x2)
    n_x1, n_x2 = len(x1), len(x2)

    # Mean of ranks for the second sample
    mean_rank2 = np.mean(rank2)

    # Compute the relative treatment effect estimate
    relative_treatment_effect_estimate = (mean_rank2 - (n_x2 + 1) / 2) / n_x1

    return relative_treatment_effect_estimate


def analyze_campaigns(campaign_data: Dict[str, Dict[str, np.ndarray]]) -> GT:
    """
    Perform Brunner-Munzel tests on control and test campaign data and calculate estimated treatment effect.

    Parameters
    ----------
    campaign_data : dict
        A dictionary containing control and test campaign data, where the keys are 'control' and 'test', and values
        are dictionaries with field names as keys and numpy arrays as values for each field's data.

    Returns
    -------
    GT
        A Polars DataFrame containing the fields, Brunner-Munzel statistics, estimated treatment effect (test minus control),
        p-values, and conclusion for each field.
    """
    stats: List[float] = []
    p_values: List[float] = []
    p_hats: List[float] = []
    fields: List[str] = []
    conclusions: List[str] = []

    control_campaign = campaign_data["control"]
    test_campaign = campaign_data["test"]

    for (col_control, control_data), (_, test_data) in zip(
        control_campaign.items(), test_campaign.items()
    ):
        # Perform Brunner-Munzel test
        brunnermunzel_stat, p_value = brunnermunzel(
            x=control_data,
            y=test_data,
            alternative="two-sided",
            distribution="t",
            nan_policy="omit",
        )
        # Compute the estimated treatment effect
        estimated_treatment_effect = compute_treatment_effect(control_data, test_data)

        # Store results
        p_hats.append(estimated_treatment_effect)
        stats.append(brunnermunzel_stat)
        p_values.append(p_value)
        fields.append(col_control)
        conclusions.append(
            "Reject null hypothesis of no treatment effect"
            if p_value < 0.05
            else "Fail to reject null hypothesis of no treatment effect"
        )

    # Create a Polars DataFrame with results
    result = pl.DataFrame(
        {
            "field": fields,
            "stat": stats,
            "estimated_treatment_effect_test_minus_control": p_hats,
            "p_value": p_values,
            "conclusion": conclusions,
        }
    )

    # Style the DataFrame: green for significant p-values, red for non-significant
    result_gt: GT = result.style.tab_style(
        style.fill("#90EE90"),  # Fill green for significant p-values (< 0.05)
        loc.body(rows=pl.col("p_value") < 0.05),
    ).tab_style(
        style.fill("#FF6666"),  # Fill red for non-significant p-values (>= 0.05)
        loc.body(rows=pl.col("p_value") >= 0.05),
    )
    return result_gt
