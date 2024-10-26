from typing import Optional
import numpy as np
from scipy.stats import norm


def two_sided_sample_size(
    control_prop: np.ndarray[float],
    treatment_prop: np.ndarray[float],
    treatment_fraction: float,
    alpha: Optional[float] = 0.05,
    power: Optional[float] = 0.8,
) -> float:
    """
    Calculate the total sample size required for a two-sided Wilcoxon-Mann-Whitney test
    with adjustments for ties in ordinal or non-normal data. This method follows the
    sample size calculation approach described in the work of Zhao et al. for the
    Wilcoxon-Mann-Whitney test under tied observations.

    Parameters:
    ----------
    control_prop : np.ndarray[float]
        Proportion of each category in the control group.
    treatment_prop : np.ndarray[float]
        Proportion of each category in the treatment group.
    treatment_fraction : float
        Fraction of the total sample that will be allocated to the treatment group.
    alpha : Optional[float], default=0.05
        Significance level for the test.
    power : Optional[float], default=0.8
        Desired power level for the test.

    Returns:
    -------
    float
        The calculated total sample size required to achieve the specified power
        at the given significance level.

    Notes:
    ------
    This method implements the sample size calculation for the Wilcoxon-Mann-Whitney
    test with adjustments for ties based on the method described in:

    Zhao, Y.D., Rahardja, D., & Qu, Y. (2008). Sample size calculation for the Wilcoxon–
    Mann–Whitney test adjusting for ties. Statistics in Medicine, 27(24), 4620–4635.

    """
    treatment_prop = np.asarray(treatment_prop)
    control_prop = np.asarray(control_prop)

    # With equal variance assumption simplification for wmw
    z_alpha = norm.ppf(1 - alpha / 2, loc=0, scale=1)
    z_beta = norm.ppf(power, loc=0, scale=1)
    quantile_term = (z_alpha + z_beta) ** 2

    # Variance under null hypothesis
    cubed_prop_sum = (
        (1 - treatment_fraction) * control_prop + (treatment_fraction * treatment_prop)
    ) ** 3
    adj_term = 12 * treatment_fraction * (1 - treatment_fraction)
    adj_var_null = (1 - cubed_prop_sum.sum()) / adj_term

    prod_term = np.sum(control_prop * treatment_prop)
    mixed_term = np.sum(control_prop[1:] * np.cumsum(treatment_prop)[:-1])
    est_treat_eff = mixed_term + 0.5 * prod_term

    n_total = (quantile_term * adj_var_null) / (est_treat_eff - 0.5) ** 2

    return n_total


def main() -> int:
    treatment_fraction = 0.5
    control_prop = [0.1, 0.2, 0.3, 0.2, 0.2]
    treatment_prop = [0.2, 0.3, 0.3, 0.1, 0.1]

    n_total = two_sided_sample_size(control_prop, treatment_prop, treatment_fraction)

    print(f"Total sample size: {n_total}")

    return 0


if __name__ == "__main__":
    main()
