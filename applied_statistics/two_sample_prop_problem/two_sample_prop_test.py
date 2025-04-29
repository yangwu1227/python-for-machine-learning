import argparse

import numpy as np
import pandas as pd
from statsmodels.stats.proportion import proportion_effectsize, proportions_ztest


def test_and_effect_size(data: pd.DataFrame, alpha: float) -> pd.DataFrame:
    """
    Compute statistical metrics for A/B test proportion data.

    This function performs a two-sample proportions z-test and calculates the
    cohen'h.

    Parameters
    ----------
    data : pd.DataFrame
        Input DataFrame with columns:
        - metric: Name of the metric being tested
        - treatment_sample_size: Total observations in treatment group
        - treatment_success_count: Successful outcomes in treatment group
        - control_sample_size: Total observations in control group
        - control_success_count: Successful outcomes in control group
    alpha : float
        Significance level for the statistical test (default: 0.05).

    Returns
    -------
    pd.DataFrame
        Original DataFrame with additional columns:
        - treatment_prop: Success proportion in treatment group
        - control_prop: Success proportion in control group
        - diff: Difference between proportions (treatment - control)
        - prop_pooled: Pooled proportion across both groups
        - std_diff: Standard deviation of the difference
        - z_stat: Z-statistic from the proportion test
        - p_value: P-value from the z-test
        - statistically_significant: 'Yes' if p_value < alpha, else 'No'
        - effect_size: Cohen's h effect size
        - interpretation_guide: Interpretation of effect size magnitude
    """

    # Calculate proportions
    data["treatment_prop"] = (
        data["treatment_success_count"] / data["treatment_sample_size"]
    )
    data["control_prop"] = data["control_success_count"] / data["control_sample_size"]

    # Intermediary columns
    data["diff"] = data["treatment_prop"] - data["control_prop"]
    data["prop_pooled"] = (
        data["treatment_success_count"] + data["control_success_count"]
    ) / (data["treatment_sample_size"] + data["control_sample_size"])
    data["std_diff"] = np.sqrt(
        data["prop_pooled"]
        * (1 - data["prop_pooled"])
        * (1 / data["treatment_sample_size"] + 1 / data["control_sample_size"])
    )

    # Compute z-stat, p-value, significance, and effect size
    z_stats = []
    p_values = []
    sig_flags = []
    effect_sizes = []
    interpretations = []

    for _, row in data.iterrows():
        counts = [row["treatment_success_count"], row["control_success_count"]]
        nobs = [row["treatment_sample_size"], row["control_sample_size"]]
        zstat, pval = proportions_ztest(counts, nobs)
        eff = proportion_effectsize(
            row["treatment_prop"], row["control_prop"], method="normal"
        )

        z_stats.append(zstat)
        p_values.append(pval)
        sig_flags.append("Yes" if pval < alpha else "No")
        effect_sizes.append(eff)
        # Cohen's h interpretation rule of thumb: small (<0.2), medium (<0.5), large (>=0.5)
        h = abs(eff)
        if h < 0.2:
            interpretations.append("small effect")
        elif h < 0.5:
            interpretations.append("medium effect")
        else:
            interpretations.append("large effect")

    data["z_stat"] = z_stats
    data["p_value"] = p_values
    data["statistically_significant"] = sig_flags
    data["effect_size"] = effect_sizes
    data["interpretation_guide"] = interpretations

    return data


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run two-sample proportions z-test on A/B data and output metrics."
    )
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="Path to input CSV file with columns: metric, treatment_sample_size, treatment_success_count, control_sample_size, control_success_count",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Path to output CSV file where results will be written",
    )
    parser.add_argument(
        "-a",
        "--alpha",
        type=float,
        default=0.05,
        help="Significance level for the z-test (default: 0.05)",
    )
    args = parser.parse_args()

    data = pd.read_csv(args.input)
    result_data = test_and_effect_size(data, args.alpha)

    # Format proportions and statistics for output
    result_data["treatment_prop"] = (result_data["treatment_prop"] * 100).map(
        "{:.2f}%".format
    )
    result_data["control_prop"] = (result_data["control_prop"] * 100).map(
        "{:.2f}%".format
    )
    result_data["diff"] = result_data["diff"].map("{:.5f}".format)
    result_data["prop_pooled"] = (result_data["prop_pooled"] * 100).map(
        "{:.2f}%".format
    )
    result_data["std_diff"] = result_data["std_diff"].map("{:.12f}".format)
    result_data["z_stat"] = result_data["z_stat"].map("{:.4f}".format)
    result_data["p_value"] = result_data["p_value"].map("{:.12f}".format)
    result_data["effect_size"] = result_data["effect_size"].map("{:.12f}".format)

    result_data.to_csv(args.output, index=False)
    print(f"Results written to {args.output}")

    return 0


if __name__ == "__main__":
    main()
