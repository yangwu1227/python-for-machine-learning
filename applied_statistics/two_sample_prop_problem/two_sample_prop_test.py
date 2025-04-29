import argparse
from typing import Any, Dict, List

import polars as pl
from statsmodels.stats.proportion import proportion_effectsize, proportions_ztest


def apply_statsmodels(row: Dict[str, Any], alpha: float) -> Dict[str, Any]:
    """
    Apply statsmodels tests to a single row of data.

    Parameters
    ----------
    row : Dict[str, Any]
        Row data with treatment and control metrics
    alpha : float
        Significance level

    Returns
    -------
    Dict[str, Any]
        Dictionary of statistical test results
    """
    counts = [row["treatment_success_count"], row["control_success_count"]]
    nobs = [row["treatment_sample_size"], row["control_sample_size"]]
    z_stat, p_val = proportions_ztest(counts, nobs)
    eff = proportion_effectsize(
        row["treatment_prop"], row["control_prop"], method="normal"
    )

    # Cohen's h interpretation rule of thumb
    h = abs(eff)
    if h < 0.2:
        interpretation = "small effect"
    elif h < 0.5:
        interpretation = "medium effect"
    else:
        interpretation = "large effect"

    return {
        "z_stat": z_stat,
        "p_value": p_val,
        "statistically_significant": "yes" if p_val < alpha else "no",
        "effect_size": eff,
        "interpretation_guide": interpretation,
    }


def test_and_effect_size(data: pl.DataFrame, alpha: float) -> pl.DataFrame:
    """
    This function performs a two-sample proportions z-test (two-sided `H_0: prop_1 - prop_2 = 0`)
    and calculates Cohen's h.

    Parameters
    ----------
    data : pl.DataFrame
        Input DataFrame with columns:
            - metric: Name of the metric being tested
            - treatment_sample_size: Total observations in treatment group
            - treatment_success_count: Successful outcomes in treatment group
            - control_sample_size: Total observations in control group
            - control_success_count: Successful outcomes in control group
    alpha : float
        Significance level for the statistical test.

    Returns
    -------
    pl.DataFrame
        Original DataFrame with additional columns:
        - treatment_prop: Success proportion in treatment group
        - control_prop: Success proportion in control group
        - diff: Difference between proportions (treatment - control)
        - prop_pooled: Pooled proportion across both groups
        - std_diff: Standard deviation of the difference
        - z_stat: Z-statistic from the proportion test
        - p_value: P-value from the z-test
        - statistically_significant: 'yes' if p_value < alpha, else 'no'
        - effect_size: Cohen's h effect size
        - interpretation_guide: Interpretation of effect size magnitude
    """

    # Calculate proportions
    data = data.with_columns(
        [
            (pl.col("treatment_success_count") / pl.col("treatment_sample_size")).alias(
                "treatment_prop"
            ),
            (pl.col("control_success_count") / pl.col("control_sample_size")).alias(
                "control_prop"
            ),
        ]
    )

    # Intermediary columns for computing z-stat
    data = data.with_columns(
        [
            (pl.col("treatment_prop") - pl.col("control_prop")).alias("diff"),
            (
                (pl.col("treatment_success_count") + pl.col("control_success_count"))
                / (pl.col("treatment_sample_size") + pl.col("control_sample_size"))
            ).alias("prop_pooled"),
        ]
    )

    data = data.with_columns(
        [
            (
                (
                    pl.col("prop_pooled")
                    * (1 - pl.col("prop_pooled"))
                    * (
                        1 / pl.col("treatment_sample_size")
                        + 1 / pl.col("control_sample_size")
                    )
                ).sqrt()
            ).alias("std_diff")
        ]
    )

    # Apply the function to each row and collect results
    results: List[Dict[str, Any]] = [
        apply_statsmodels(row, alpha) for row in data.iter_rows(named=True)
    ]

    stats_data = pl.DataFrame(results)

    # Column-bind and sort by p-value (most significant first)
    result = pl.concat([data, stats_data], how="horizontal")
    return result.sort(by=pl.col("p_value"), descending=False)


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

    data = pl.read_csv(args.input)
    result_data = test_and_effect_size(data, args.alpha)

    # Format proportions and statistics for output with proper return_dtype
    result_data = result_data.with_columns(
        [
            (pl.col("treatment_prop") * 100)
            .map_elements(lambda x: f"{x:.2f}%", return_dtype=pl.Utf8)
            .alias("treatment_prop"),
            (pl.col("control_prop") * 100)
            .map_elements(lambda x: f"{x:.2f}%", return_dtype=pl.Utf8)
            .alias("control_prop"),
            pl.col("diff")
            .map_elements(lambda x: f"{x:.5f}", return_dtype=pl.Utf8)
            .alias("diff"),
            (pl.col("prop_pooled") * 100)
            .map_elements(lambda x: f"{x:.2f}%", return_dtype=pl.Utf8)
            .alias("prop_pooled"),
            pl.col("std_diff")
            .map_elements(lambda x: f"{x:.12f}", return_dtype=pl.Utf8)
            .alias("std_diff"),
            pl.col("z_stat")
            .map_elements(lambda x: f"{x:.4f}", return_dtype=pl.Utf8)
            .alias("z_stat"),
            pl.col("p_value")
            .map_elements(lambda x: f"{x:.12f}", return_dtype=pl.Utf8)
            .alias("p_value"),
            pl.col("effect_size")
            .map_elements(lambda x: f"{x:.12f}", return_dtype=pl.Utf8)
            .alias("effect_size"),
        ]
    )

    result_data.write_csv(args.output)
    print(f"Results written to {args.output}")

    return 0


if __name__ == "__main__":
    main()
