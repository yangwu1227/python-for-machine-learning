import math
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import polars as pl
from scipy.special import betaln, digamma
from scipy.stats import beta as beta_dist
from scipy.stats import betabinom


@dataclass(frozen=True)
class BetaParams(object):
    """
    Shape parameters for a Beta distribution.
    """

    alpha: float
    beta: float


@dataclass(frozen=True)
class DataScenario(object):
    """
    Binomial data summary: `s` successes in `n` trials.
    """

    successes: int
    trials: int


@dataclass(frozen=True)
class PosteriorSummary:
    """
    Posterior summary statistics for Beta(alpha, beta).
    """

    mean: float
    variance: float
    mode: float
    ci_low: float
    ci_high: float


def posterior_params(prior: BetaParams, data: DataScenario) -> BetaParams:
    """
    Compute posterior Beta parameters for the Beta–Binomial model.

    Parameters
    ----------
    prior : BetaParams
        Prior parameters with alpha > 0 and beta > 0.
    data : DataScenario
        Observed counts with 0 <= successes <= trials.

    Returns
    -------
    BetaParams
        Posterior parameters Beta(alpha_post, beta_post).
    """
    s: int = data.successes
    n: int = data.trials
    return BetaParams(prior.alpha + s, prior.beta + (n - s))


def beta_summaries(params: BetaParams, ci: float = 0.95) -> PosteriorSummary:
    """
    Compute summary statistics for Beta(alpha, beta).

    Parameters
    ----------
    params : BetaParams
        Beta shape parameters with alpha > 0 and beta > 0.
    ci : float, optional
        Central credible interval mass, by default 0.95.

    Returns
    -------
    PosteriorSummary
        Posterior mean, variance, mode, and central credible interval.
    """
    a: float = params.alpha
    b: float = params.beta
    mean: float = a / (a + b)
    variance: float = (a * b) / (((a + b) ** 2) * (a + b + 1))
    mode: float = (a - 1) / (a + b - 2) if (a > 1 and b > 1) else math.nan
    lo, hi = beta_dist.ppf([(1 - ci) / 2, 1 - (1 - ci) / 2], a, b)
    return PosteriorSummary(
        mean=mean, variance=variance, mode=mode, ci_low=float(lo), ci_high=float(hi)
    )


def kl_beta(p: BetaParams, q: BetaParams) -> float:
    """
    KL divergence KL(Beta(p) || Beta(q)), which can be considered
    as a measure of inefficiency of assuming that the distribution
    is `q` when the true distribution is `p`.

    Parameters
    ----------
    p : BetaParams
        Distribution in the numerator of KL. Typically posterior.
    q : BetaParams
        Reference distribution. Typically prior.

    Returns
    -------
    float
        Nonnegative divergence.

    Notes
    -----
    KL(Beta(a_1, b_1) || Beta(a_0, b_0)) =
        ln B(a_0, b_0) - ln B(a_1, b_1)
        + (a_1 - a_0) * digamma(a_1)
        + (b_1 - b_0) * digamma(b_1)
        + (a_0 - a_1 + b_0 - b_1) * digamma(a_1 + b_1)
    where B is the Beta function and digamma is the digamma function.

    Reference: https://en.wikipedia.org/wiki/Beta_distribution#Quantities_of_information_(entropy)
    """
    a_1, b_1 = p.alpha, p.beta
    a_0, b_0 = q.alpha, q.beta
    return float(
        betaln(a_0, b_0)
        - betaln(a_1, b_1)
        + ((a_1 - a_0) * digamma(a_1))
        + ((b_1 - b_0) * digamma(b_1))
        + ((a_0 - a_1 + b_0 - b_1) * digamma(a_1 + b_1))
    )


def beta_binomial_pmf(k: np.ndarray, n: int, params: BetaParams) -> np.ndarray:
    """
    Beta–Binomial predictive pmf for counts.

    Parameters
    ----------
    k : np.ndarray
        Integer support values from 0 to n inclusive.
    n : int
        Number of trials for the predictive distribution.
    params : BetaParams
        Beta parameters for the predictive (typically posterior).

    Returns
    -------
    np.ndarray
        pmf over k with shape (len(k),).
    """
    return betabinom(n, params.alpha, params.beta).pmf(k)


def compute_density_grid(
    x: np.ndarray,
    priors: Sequence[BetaParams],
    data_scenarios: Sequence[DataScenario],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Precompute all prior and posterior densities over a grid.

    Parameters
    ----------
    x : np.ndarray
        Grid on [0, 1] for density evaluation.
    priors : Sequence[BetaParams]
        List of prior Beta parameters.
    data_scenarios : Sequence[DataScenario]
        List of data scenarios.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        prior_pdf[i, j, :] and post_pdf[i, j, :] for prior i and scenario j.
    """
    I: int = len(priors)
    J: int = len(data_scenarios)
    prior_pdf = np.empty((I, J, x.size), dtype=float)
    post_pdf = np.empty((I, J, x.size), dtype=float)
    for i, prior in enumerate(priors):
        prior_curve = beta_dist.pdf(x, prior.alpha, prior.beta)
        for j, ds in enumerate(data_scenarios):
            post = posterior_params(prior, ds)
            post_curve = beta_dist.pdf(x, post.alpha, post.beta)
            prior_pdf[i, j, :] = prior_curve
            post_pdf[i, j, :] = post_curve
    return prior_pdf, post_pdf


def _safe_pdf(a: float, b: float, x: np.ndarray) -> np.ndarray:
    """
    Evaluate Beta pdf and mask nonfinite values.

    Parameters
    ----------
    a : float
        Beta alpha parameter.
    b : float
        Beta beta parameter.
    x : np.ndarray
        Points in (0, 1) to evaluate.
    """
    y = beta_dist.pdf(x, a, b)
    return np.where(np.isfinite(y), y, np.nan)


def _format_panel_text(
    prior: BetaParams,
    posterior: BetaParams,
    ds: DataScenario,
    ci: float = 0.95,
) -> str:
    """
    Create a clear, spaced annotation block for a panel.

    Parameters
    ----------
    prior : BetaParams
        Prior parameters.
    posterior : BetaParams
        Posterior parameters.
    ds : DataScenario
        Data scenario s, n.
    ci : float, optional
        Credible interval mass, by default 0.95.

    Returns
    -------
    str
        Formatted multi-line string.
    """
    s, n = ds.successes, ds.trials
    mu0: float = prior.alpha / (prior.alpha + prior.beta)
    n0: float = prior.alpha + prior.beta
    summ: PosteriorSummary = beta_summaries(posterior, ci=0.95)
    phat: float = s / n if n > 0 else float("nan")
    shrink: float = abs(summ.mean - phat) if n > 0 else float("nan")
    dkl: float = kl_beta(posterior, prior)
    return (
        f"Prior: n₀={n0:.1f}, μ₀={mu0:.3f}\n"
        f"Post: μ₁={summ.mean:.3f} [{summ.ci_low:.3f},{summ.ci_high:.3f}]\n"
        f"Data: {s}/{n}, p̂={phat:.3f}\n"
        f"Shrink: {shrink:.3f}, KL: {dkl:.3f}"
    )


def annotate_panel(
    ax: plt.Axes,
    prior: BetaParams,
    posterior: BetaParams,
    ds: DataScenario,
    ci: float = 0.95,
    xloc: float = 0.02,
    yloc: float = 0.98,
) -> None:
    """
    Annotate a panel with compact numerical diagnostics.

    Parameters
    ----------
    ax : plt.Axes
        Target axes.
    prior : BetaParams
        Prior parameters.
    posterior : BetaParams
        Posterior parameters.
    ds : DataScenario
        Data scenario s, n.
    ci : float, optional
        Credible interval mass, by default 0.95.
    xloc : float, optional
        Text x position in axes fraction, by default 0.02.
    yloc : float, optional
        Text y position in axes fraction, by default 0.98.
    """
    txt: str = _format_panel_text(prior, posterior, ds, ci=ci)
    ax.text(
        xloc,
        yloc,
        txt,
        ha="left",
        va="top",
        fontsize=9,
        transform=ax.transAxes,
        linespacing=1.4,
        bbox=dict(
            facecolor="white",
            alpha=0.8,
            edgecolor="none",
            boxstyle="round,pad=0.4",
        ),
        family="monospace",
    )


def make_density_figure(
    priors: Sequence[BetaParams],
    data_scenarios: Sequence[DataScenario],
    x: npt.NDArray[np.floating],
    output_path: Optional[Union[Path, str]] = None,
) -> plt.Figure:
    """
    Create a grid of prior and posterior densities with diagnostics.

    Handles boundary singularities by masking nonfinite pdf values and
    setting y-limits using a high percentile of finite values.

    Parameters
    ----------
    priors : Sequence[BetaParams]
        List of prior Beta parameters.
    data_scenarios : Sequence[DataScenario]
        List of data scenarios.
    x : npt.NDArray[np.floating]
        Grid on (0, 1) for density evaluation.
    output_path : Optional[Union[Path, str]]
        If given, save the figure to this path, by default None.

    Returns
    -------
    plt.Figure
        The created figure.
    """
    I, J = len(priors), len(data_scenarios)

    # Compute a robust global y-limit
    prior_pdf, post_pdf = compute_density_grid(x, priors, data_scenarios)
    # Element-wise max with shape (I, J, len(x))
    combined: npt.NDArray[np.floating] = np.maximum(prior_pdf, post_pdf)
    combined_processed: npt.NDArray[np.floating] = np.where(
        np.isfinite(combined), combined, np.nan
    )
    ymax: float = (
        float(np.nanpercentile(combined_processed, 99.9))
        if np.isfinite(np.nanmax(combined_processed))
        else 1.0
    )
    if not np.isfinite(ymax) or ymax <= 0:
        ymax = 1.0

    fig, axes = plt.subplots(I, J, figsize=(4.0 * J, 3.2 * I), sharex=True, sharey=True)
    if I == 1 and J == 1:
        axes = np.array([[axes]])
    elif I == 1:
        axes = axes[np.newaxis, :]
    elif J == 1:
        axes = axes[:, np.newaxis]

    for i, prior in enumerate(priors):
        for j, ds in enumerate(data_scenarios):
            ax: plt.Axes = axes[i, j]
            post = posterior_params(prior, ds)

            y_prior = _safe_pdf(prior.alpha, prior.beta, x)
            y_post = _safe_pdf(post.alpha, post.beta, x)

            ax.plot(x, y_prior, linestyle="--", label="Prior", color="red")
            ax.plot(x, y_post, linestyle="-", label="Posterior", color="blue")
            ax.fill_between(x, y_post, alpha=0.20, color="blue")
            ax.set_ylim(0, ymax * 1.05)

            if j == 0:
                ax.set_ylabel(f"Beta({prior.alpha:g}, {prior.beta:g})", fontsize=10)
            if i == I - 1:
                ax.set_xlabel(f"{ds.successes}/{ds.trials}", fontsize=10)

            annotate_panel(ax, prior, post, ds)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.suptitle("Beta–Binomial: Prior vs Posterior on Theta", fontsize=14)
    fig.supxlabel("Theta", fontsize=12)
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=160, bbox_inches="tight")
    return fig


def make_predictive_figure(
    priors: Sequence[BetaParams],
    data_scenarios: Sequence[DataScenario],
    output_path: Optional[Union[Path, str]] = None,
) -> plt.Figure:
    """
    Create a grid of Beta–Binomial posterior predictive distributions.

    Parameters
    ----------
    priors : Sequence[BetaParams]
        List of prior Beta parameters.
    data_scenarios : Sequence[DataScenario]
        List of data scenarios.
    output_path : Optional[Union[Path, str]]
        If given, save the figure to this path, by default None.

    Returns
    -------
    plt.Figure
        The created figure.
    """
    I, J = len(priors), len(data_scenarios)
    fig, axes = plt.subplots(
        I,
        J,
        figsize=(4.0 * J, 3.5 * I),
        sharex=False,
        sharey=True,  # Increased figure size
    )
    if I == 1 and J == 1:
        axes = np.array([[axes]])
    elif I == 1:
        axes = axes[np.newaxis, :]
    elif J == 1:
        axes = axes[:, np.newaxis]

    ymax: float = 0.0
    pmf_cache: Dict[Tuple[int, float, float], np.ndarray] = {}

    # First pass to compute a global ymax
    for prior in priors:
        for ds in data_scenarios:
            post = posterior_params(prior, ds)
            k = np.arange(ds.trials + 1)
            key = (ds.trials, post.alpha, post.beta)
            if key not in pmf_cache:
                pmf_cache[key] = beta_binomial_pmf(k, ds.trials, post)
            ymax = max(ymax, float(pmf_cache[key].max()))

    # Second pass to plot
    for i, prior in enumerate(priors):
        for j, ds in enumerate(data_scenarios):
            ax: plt.Axes = axes[i, j]
            post = posterior_params(prior, ds)
            k = np.arange(ds.trials + 1)
            pmf = pmf_cache[(ds.trials, post.alpha, post.beta)]

            ax.vlines(k, 0.0, pmf, linewidth=1.0)
            ax.set_ylim(0, ymax * 1.05)
            ax.set_xlim(-0.5, ds.trials + 0.5)

            # Mark the observed s
            ax.plot([ds.successes], [pmf[ds.successes]], marker="o", ms=4)

            if j == 0:
                ax.set_ylabel(f"Beta({prior.alpha:g}, {prior.beta:g})", fontsize=10)
            ax.set_title(
                f"s = {ds.successes}, n = {ds.trials}",
                fontsize=10,
            )

    fig.suptitle(
        "Posterior Predictive: P(K = k | data) under Beta–Binomial",
        fontsize=14,  # Increased font size
    )
    fig.supxlabel("k = successes", fontsize=12)
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=160, bbox_inches="tight")
    return fig


def report_table(
    priors: Sequence[BetaParams],
    data_scenarios: Sequence[DataScenario],
) -> pl.DataFrame:
    """
    Create a Polars DataFrame of posterior summaries and KL divergence.

    Parameters
    ----------
    priors : Sequence[BetaParams]
        Prior parameters.
    data_scenarios : Sequence[DataScenario]
        Data scenarios.

    Returns
    -------
    pl.DataFrame
        Structured table of posterior summaries.
    """
    rows: list[dict[str, float | str]] = []

    for prior in priors:
        mu0 = prior.alpha / (prior.alpha + prior.beta)
        n0 = prior.alpha + prior.beta
        for ds in data_scenarios:
            post = posterior_params(prior, ds)
            summ = beta_summaries(post, ci=0.95)
            phat = ds.successes / ds.trials if ds.trials > 0 else float("nan")
            shrink = abs(summ.mean - phat) if ds.trials > 0 else float("nan")
            dkl = kl_beta(post, prior)

            rows.append(
                {
                    "prior": f"Beta({prior.alpha:g}, {prior.beta:g})",
                    "data": f"{ds.successes}/{ds.trials}",
                    "prior_mean": mu0,
                    "prior_ess": n0,
                    "post_mean": summ.mean,
                    "post_ci_low": summ.ci_low,
                    "post_ci_high": summ.ci_high,
                    "abs(shrink)": shrink,
                    "KL(post||prior)": dkl,
                }
            )

    return pl.DataFrame(rows)


def main() -> int:
    parser: ArgumentParser = ArgumentParser(
        description="Beta-Binomial model with various Beta priors and data scenarios"
    )
    parser.add_argument(
        "--output_dir", type=str, help="Directory to save output diagrams"
    )
    args, _ = parser.parse_known_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Priors
    priors: List[BetaParams] = [
        BetaParams(1.0, 1.0),  # Uniform
        BetaParams(2.0, 2.0),  # Weakly informative, symmetric
        BetaParams(0.5, 0.5),  # Favors extremes
        BetaParams(5.0, 1.0),  # Favors high probabilities
        BetaParams(1.0, 5.0),  # Favors low probabilities
    ]

    # Data scenarios, including edge cases s = 0 and s = n
    data_scenarios: List[DataScenario] = [
        DataScenario(5, 100),
        DataScenario(10, 100),
        DataScenario(25, 100),
        DataScenario(50, 100),
        DataScenario(75, 100),
        DataScenario(90, 100),
    ]

    # Grid for θ densities; avoid exact 0 and 1 to prevent boundary singularities
    eps: float = 1e-6
    n_values: int = 1000
    x: npt.NDArray[np.floating] = np.linspace(eps, 1.0 - eps, n_values)

    _ = make_density_figure(
        priors, data_scenarios, x, output_path=output_dir / "beta_binomial_theta.png"
    )
    _ = make_predictive_figure(
        priors, data_scenarios, output_path=output_dir / "beta_binomial_predictive.png"
    )

    print(report_table(priors, data_scenarios))
    plt.show()
    return 0


if __name__ == "__main__":
    main()
