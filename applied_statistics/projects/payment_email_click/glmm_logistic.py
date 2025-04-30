#!/usr/bin/env python3
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
import statsmodels.api as sm
from scipy.special import expit
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM


def simulate_data(
    class_dist: Dict[int, float],
    click_dist: Optional[Dict[int, float]] = None,
    n_groups: int = 20,
    group_size_range: Tuple[int, int] = (100, 500),
    beta1: float = np.log(2),
    sigma_u: float = 1.0,
    random_seed: int = 12,
) -> pd.DataFrame:
    """
    Simulate GLMM data with specified class distribution for payment and click rate.

    Parameters
    ----------
    class_dist : Dict[int, float]
        Desired marginal distribution for 'payment', e.g. {0: 0.91, 1: 0.09}.
        Must contain keys 0 and 1 with probability values that sum to 1.
    click_dist : Optional[Dict[int, float]], default=None
        Desired marginal distribution for 'email_click', e.g. {0: 0.7, 1: 0.3}.
        If None, defaults to {0: 0.5, 1: 0.5}.
    n_groups : int, default=20
        Number of groups (levels of alps_twentil).
    group_size_range : Tuple[int, int], default=(100, 500)
        Min and max size for each group; actual sizes are uniform random in this range.
    beta1 : float, default=np.log(2)
        Fixed effect log-odds ratio for email_click.
    sigma_u : float, default=1.0
        Standard deviation of random intercepts.
    random_seed : int, default 12
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - 'alps_twentil': Group identifier (1 to n_groups)
        - 'email_click': Binary indicator for email click (0 or 1)
        - 'payment': Binary outcome variable (0 or 1)
    """
    rng = np.random.default_rng(random_seed)

    # Determine baseline intercept to match marginal P(payment=1 | click=0, u=0)
    p1_target = class_dist.get(1, 0.5)
    beta0 = np.log(p1_target / (1 - p1_target))

    # Determine click probability
    if click_dist is None:
        click_prob = 0.5
    else:
        click_prob = click_dist.get(1, 0.5)

    # Simulate group (twentile) membership
    sizes = rng.integers(group_size_range[0], group_size_range[1] + 1, size=n_groups)
    groups = np.repeat(np.arange(1, n_groups + 1), sizes)
    N = len(groups)

    # Random intercepts
    u = rng.normal(0, sigma_u, size=n_groups)
    u_obs = u[groups - 1]

    # Simulate click and outcome
    email_click = rng.binomial(1, click_prob, size=N)
    eta = beta0 + beta1 * email_click + u_obs
    p = expit(eta)
    payment = rng.binomial(1, p, size=N)

    return pd.DataFrame(
        {"alps_twentil": groups, "email_click": email_click, "payment": payment}
    )


def main() -> int:
    class_dist = {0: 0.97, 1: 0.03}
    click_dist = {0: 0.985, 1: 0.015}

    data = simulate_data(class_dist, click_dist)

    endog = data["payment"]
    exog = sm.add_constant(data["email_click"])
    Z_df = pd.get_dummies(data["alps_twentil"], drop_first=False)
    exog_vc = sp.csr_matrix(Z_df.values)
    ident = np.zeros(exog_vc.shape[1], dtype=int)
    model = BinomialBayesMixedGLM(endog, exog, exog_vc, ident)
    result = model.fit_map()

    beta0_hat, beta1_hat = result.params[0], result.params[1]
    u_means = result.random_effects()["Mean"].values

    n = u_means.size
    ncols = 4
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(12, nrows * 3), sharex=True, sharey=True
    )

    for j in range(1, n + 1):
        ax = axes[(j - 1) // ncols, (j - 1) % ncols]
        uj = u_means[j - 1]
        p0 = expit(beta0_hat + uj)
        p1 = expit(beta0_hat + beta1_hat + uj)
        lift = p1 - p0

        ax.plot([0, 1], [p0, p1], marker="o", linewidth=2)
        ax.text(
            0.5,
            max(p0, p1) + 0.01,
            f"Lift={lift:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
        ax.set_title(f"alps_twentil={j}")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["click=0", "click=1"])
        ax.set_ylim(0, 1)

    # Remove any unused axes
    for idx in range(n, nrows * ncols):
        fig.delaxes(axes.flatten()[idx])

    fig.suptitle(
        "Predicted Payment Probability: Click Effect by alps_twentil",
        y=1.02,
        fontsize=16,
    )
    fig.text(0.5, 0.04, "Email click", ha="center")
    fig.text(
        0.04, 0.5, "Predicted probability of payment", va="center", rotation="vertical"
    )
    plt.tight_layout()
    plt.show()

    return 0


if __name__ == "__main__":
    main()
