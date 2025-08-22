from copy import deepcopy
from typing import List

import numpy as np
import scipy.stats as stats
from scipy.optimize import minimize

"""
Problem Setting
---------------
We have data generated from a normal distribution with unknown mean (mu) and 
standard deviation (sigma). However, some data points are subject to right 
censoring at a specified threshold (censoring_threshold). 

In right-censoring, if the true data value exceeds the threshold, it is not 
observed directly, and we only know that it is larger than the threshold. 

Our goal is to estimate the unknown parameters (mu and sigma) of the normal 
distribution using Maximum Likelihood Estimation (MLE), taking the censoring 
into account.
"""

# True unknown parameters for data generation
mu_true = 8
sigma_true = 3
n = 120
censoring_threshold = 10

# Generate data from a normal distribution
x = np.random.normal(mu_true, sigma_true, n)

# Create censoring indicator: 1 if uncensored (observed), 0 if censored (unobserved)
censor_indicator = (x <= censoring_threshold).astype(int)

# Observed data: Y = X if uncensored, NaN if censored
y = deepcopy(x)
y[censor_indicator == 0] = np.nan  # Mask censored data with NaN


def negative_log_likelihood(
    params: List[float],
    y: np.ndarray,
    censor_indicator: np.ndarray,
    censoring_threshold: float,
) -> float:
    """
    Compute the negative log-likelihood for censored data under a normal
    distribution.

    Parameters
    ----------
    params : List[float]
        List containing [mu, log_sigma2], where log_sigma2 is the logarithm of
        the variance.
    y : np.ndarray
        Array of observed data, where NaN indicates censored observations.
    censor_indicator : np.ndarray
        Array of indicators (1 if uncensored, 0 if censored).
    censoring_threshold : float
        The threshold above which data is considered censored.

    Returns
    -------
    float
        The total negative log-likelihood, combining contributions from
        uncensored and censored data.

    Notes
    -----
    For uncensored data, the standard normal log-density (log of PDF) is used.
    For censored data, the survival function (1 - CDF) is used to account for
    the probability that the true value exceeds the censoring threshold. This
    ensures that the likelihood reflects the probability of observing the data
    given the right-censoring mechanism.
    """
    mu, log_sigma2 = params
    # Input variance is in log scale, so exponentiate to get the actual variance
    sigma2 = np.exp(log_sigma2)
    sigma = np.sqrt(sigma2)

    y_uncensored = y[censor_indicator == 1]
    n_censored = np.sum(censor_indicator == 0)

    # Negative log-likelihood for uncensored data (standard normal log PDF)
    neg_log_likelihood_uncensored = -np.sum(
        stats.norm.logpdf(y_uncensored, loc=mu, scale=sigma)
    )

    # Represents how many standard deviations the threshold is above the mean
    z = (censoring_threshold - mu) / sigma
    # Survival function 1 - CDF(z) gives P(X > censoring_threshold)
    prob_censored = 1 - stats.norm.cdf(z)
    # Avoid log(0) by using a small positive probability floor (1e-12)
    neg_log_likelihood_censored = -n_censored * np.log(np.maximum(prob_censored, 1e-12))

    # Total negative log-likelihood
    total_neg_log_likelihood = (
        neg_log_likelihood_uncensored + neg_log_likelihood_censored
    )
    return total_neg_log_likelihood


def main() -> int:
    # Initial estimates for the parameters based on observed (uncensored) data
    y_observed = y[~np.isnan(y)]
    mu_init = np.mean(y_observed)
    sigma2_init = np.var(y_observed, ddof=1)  # Use sample variance with ddof=1
    sigma2_init = max(sigma2_init, 1e-6)  # Ensure non-zero initial variance
    params_init = [mu_init, np.log(sigma2_init)]

    # Perform optimization to minimize the negative log-likelihood
    result = minimize(
        negative_log_likelihood,
        params_init,
        args=(y, censor_indicator, censoring_threshold),
        # BFGS is a quasi-Newton method for optimization
        method="BFGS",
    )

    if result.success:
        mu_est, log_sigma2_est = result.x
        sigma2_est = np.exp(log_sigma2_est)
        sigma_est = np.sqrt(sigma2_est)
        print("Optimization successful.")
        print(f"Estimated mu: {mu_est:.4f}")
        print(f"Estimated sigma: {sigma_est:.4f}")
        print(f"True mu: {mu_true}")
        print(f"True sigma: {sigma_true}")
    else:
        print("Optimization failed.")
        print(result.message)

    return 0


if __name__ == "__main__":
    main()
