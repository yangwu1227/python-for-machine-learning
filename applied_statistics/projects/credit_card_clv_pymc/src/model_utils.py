from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifetimes import GammaGammaFitter, ParetoNBDFitter
from matplotlib.axes import Axes
from pymc_marketing import clv
from pymc_marketing.clv.models.pareto_nbd import ParetoNBDModel
from pymc_marketing.prior import Prior

pd.set_option("mode.copy_on_write", True)


def create_btyd_model_config_mle(
    rfm_data: pd.DataFrame, use_mle_for_alpha_beta: bool = False
) -> Dict[str, Any]:
    """
    Create a model_config dictionary for ParetoNBDModel using a lifetimes ParetoNBDFitter
    MLE approach.

    1. Fit a ParetoNBDFitter (frequentist) on the frequency, recency, and T columns.
    2. Extract the MLE parameters (alpha, r, beta, s).
    3. Map them into Prior objects for pymc_marketing. We interpret them as follows:
       - r, alpha for the transaction Gamma(r, alpha)
       - s, beta for the dropout Gamma(s, beta)
    4. If `use_mle_for_alpha_beta` is True, we also set alpha_prior and beta_prior
       to the MLE rates. Otherwise, we fall back to a default Gamma(2.0, 0.2) for both.
    5. Return a dict of priors (r_prior, alpha_prior, s_prior, beta_prior).

    Parameters
    ----------
    rfm_data : pd.DataFrame
        DataFrame with columns ["frequency", "recency", "T"] for each customer.
        - frequency : int
            Number of repeat purchases.
        - recency : float
            Time between the first and last purchase (same units as T).
        - T : float
            Time from first purchase until end of observation (T >= recency).

    use_mle_for_alpha_beta : bool, optional
        If True, alpha_prior and beta_prior will also use the MLE rate from lifetimes.
        Otherwise, they default to a moderate Gamma(2.0, 0.2).

    Returns
    -------
    Dict[str, Any]
        A dictionary that can be passed as `model_config` to `ParetoNBDModel`. It
        contains keys "r_prior", "alpha_prior", "s_prior", "beta_prior", each mapped
        to a Prior object from `pymc_marketing`.
    """

    freq: np.ndarray = rfm_data["frequency"].values
    rec: np.ndarray = rfm_data["recency"].values
    T_: np.ndarray = rfm_data["T"].values

    # ---------- 1. Fit the Lifetimes ParetoNBD model (frequentist MLE) ---------- #

    pnbd_lt = ParetoNBDFitter()
    pnbd_lt.fit(freq, rec, T_)

    # The `pnbd_lt.params_` OrderedDict should have alpha, beta, r, s
    # These are shape/rate parameters for Lifetimes' parameterization:
    #    p(lambda) ~ Gamma(r, alpha), with alpha as rate
    #    p(mu)     ~ Gamma(s, beta),  with beta  as rate
    r_mle = pnbd_lt.params_["r"]
    alpha_mle = pnbd_lt.params_["alpha"]  # Rate for lambda
    s_mle = pnbd_lt.params_["s"]
    beta_mle = pnbd_lt.params_["beta"]  # Rate for mu

    # -------------------------- 2. Build Prior objects -------------------------- #

    # The r_prior and s_prior always come from MLE
    r_prior = Prior("Gamma", alpha=r_mle, beta=alpha_mle)
    s_prior = Prior("Gamma", alpha=s_mle, beta=beta_mle)

    # The alpha_prior and beta_prior can either use the MLE or defaults
    if use_mle_for_alpha_beta:
        # Here we use MLE to inform the rate parameter but fix a shape that reflects a moderate prior belief
        alpha_prior = Prior("Gamma", alpha=2.0, beta=alpha_mle)
        beta_prior = Prior("Gamma", alpha=2.0, beta=beta_mle)
    else:
        # A weakly informative prior (values here should be chosen based on domain knowledge or prior predictive checks)
        alpha_prior = Prior("Gamma", alpha=2.0, beta=0.2)
        beta_prior = Prior("Gamma", alpha=2.0, beta=0.2)

    model_config = {
        "r_prior": r_prior,
        "alpha_prior": alpha_prior,
        "s_prior": s_prior,
        "beta_prior": beta_prior,
    }

    return model_config


def run_pnbd_ppcs(
    rfm_data: pd.DataFrame,
    model_config: Optional[Dict[str, Prior]] = None,
    fit_method: str = "map",
    prior_samples: int = 100,
    max_purchases: int = 10,
    random_seed: int = 42,
    fig_size: Tuple[int, int] = (20, 7),
    **kwargs,
) -> Tuple[ParetoNBDModel, Axes, Axes]:
    """
    End-to-end workflow for a Pareto/NBD model with prior and posterior predictive checks.

    Parameters
    ----------
    rfm_data : pd.DataFrame
        Must include columns ['customer_id', 'frequency', 'recency', 'T'] (and optionally 'monetary_value').
    model_config : Optional[Dict[str, Prior]]
        Dictionary specifying the priors and covariates. If None, defaults to built-in priors.
    fit_method : str, optional
        Method used to fit the model. Options are 'map', 'demz', or 'mcmc'.
        Defaults to 'map'.
    prior_samples : int, optional
        Number of samples to draw during the prior predictive check. Defaults to 100.
    max_purchases : int, optional
        Cutoff for bars of purchase counts to plot. Defaults to 10.
    random_seed : int, optional
        Random seed for reproducibility of sampling. Defaults to 42.
    fig_size : Tuple[int, int], optional
        Size of the plots for the prior and posterior predictive checks. Defaults to (20, 7).
    **kwargs
        Additional keyword arguments to pass to the model fitting method.

    Returns
    -------
    Tuple[ParetoNBDModel, Axes, Axes]
        - The fitted ParetoNBDModel object, containing the model, idata, etc.
        - Axes object for the prior predictive check plot.
        - Axes object for the posterior predictive check plot.
    """
    pnbd_model: ParetoNBDModel = ParetoNBDModel(
        data=rfm_data,
        model_config=model_config,
    )

    # Build the model (must be done before prior predictive sampling)
    pnbd_model.build_model()

    # Plot a prior predictive check for the customer purchase frequency distribution
    _, ax_prior = plt.subplots(figsize=fig_size)
    prior_check: Axes = clv.plot_expected_purchases_ppc(
        model=pnbd_model,
        ppc="prior",
        samples=prior_samples,  # Number of draws from prior
        max_purchases=max_purchases,
        random_seed=random_seed,
        ax=ax_prior,
    )
    # Fit the model (map, demz, or mcmc)
    pnbd_model.fit(fit_method=fit_method, **kwargs)

    # Plot a posterior predictive check for the customer purchase frequency distribution
    _, ax_posterior = plt.subplots(figsize=fig_size)
    posterior_check: Axes = clv.plot_expected_purchases_ppc(
        model=pnbd_model,
        ppc="posterior",
        max_purchases=max_purchases,
        random_seed=random_seed,
        ax=ax_posterior,
    )
    return pnbd_model, prior_check, posterior_check


def create_spend_model_config_mle(
    rfm_data: pd.DataFrame, use_mle_for_v: bool = False
) -> Dict[str, Any]:
    """
    Create a model_config dictionary for a GammaGammaModel using a lifetimes
    GammaGammaFitter MLE approach.

    1. Fit a GammaGammaFitter (frequentist) on the frequency and monetary_value columns.
    2. Extract the MLE parameters (p, q, v).
        - p: Shape parameter of the per-transaction Gamma
        - q: Shape parameter of the Gamma distribution on the rate nu
        - v: Rate parameter for the Gamma distribution on nu
    3. Map them into Prior objects for pymc_marketing. By default, we set:
        - p_prior = Gamma(p_mle, 1.0)      # shape=p_mle, rate=1.0
        - q_prior = Gamma(q_mle, 1.0)      # shape=q_mle, rate=1.0
        - v_prior = Gamma(2.0, v_mle)      # shape=2.0, rate=v_mle, if use_mle_for_v
                                           # else shape=2.0, rate=0.2
      This approach ensures the priors are centered around the MLE, while still
      allowing further Bayesian updating. You can adjust shapes/rates to suit your data.
    4. Return a dict of priors (p_prior, q_prior, v_prior).

    Parameters
    ----------
    rfm_data : pd.DataFrame
        DataFrame with columns:
          * "frequency": Number of repeat purchases
          * "monetary_value": Mean (or total) spend for those repeat purchases
          (no need for "recency" or "T" in the Gamma-Gamma model).
        Must have non-zero frequency to fit the model.
    use_mle_for_v : bool, optional
        If True, the v_prior (rate parameter for the second Gamma) will incorporate
        the MLE estimate directly. If False, defaults to a moderate prior of Gamma(2.0, 0.2).

    Returns
    -------
    Dict[str, Any]
        A dictionary that can be passed as `model_config` to `GammaGammaModel`.
        It contains keys "p_prior", "q_prior", "v_prior", each mapped to a
        `pymc_marketing.prior.Prior` object.
    """

    # 1. Fit lifetimes GammaGammaFitter on frequency and monetary_value
    freq = rfm_data["frequency"].values
    monetary = rfm_data["monetary_value"].values

    ggf = GammaGammaFitter()
    ggf.fit(frequency=freq, monetary_value=monetary)

    # Extract MLE parameters p, q, v
    p_mle = ggf.params_["p"]
    q_mle = ggf.params_["q"]
    v_mle = ggf.params_["v"]  # usually a rate parameter for the distribution of nu

    # 2. Build Prior objects
    # p_prior and q_prior always come from the MLE, with shape=p_mle and shape=q_mle, respectively,
    # and a small rate=1.0 to center means at p_mle and q_mle.
    p_prior = Prior("Gamma", alpha=p_mle, beta=1.0)
    q_prior = Prior("Gamma", alpha=q_mle, beta=1.0)

    # The v_prior optionally uses the MLE defaults to a moderate Gamma(2.0, 0.2)
    if use_mle_for_v:
        # Shape=2.0, rate=v_mle => prior mean = 2.0 / v_mle
        v_prior = Prior("Gamma", alpha=2.0, beta=v_mle)
    else:
        # Fallback default: shape=2.0, rate=0.2 => prior mean=10
        v_prior = Prior("Gamma", alpha=2.0, beta=0.2)

    model_config = {
        "p_prior": p_prior,
        "q_prior": q_prior,
        "v_prior": v_prior,
    }

    return model_config
