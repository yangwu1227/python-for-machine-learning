import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats


def main() -> int:
    # Define a range of prior parameters (alpha, beta) for the Beta distribution
    prior_params = [
        (1, 1),  # Uniform prior
        (2, 2),  # Weakly informative prior
        (0.5, 0.5),  # Informative prior favoring extremes
        (5, 1),  # Informative prior favoring high probabilities
        (1, 5),  # Informative prior favoring low probabilities
    ]

    # Define different observed data scenarios (number of successes and trials)
    data_scenarios = [
        (1, 1),  # Very little data
        (5, 10),  # Moderate data
        (20, 40),  # More data
        (50, 100),  # Large dataset
    ]

    fig, axes = plt.subplots(
        len(prior_params),
        len(data_scenarios),
        figsize=(15, 12),
        sharex=True,
        sharey=True,
    )
    x = np.linspace(0, 1, 1000)

    for i, (alpha_prior, beta_prior) in enumerate(prior_params):
        for j, (successes, trials) in enumerate(data_scenarios):
            print(
                f"Prior: Beta({alpha_prior}, {beta_prior}), Data: {successes}/{trials}"
            )
            # Compute posterior parameters
            alpha_post = alpha_prior + successes
            beta_post = beta_prior + (trials - successes)
            print(f"Posterior: Beta({alpha_post}, {beta_post})")

            # Compute prior and posterior distributions
            prior_pdf = stats.beta.pdf(x, alpha_prior, beta_prior)
            posterior_pdf = stats.beta.pdf(x, alpha_post, beta_post)

            ax = axes[i, j]
            ax.plot(x, prior_pdf, "r--", label="Prior")
            ax.plot(x, posterior_pdf, "b-", label="Posterior")
            ax.fill_between(x, posterior_pdf, color="blue", alpha=0.2)

            if j == 0:
                ax.set_ylabel(f"Beta({alpha_prior}, {beta_prior})")
            if i == len(prior_params) - 1:
                ax.set_xlabel(f"{successes}/{trials} successes")

            ax.legend()

    fig.suptitle(
        "Impact of Different Priors on the Posterior in a Beta-Bernoulli Model",
        fontsize=16,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

    return 0


if __name__ == "__main__":
    main()
