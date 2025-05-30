import numpy as np
import polars as pl
from scipy.stats import vonmises

seed = 12
rng = np.random.default_rng(seed)
n_trials = 8
n_tx = 30  # Transactions per trial


def main() -> int:
    rows = []
    for trial in range(1, n_trials + 1):
        kappa = rng.uniform(0.5, 6.0)
        loc = rng.uniform(-np.pi, np.pi)

        radians = vonmises.rvs(kappa, loc=loc, size=n_tx, random_state=rng)

        latest = radians[-1]

        # Representation A: [0, 2π)
        latest_mod_2pi = latest % (2 * np.pi)

        # Representation B: (-π, π]
        latest_pm_pi = ((latest_mod_2pi + np.pi) % (2 * np.pi)) - np.pi

        print(f"Trial {trial}:")
        print(f"  kappa: {kappa}")
        print(f"  loc: {loc}")
        print(f"  Latest angle [0, 2π): {latest_mod_2pi}")
        print(f"  Latest angle (-π, π]: {latest_pm_pi}")

        # Evaluate the PDF for both representations
        pdf_2pi = vonmises.pdf(latest_mod_2pi, kappa, loc=loc)
        pdf_pm = vonmises.pdf(latest_pm_pi, kappa, loc=loc)

        rows.append(
            {
                "trial": trial + 1,
                "kappa": round(kappa, 3),
                "loc": round(loc, 3),
                "latest_[0,2π)": round(latest_mod_2pi, 5),
                "latest_(−π,π]": round(latest_pm_pi, 5),
                "pdf_[0,2π)": pdf_2pi,
                "pdf_(−π,π]": pdf_pm,
                "difference": abs(pdf_2pi - pdf_pm),
            }
        )

    data = pl.from_dicts(rows)

    print(data)

    return 0


if __name__ == "__main__":
    main()
