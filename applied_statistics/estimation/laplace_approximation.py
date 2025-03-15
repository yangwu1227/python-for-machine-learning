import matplotlib.pyplot as plt
import numpy as np
import sympy as sp


def main() -> int:
    alpha, beta, x = sp.symbols("alpha beta x", positive=True)
    # Define the log-density (ignoring the normalization constant, since Laplace works with the shape)
    log_f = (alpha - 1) * sp.log(x) + (beta - 1) * sp.log(1 - x)

    # Compute the derivative with respect to x and solve for the mode
    dlog_f = sp.diff(log_f, x)
    mode_expr = sp.solve(dlog_f, x)[0]
    print("Mode (symbolic):")
    sp.pprint(mode_expr)

    # Compute the second derivative of the log-density
    d2log_f = sp.diff(log_f, x, 2)
    d2log_f_simpl = sp.simplify(d2log_f)
    print("\nSecond derivative (symbolic):\n")
    sp.pprint(d2log_f_simpl)

    # Laplace approximation variance: sigma^2 = -1 / (second derivative at the mode)
    sigma2_expr = -1 / d2log_f_simpl.subs(x, mode_expr)
    sigma2_expr = sp.simplify(sigma2_expr)
    print("\nVariance (sigma^2) (symbolic):\n")
    sp.pprint(sigma2_expr)

    # Choose specific parameter values (α > 1 and β > 1)
    alpha_val = np.random.uniform(1.5, 10.0)
    beta_val = np.random.uniform(1.5, 10.0)

    # Evaluate the mode and variance numerically
    mode_val = mode_expr.subs({alpha: alpha_val, beta: beta_val})
    sigma2_val = sigma2_expr.subs({alpha: alpha_val, beta: beta_val})
    sigma_val = sp.sqrt(sigma2_val)

    print("\nMode (numeric):", mode_val)
    print("Variance (numeric):", sigma2_val)
    print("Sigma (numeric):", sigma_val)

    # Create functions for the Beta density and the Laplace approximation
    beta_pdf_expr = (x ** (alpha - 1) * (1 - x) ** (beta - 1)) / sp.beta(alpha, beta)
    beta_pdf_expr = beta_pdf_expr.subs({alpha: alpha_val, beta: beta_val})
    beta_pdf = sp.lambdify(x, beta_pdf_expr, "numpy")

    # The Laplace approximation is a normal density with mean = mode and variance = sigma^2
    mode_num = float(mode_val)
    sigma_num = float(sigma_val)

    def laplace_pdf(x_val):
        return (
            1
            / np.sqrt(2 * np.pi * sigma_num**2)
            * np.exp(-0.5 * ((x_val - mode_num) / sigma_num) ** 2)
        )

    x_vals = np.linspace(0, 1, 400)
    beta_vals = beta_pdf(x_vals)
    laplace_vals = laplace_pdf(x_vals)

    plt.figure(figsize=(8, 5))
    plt.plot(x_vals, beta_vals, label="Beta Distribution", linewidth=2)
    plt.plot(
        x_vals,
        laplace_vals,
        label="Laplace Approximation (Normal)",
        linestyle="--",
        linewidth=2,
    )
    plt.title("Beta Distribution vs Laplace Approximation")
    plt.xlabel("x")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True)
    plt.show()

    return 0


if __name__ == "__main__":
    main()
