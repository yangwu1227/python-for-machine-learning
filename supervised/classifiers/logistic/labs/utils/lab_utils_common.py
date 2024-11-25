"""
LAB_UTILS_COMMON
Contains common routines and variable definitions
Used by all the labs in this week.
By contrast, specific, large plotting routines will be in separate files
And are generally imported into the week where they are used.
Those files will import this file.
"""

import copy
import math

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch

np.set_printoptions(precision=2)

dlc = {
    "dlblue": "#0096ff",
    "dlorange": "#FF9300",
    "dldarkred": "#C00000",
    "dlmagenta": "#FF40FF",
    "dlpurple": "#7030A0",
}

dlblue = "#0096ff"
dlorange = "#FF9300"
dldarkred = "#C00000"
dlmagenta = "#FF40FF"
dlpurple = "#7030A0"
dlcolors = [dlblue, dlorange, dldarkred, dlmagenta, dlpurple]

plt.style.use("utils/plot_style.mplstyle")


def sigmoid(z):
    """
    COMPUTE THE SIGMOID OF Z

    PARAMETERS
    ----------
    Z : ARRAY_LIKE
        A scalar or numpy array of any size.

    RETURNS
    -------
    G : ARRAY_LIKE
        sigmoid(z)
    """
    z = np.clip(z, -500, 500)  # Protect against overflow
    g = 1.0 / (1.0 + np.exp(-z))

    return g


# Regression Routines


def predict_logistic(X, w, b):
    """Performs prediction"""
    return sigmoid(X @ w + b)


def predict_linear(X, w, b):
    """Performs prediction"""
    return X @ w + b


def compute_cost_logistic(X, y, w, b, lambda_=0, safe=False):
    """
    COMPUTES COST USING LOGISTIC LOSS, NON-MATRIX VERSION

    ARGS:
      X (NDARRAY): Shape (m,n)  matrix of examples with n features
      Y (NDARRAY): Shape (m,)   target values
      W (NDARRAY): Shape (n,)   parameters for prediction
      B (SCALAR):               parameter  for prediction
      LAMBDA_ : (SCALAR, FLOAT) Controls amount of regularization, 0 = no regularization
      SAFE : (BOOLEAN)          True-selects under/overflow safe algorithm
    RETURNS:
      COST (SCALAR): COST
    """

    m, n = X.shape
    cost = 0.0
    for i in range(m):
        z_i = np.dot(X[i], w) + b  # (n,)(n,) or (n,) ()
        if safe:  # Avoids overflows
            cost += -(y[i] * z_i) + log_1pexp(z_i)
        else:
            f_wb_i = sigmoid(z_i)  # (n,)
            cost += -y[i] * np.log(f_wb_i) - (1 - y[i]) * np.log(1 - f_wb_i)  # Scalar
    cost = cost / m

    reg_cost = 0
    if lambda_ != 0:
        for j in range(n):
            # Scalar
            reg_cost += w[j] ** 2
        reg_cost = (lambda_ / (2 * m)) * reg_cost

    return cost + reg_cost


def log_1pexp(x, maximum=20):
    """
    APPROXIMATE LOG(1+EXP^X)
    https://stats.stackexchange.com/questions/475589/numerical-computation-of-cross-entropy-in-practice
    ARGS:
    X   : (NDARRAY SHAPE (n,1) OR (n,)  INPUT
    OUT : (NDARRAY SHAPE MATCHES X      OUTPUT ~= np.log(1+exp(x))
    """

    out = np.zeros_like(x, dtype=float)
    i = x <= maximum
    ni = np.logical_not(i)

    out[i] = np.log(1 + np.exp(x[i]))
    out[ni] = x[ni]
    return out


def compute_cost_matrix(X, y, w, b, logistic=False, lambda_=0, safe=True):
    """
    Computes the cost using matrix operations

    Parameters
    ----------
    X : (ndarray, Shape (m,n))          matrix of examples
    y : (ndarray  Shape (m,) or (m,1))  target value of each example
    w : (ndarray  Shape (n,) or (n,1))  Values of parameters of the model
    b : (scalar )                       Values of parameter of the model
    logistic: (boolean)                 linear if false, logistic if true
    lambda_:  (float)                   applies regularization if non-zero
    safe: (boolean)                     True-selects under/overflow safe algorithm


    Returns
    -------
    cost: (float)                       The cost of the model
    """
    m = X.shape[0]
    y = y.reshape(-1, 1)  # Ensure 2D
    w = w.reshape(-1, 1)  # Ensure 2D
    if logistic:
        if safe:  # Safe from overflow
            z = X @ w + b  # (m,n)(n,1)=(m,1)
            cost = -(y * z) + log_1pexp(z)
            # (Scalar)
            cost = np.sum(cost) / m
        else:
            # (m,n)(n,1) = (m,1)
            f = sigmoid(X @ w + b)
            cost = (1 / m) * (
                np.dot(-y.T, np.log(f)) - np.dot((1 - y).T, np.log(1 - f))
            )  # (1,m)(m,1) = (1,1)
            # Scalar
            cost = cost[0, 0]
    else:
        # (m,n)(n,1) = (m,1)
        f = X @ w + b
        # Scalar
        cost = (1 / (2 * m)) * np.sum((f - y) ** 2)

    # Scalar
    reg_cost = (lambda_ / (2 * m)) * np.sum(w**2)

    total_cost = cost + reg_cost  # Scalar

    # Scalar
    return total_cost


def compute_gradient_matrix(X, y, w, b, logistic=False, lambda_=0):
    """
    Computes the gradient using matrices.

    Parameters
    ----------
    X : ndarray
        Matrix of examples, shape (m, n).
    y : ndarray
        Target value of each example, shape (m,) or (m, 1).
    w : ndarray
        Values of parameters of the model, shape (n,) or (n, 1).
    b : scalar
        Values of parameter of the model.
    logistic : bool, optional
        Linear if False, logistic if True (default is False).
    lambda_ : float, optional
        Applies regularization if non-zero (default is 0).

    Returns
    -------
    dj_dw : array_like
        The gradient of the cost w.r.t. the parameters w, shape (n, 1).
    dj_db : scalar
        The gradient of the cost w.r.t. the parameter b.
    """
    m = X.shape[0]
    y = y.reshape(-1, 1)  # Ensure 2D.
    w = w.reshape(-1, 1)  # Ensure 2D.

    # (m, n)(n, 1) = (m, 1).
    f_wb = sigmoid(X @ w + b) if logistic else X @ w + b
    err = f_wb - y  # (m, 1).

    # (n, m)(m, 1) = (n, 1).
    dj_dw = (1 / m) * (X.T @ err)
    dj_db = (1 / m) * np.sum(err)  # Scalar.

    dj_dw += (lambda_ / m) * w  # Regularize (n, 1).

    # Scalar, (n, 1).
    return dj_db, dj_dw


def gradient_descent(
    X, y, w_in, b_in, alpha, num_iters, logistic=False, lambda_=0, verbose=True
):
    """
    Performs batch gradient descent to learn theta. Updates theta by taking
    num_iters gradient steps with learning rate alpha.

    Parameters
    ----------
    X : ndarray
        Matrix of examples, shape (m, n).
    y : ndarray
        Target value of each example, shape (m,) or (m, 1).
    w_in : ndarray
        Initial values of parameters of the model, shape (n,) or (n, 1).
    b_in : scalar
        Initial value of parameter of the model.
    alpha : float
        Learning rate.
    num_iters : int
        Number of iterations to run gradient descent.
    logistic : bool, optional
        Linear if False, logistic if True (default is False).
    lambda_ : float, optional
        Applies regularization if non-zero (default is 0).
    verbose : bool, optional
        Print cost every at intervals 10 times or as many iterations if < 10 (default is True).

    Returns
    -------
    w : ndarray
        Updated values of parameters; matches incoming shape, shape (n,) or (n, 1).
    b : scalar
        Updated value of parameter.
    J_history : list
        Cost J at each iteration.
    """
    # An array to store cost J and w's at each iteration primarily for graphing later.
    J_history = []
    w = copy.deepcopy(w_in)  # Avoid modifying global w within function.
    b = b_in
    w = w.reshape(-1, 1)  # Prep for matrix operations.
    y = y.reshape(-1, 1)

    for i in range(num_iters):
        # Calculate the gradient and update the parameters.
        dj_db, dj_dw = compute_gradient_matrix(X, y, w, b, logistic, lambda_)

        # Update Parameters using w, b, alpha and gradient.
        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        # Save cost J at each iteration.
        if i < 100000:  # Prevent resource exhaustion.
            J_history.append(compute_cost_matrix(X, y, w, b, logistic, lambda_))

        # Print cost every at intervals 10 times or as many iterations if < 10.
        if i % math.ceil(num_iters / 10) == 0:
            if verbose:
                print(f"Iteration {i:4d}: Cost {J_history[-1]}   ")

    # Return final w, b and J history for graphing.
    return w.reshape(w_in.shape), b, J_history


def zscore_normalize_features(X):
    """
    Computes X, zcore normalized by column.

    Parameters
    ----------
    X : ndarray
        Input data, m examples, n features, shape (m, n).

    Returns
    -------
    X_norm : ndarray
        Input normalized by column, shape (m, n).
    mu : ndarray
        Mean of each feature, shape (n,).
    sigma : ndarray
        Standard deviation of each feature, shape (n,).
    """
    # Find the mean of each column/feature.
    mu = np.mean(X, axis=0)  # Mu will have shape (n,).
    # Find the standard deviation of each column/feature.
    sigma = np.std(X, axis=0)  # Sigma will have shape (n,).
    # Element-wise, subtract mu for that column from each example, divide by std for that column.
    X_norm = (X - mu) / sigma

    return X_norm, mu, sigma


# ----------------------------- Plotting routines ---------------------------- #


def plot_data(X, y, ax, pos_label="y=1", neg_label="y=0", s=80, loc="best"):
    """plots logistic data with two axis"""
    # Find Indices of Positive and Negative Examples
    pos = y == 1
    neg = y == 0
    pos = pos.reshape(
        -1,
    )  # work with 1D or 1D y vectors
    neg = neg.reshape(
        -1,
    )

    # Plot examples
    ax.scatter(X[pos, 0], X[pos, 1], marker="x", s=s, c="red", label=pos_label)
    ax.scatter(
        X[neg, 0],
        X[neg, 1],
        marker="o",
        s=s,
        label=neg_label,
        facecolors="none",
        edgecolors=dlblue,
        lw=3,
    )
    ax.legend(loc=loc)

    ax.figure.canvas.toolbar_visible = False
    ax.figure.canvas.header_visible = False
    ax.figure.canvas.footer_visible = False


def plt_tumor_data(x, y, ax):
    """plots tumor data on one axis"""
    pos = y == 1
    neg = y == 0

    ax.scatter(x[pos], y[pos], marker="x", s=80, c="red", label="malignant")
    ax.scatter(
        x[neg],
        y[neg],
        marker="o",
        s=100,
        label="benign",
        facecolors="none",
        edgecolors=dlblue,
        lw=3,
    )
    ax.set_ylim(-0.175, 1.1)
    ax.set_ylabel("y")
    ax.set_xlabel("Tumor Size")
    ax.set_title("Logistic Regression on Categorical Data")

    ax.figure.canvas.toolbar_visible = False
    ax.figure.canvas.header_visible = False
    ax.figure.canvas.footer_visible = False


def draw_vthresh(ax, x):
    """draws a threshold"""
    ylim = ax.get_ylim()
    xlim = ax.get_xlim()
    ax.fill_between([xlim[0], x], [ylim[1], ylim[1]], alpha=0.2, color=dlblue)
    ax.fill_between([x, xlim[1]], [ylim[1], ylim[1]], alpha=0.2, color=dldarkred)
    ax.annotate(
        "z >= 0",
        xy=[x, 0.5],
        xycoords="data",
        xytext=[30, 5],
        textcoords="offset points",
    )
    d = FancyArrowPatch(
        posA=(x, 0.5),
        posB=(x + 3, 0.5),
        color=dldarkred,
        arrowstyle="simple, head_width=5, head_length=10, tail_width=0.0",
    )
    ax.add_artist(d)
    ax.annotate(
        "z < 0",
        xy=[x, 0.5],
        xycoords="data",
        xytext=[-50, 5],
        textcoords="offset points",
        ha="left",
    )
    f = FancyArrowPatch(
        posA=(x, 0.5),
        posB=(x - 3, 0.5),
        color=dlblue,
        arrowstyle="simple, head_width=5, head_length=10, tail_width=0.0",
    )
    ax.add_artist(f)
