"""
Lab_utils_common
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
from ipywidgets import Output
from matplotlib.patches import FancyArrowPatch
from matplotlib.widgets import Button, CheckButtons

np.set_printoptions(precision=2)

dlc = {
    "dlblue": "#0096ff",
    "dlorange": "#FF9300",
    "dldarkred": "#C00000",
    "dlmagenta": "#FF40FF",
    "dlpurple": "#7030A0",
    "dldarkblue": "#0D5BDC",
    "dlmedblue": "#4285F4",
}

dlblue = "#0096ff"
dlorange = "#FF9300"
dldarkred = "#C00000"
dlmagenta = "#FF40FF"
dlpurple = "#7030A0"
dldarkblue = "#0D5BDC"
dlmedblue = "#4285F4"

dlcolors = [dlblue, dlorange, dldarkred, dlmagenta, dlpurple]

plt.style.use("utils/deeplearning.mplstyle")


def sigmoid(z):
    """
    Compute the sigmoid of z

    Parameters
    ----------
    z : array_like
        A scalar or numpy array of any size.

    Returns
    -------
     g : array_like
         sigmoid(z)
    """
    z = np.clip(z, -500, 500)  # protect against overflow
    g = 1.0 / (1.0 + np.exp(-z))

    return g


def predict_logistic(X, w, b):
    """Performs prediction."""
    return sigmoid(X @ w + b)


def predict_linear(X, w, b):
    """Performs prediction."""
    return X @ w + b


def compute_cost_logistic(X, y, w, b, lambda_=0, safe=False):
    """
    Computes cost using logistic loss, non-matrix version.

    Args:
      X (ndarray): Shape (m,n)  matrix of examples with n features
      y (ndarray): Shape (m,)   target values
      w (ndarray): Shape (n,)   parameters for prediction
      b (scalar):               parameter  for prediction
      lambda_ : (scalar, float) Controls amount of regularization, 0 = no regularization
      safe : (boolean)          True-selects under/overflow safe algorithm

    Returns:
      cost (scalar): cost
    """
    m, n = X.shape
    cost = 0.0
    for i in range(m):
        z_i = np.dot(X[i], w) + b
        if safe:  # avoids overflows
            cost += -(y[i] * z_i) + log_1pexp(z_i)
        else:
            f_wb_i = sigmoid(z_i)
            cost += -y[i] * np.log(f_wb_i) - (1 - y[i]) * np.log(1 - f_wb_i)
    cost = cost / m

    reg_cost = 0
    if lambda_ != 0:
        for j in range(n):
            reg_cost += w[j] ** 2
        reg_cost = (lambda_ / (2 * m)) * reg_cost

    return cost + reg_cost


def log_1pexp(x, maximum=20):
    """
    Approximate log(1+exp^x).
    https://stats.stackexchange.com/questions/475589/numerical-computation-of-cross-entropy-in-practice

    Args:
    x   : (ndarray Shape (n,1) or (n,)  input
    out : (ndarray Shape matches x      output ~= np.log(1+exp(x))
    """
    out = np.zeros_like(x, dtype=float)
    i = x <= maximum
    ni = np.logical_not(i)

    out[i] = np.log(1 + np.exp(x[i]))
    out[ni] = x[ni]
    return out


def compute_cost_matrix(X, y, w, b, logistic=False, lambda_=0, safe=True):
    """
    Computes the cost using using matrices.

    Args:
      X : (ndarray, Shape (m,n))          matrix of examples
      y : (ndarray  Shape (m,) or (m,1))  target value of each example
      w : (ndarray  Shape (n,) or (n,1))  Values of parameter(s) of the model
      b : (scalar )                       Values of parameter of the model
      verbose : (Boolean) If true, print out intermediate value f_wb

    Returns:
      total_cost: (scalar)                cost
    """
    m = X.shape[0]
    y = y.reshape(-1, 1)  # Ensure 2D
    w = w.reshape(-1, 1)  # Ensure 2D
    if logistic:
        if safe:  # Safe from overflow
            z = X @ w + b
            cost = -(y * z) + log_1pexp(z)
            cost = np.sum(cost) / m
        else:
            f = sigmoid(X @ w + b)
            cost = (1 / m) * (
                np.dot(-y.T, np.log(f)) - np.dot((1 - y).T, np.log(1 - f))
            )
            cost = cost[0, 0]
    else:
        f = X @ w + b
        cost = (1 / (2 * m)) * np.sum((f - y) ** 2)

    reg_cost = (lambda_ / (2 * m)) * np.sum(w**2)
    total_cost = cost + reg_cost

    return total_cost
    # scalar


def compute_gradient_matrix(X, y, w, b, logistic=False, lambda_=0):
    """
    Computes the gradient using matrices.

    Args:
      X : (ndarray, Shape (m,n))          matrix of examples
      y : (ndarray  Shape (m,) or (m,1))  target value of each example
      w : (ndarray  Shape (n,) or (n,1))  Values of parameters of the model
      b : (scalar )                       Values of parameter of the model
      logistic: (boolean)                 linear if false, logistic if true
      lambda_:  (float)                   applies regularization if non-zero

    Returns:
      dj_dw: (array_like Shape (n,1))     The gradient of the cost w.r.t. the parameters w
      dj_db: (scalar)                     The gradient of the cost w.r.t. the parameter b
    """
    m = X.shape[0]
    y = y.reshape(-1, 1)  # Ensure 2D
    w = w.reshape(-1, 1)  # Ensure 2D

    f_wb = sigmoid(X @ w + b) if logistic else X @ w + b
    err = f_wb - y
    dj_dw = (1 / m) * (X.T @ err)
    dj_db = (1 / m) * np.sum(err)

    dj_dw += (lambda_ / m) * w  # Regularize

    return dj_db, dj_dw


def gradient_descent(
    X,
    y,
    w_in,
    b_in,
    alpha,
    num_iters,
    logistic=False,
    lambda_=0,
    verbose=True,
    Trace=True,
):
    """
    Performs batch gradient descent to learn theta. Updates theta by taking
    num_iters gradient steps with learning rate alpha.

    Args:
      X (ndarray):    Shape (m,n)         matrix of examples
      y (ndarray):    Shape (m,) or (m,1) target value of each example
      w_in (ndarray): Shape (n,) or (n,1) Initial values of parameters of the model
      b_in (scalar):                      Initial value of parameter of the model
      logistic: (boolean)                 linear if false, logistic if true
      lambda_:  (float)                   applies regularization if non-zero
      alpha (float):                      Learning rate
      num_iters (int):                    number of iterations to run gradient descent

    Returns:
      w (ndarray): Shape (n,) or (n,1)    Updated values of parameters; matches incoming shape
      b (scalar):                         Updated value of parameter
    """
    # An array to store cost J and w's at each iteration primarily for graphing later
    J_history = []
    w = copy.deepcopy(w_in)  # Avoid modifying global w within function
    b = b_in
    w = w.reshape(-1, 1)  # Prep for matrix operations
    y = y.reshape(-1, 1)
    last_cost = np.Inf

    for i in range(num_iters):
        # Calculate the gradient and update the parameters
        dj_db, dj_dw = compute_gradient_matrix(X, y, w, b, logistic, lambda_)

        # Update Parameters using w, b, alpha and gradient
        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        # Save cost J at each iteration
        ccost = compute_cost_matrix(X, y, w, b, logistic, lambda_)
        if Trace and i < 100000:  # Prevent resource exhaustion
            J_history.append(ccost)

        # Print cost every at intervals 10 times or as many iterations if < 10
        if i % math.ceil(num_iters / 10) == 0:
            if verbose:
                print(f"Iteration {i:4d}: Cost {ccost}")
            if verbose == 2:
                print(f"dj_db, dj_dw = {dj_db: 0.3f}, {dj_dw.reshape(-1)}")

            if ccost == last_cost:
                alpha = alpha / 10
                print(f"alpha now {alpha}")
            last_cost = ccost

    return (
        w.reshape(w_in.shape),
        b,
        J_history,
    )  # Return final w, b and J history for graphing


def zscore_normalize_features(X):
    """
    Computes X, zcore normalized by column.

    Args:
      X (ndarray): Shape (m,n) input data, m examples, n features.

    Returns:
      X_norm (ndarray): Shape (m,n)  input normalized by column.
      mu (ndarray):     Shape (n,)   mean of each feature.
      sigma (ndarray):  Shape (n,)   standard deviation of each feature.
    """
    # Find the mean of each column/feature
    mu = np.mean(X, axis=0)  # mu will have shape (n,)
    # Find the standard deviation of each column/feature
    sigma = np.std(X, axis=0)  # sigma will have shape (n,)
    # Element-wise, subtract mu for that column from each example, divide by std for that column
    X_norm = (X - mu) / sigma

    return X_norm, mu, sigma


def plot_data(X, y, ax, pos_label="y=1", neg_label="y=0", s=80, loc="best"):
    """Plots logistic data with two axis."""
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
    """Plots tumor data on one axis."""
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
    """Draws a threshold."""
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


class ButtonManager:
    """Handles some missing features of matplotlib check buttons.

    On init:
        Creates button, links to button_click routine,
        calls call_on_click with active index and firsttime=True.
    On click:
        Maintains single button on state, calls call_on_click.
    """

    def __init__(self, fig, dim, labels, init, call_on_click):
        """
        Initializes the ButtonManager.

        Parameters
        ----------
        fig : matplotlib.figure.Figure
            The figure object.
        dim : list
            The dimensions [leftbottom_x, bottom_y, width, height].
        labels : list
            The labels for the buttons, e.g., ['1', '2', '3', '4', '5', '6'].
        init : list
            The initial state of the buttons, e.g., [True, False, False, False, False, False].
        call_on_click : callable
            The function to call when a button is clicked.
        """
        self.fig = fig
        self.ax = plt.axes(dim)  # lx, by, w, h
        self.init_state = init
        self.call_on_click = call_on_click
        self.button = CheckButtons(self.ax, labels, init)
        self.button.on_clicked(self.button_click)
        self.status = self.button.get_status()
        self.call_on_click(self.status.index(True), firsttime=True)

    def reinit(self):
        """Reinitializes the button manager to its initial state."""
        self.status = self.init_state
        self.button.set_active(
            self.status.index(True)
        )  # turn off old, will trigger update and set to status

    def button_click(self, event):
        """Maintains one-on state. If on-button is clicked, will process correctly."""
        self.button.eventson = False
        self.button.set_active(
            self.status.index(True)
        )  # turn off old or reenable if same
        self.button.eventson = True
        self.status = self.button.get_status()
        self.call_on_click(self.status.index(True))
