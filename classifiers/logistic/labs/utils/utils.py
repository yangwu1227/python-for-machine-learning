import numpy as np
import matplotlib.pyplot as plt


def load_data(filename):
    """Load data from a file.

    Parameters
    ----------
    filename : str
        The name of the file.

    Returns
    -------
    tuple
        A tuple containing the features and labels.

    """
    data = np.loadtxt(filename, delimiter=",")
    X = data[:, :2]
    y = data[:, 2]
    return X, y


def sig(z):
    """Calculate the sigmoid function.

    Parameters
    ----------
    z : array_like
        The input to the sigmoid function.

    Returns
    -------
    array_like
        The sigmoid of the input.

    """
    return 1 / (1 + np.exp(-z))


def map_feature(X1, X2):
    """Map features to a polynomial.

    Parameters
    ----------
    X1 : array_like
        The first feature.
    X2 : array_like
        The second feature.

    Returns
    -------
    array_like
        The polynomial features.

    """
    X1 = np.atleast_1d(X1)
    X2 = np.atleast_1d(X2)
    degree = 6
    out = []
    for i in range(1, degree + 1):
        for j in range(i + 1):
            out.append((X1 ** (i - j) * (X2**j)))
    return np.stack(out, axis=1)


def plot_data(X, y, pos_label="y=1", neg_label="y=0"):
    """Plot the data.

    Parameters
    ----------
    X : array_like
        The features.
    y : array_like
        The labels.
    pos_label : str, optional
        The label for positive examples.
    neg_label : str, optional
        The label for negative examples.

    """
    positive = y == 1
    negative = y == 0

    # Plot examples
    plt.plot(X[positive, 0], X[positive, 1], "k+", label=pos_label)
    plt.plot(X[negative, 0], X[negative, 1], "yo", label=neg_label)


def plot_decision_boundary(w, b, X, y):
    """Plot the decision boundary.

    Parameters
    ----------
    w : array_like
        The weights.
    b : float
        The bias.
    X : array_like
        The features.
    y : array_like
        The labels.

    """
    # Credit to dibgerge on Github for this plotting code

    plot_data(X[:, 0:2], y)

    if X.shape[1] <= 2:
        plot_x = np.array([min(X[:, 0]), max(X[:, 0])])
        plot_y = (-1.0 / w[1]) * (w[0] * plot_x + b)

        plt.plot(plot_x, plot_y, c="b")

    else:
        u = np.linspace(-1, 1.5, 50)
        v = np.linspace(-1, 1.5, 50)

        z = np.zeros((len(u), len(v)))

        # Evaluate z = theta*x over the grid
        for i in range(len(u)):
            for j in range(len(v)):
                z[i, j] = sig(np.dot(map_feature(u[i], v[j]), w) + b)

        # important to transpose z before calling contour
        z = z.T

        # Plot z = 0.5
        plt.contour(u, v, z, levels=[0.5], colors="g")
