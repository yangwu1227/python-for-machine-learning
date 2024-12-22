from typing import Any, Dict, MutableSequence, Optional, Self, Union

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.estimator_checks import check_estimator
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
from tqdm import tqdm


def custom_cosine_similarity(sample: np.ndarray, X: np.ndarray) -> np.ndarray:
    """
    Compute the cosine similarity between a sample and all samples in a dataset.

    Parameters
    ----------
    sample : np.ndarray
        A single sample as a row vector with dimensions (1, n_features).
    X : np.ndarray
        The training data matrix with dimensions (n_samples, n_features).

    Returns
    -------
    np.ndarray
        The cosine similarities between the query sample and all samples in the training data matrix.
    """
    # Compute the dot products between the sample and all samples in the dataset
    # The result is a row vector with n_samples elements
    dot_products = np.dot(X, sample.T)

    # Compute the L2 norm of the sample and all samples in the dataset
    # This is a scalar value
    sample_norm = np.linalg.norm(sample)

    # The axis=1 argument ensures that the norm is computed along the feature axis
    # X_norm is a row vector with n_samples elements
    X_norm = np.linalg.norm(X, axis=1)

    # The broadcasting of sample_norm and X_norm is handled automatically
    cosine_similarities = dot_products / (sample_norm * X_norm + 1e-8)

    return cosine_similarities


def custom_euclidean_distances(sample: np.ndarray, X: np.ndarray) -> np.ndarray:
    """
    Compute the Euclidean distance between a sample and all samples in a dataset.

    Parameters
    ----------
    sample : np.ndarray
        A single sample as a row vector with dimensions (1, n_features).
    X : np.ndarray
        The training data matrix with dimensions (n_samples, n_features).

    Returns
    -------
    np.ndarray
        The Euclidean distances between the query sample and all samples in the training data matrix.
    """
    # X is (n_samples, n_features) and sample is (1, n_features), and so the subtraction is broadcasted along the first dimension (i.e., each row of X is subtracted by sample)
    # The result is a matrix with dimensions (n_samples, n_features)
    differences = X - sample
    squared_differences = differences**2

    # Sum across the feature axis to obtain the sum of squared differences for each sample
    # The result is a row vector with n_samples elements
    squared_distances = np.sum(squared_differences, axis=1)
    euclidean_distances = np.sqrt(squared_distances)

    return euclidean_distances


class KNNClassifier(BaseEstimator, ClassifierMixin):
    """
    K-Nearest Neighbors classifier.
    """

    def __init__(
        self,
        n_neighbors: int = 5,
        weights: str = "distance",
        metric: str = "cosine",
        metric_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize the KNN classifier instance.

        Parameters
        ----------
        n_neighbors : int
            The number of nearest neighbors to consider.
        weights : str
            The weight function to use in prediction. Possible values are 'uniform' and 'distance'.
                - 'uniform' : All points in each neighborhood are weighted equally.
                - 'distance' : Weight points by the inverse of their distance.
        metric : str
            The similarity function to use. Possible values are 'cosine' and 'euclidean'.
        metric_params : Dict[str, Any]
            Additional keyword arguments for the metric function. See the documentation of scipy.spatial.distance and the metrics
            listed in distance_metrics for valid metric values.

        Returns
        -------
        None
        """
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.metric = metric
        self.metric_params = metric_params

    def fit(
        self,
        X: Union[csr_matrix, np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series, MutableSequence[Union[str, float]]],
    ) -> Self:
        """
        The KNN classifier does not learn a model. Instead, it simply stores the training data. It is a 'lazy' learner.

        Parameters
        ----------
        X : Union[csr_matrix, np.ndarray, pd.DataFrame]
            The input data.
        y : Union[np.ndarray, pd.Series, MutableSequence[Union[str, float]]]
            The target labels.

        Returns
        -------
        Self
            The instance itself.
        """
        X, y = check_X_y(
            X=X,
            y=y,
            accept_sparse="csr",
            ensure_2d=True,
            allow_nd=False,
            y_numeric=False,
        )
        check_classification_targets(y)

        # Attributes that have been estimated from the data must end with a trailing underscore
        self.X_train_ = X
        self.label_encoder_ = LabelEncoder()
        self.y_train_ = self.label_encoder_.fit_transform(y)
        self.classes_ = self.label_encoder_.classes_  # Unique classes seen during fit
        self.n_features_in_ = X.shape[1]  # Number of features seen during fit

        return self

    def predict(self, X: Union[csr_matrix, np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Predict the labels for the test set using the KNN algorithm.

        Parameters
        ----------
        X : Union[csr_matrix, np.ndarray, pd.DataFrame]
            The input data.

        Returns
        -------
        np.ndarray
            The predicted labels.
        """
        check_is_fitted(
            self,
            ["X_train_", "y_train_", "label_encoder_", "classes_", "n_features_in_"],
        )
        X = check_array(X, accept_sparse="csr")

        # Handle edge case: only one class in training data
        if len(self.classes_) == 1:
            # The classes_ attribute was already in the format before encoding, so we can return it directly
            return np.full(shape=X.shape[0], fill_value=self.classes_[0])

        predictions = np.zeros(X.shape[0], dtype=int)
        for i, test_sample in enumerate(tqdm(X)):
            # Calculate distances to all training samples
            test_sample_reshaped = test_sample.reshape(1, -1)
            distances = self._compute_distances(test_sample_reshaped)
            # Identify the indices of the k nearest neighbors
            k_indices = np.argsort(distances, axis=-1)[: self.n_neighbors]
            # Predict class based on nearest neighbors
            prediction = self._predict_from_neighbors(k_indices, distances)
            predictions[i] = prediction

        return self.label_encoder_.inverse_transform(predictions)

    def _compute_distances(
        self, test_sample: Union[np.ndarray, csr_matrix]
    ) -> np.ndarray:
        """
        Compute the distances of the test sample to all training samples.

        Parameters
        ----------
        test_sample : Union[np.ndarray, csr_matrix]
            A single test sample.

        Returns
        -------
        distances : np.ndarray
            Distances from the test sample to each training sample.
        """
        metric_params = {} if self.metric_params is None else self.metric_params
        if self.metric == "cosine":
            return (
                1
                - cosine_similarity(
                    X=test_sample, Y=self.X_train_, **metric_params
                ).flatten()
            )
        elif self.metric == "euclidean":
            return euclidean_distances(
                X=test_sample, Y=self.X_train_, **metric_params
            ).flatten()
        else:
            raise NotImplementedError(
                f"Unsupported similarity function '{self.metric}'"
            )

    def _predict_from_neighbors(
        self, k_indices: np.ndarray, distances: np.ndarray
    ) -> int:
        """
        Predict the class of a sample based on its nearest neighbors.

        Parameters
        ----------
        k_indices : np.ndarray
            Indices of the k nearest neighbors.
        distances : np.ndarray
            Distances from the sample to each neighbor.

        Returns
        -------
        prediction : int
            Predicted class label.
        """
        k_nearest_labels = self.y_train_[k_indices]
        if self.weights == "distance":
            # Inverse distance weighting, considering a small epsilon to avoid division by zero
            inv_distances = 1 / (distances[k_indices] + 1e-8)
            weighted_votes = np.bincount(
                k_nearest_labels, weights=inv_distances, minlength=len(self.classes_)
            )
        elif self.weights == "uniform":
            # Uniform weights (each neighbor contributes equally)
            weighted_votes = np.bincount(k_nearest_labels, minlength=len(self.classes_))
        else:
            raise ValueError(f"Unsupported weight function '{self.weights}'")

        return int(np.argmax(weighted_votes))


def main() -> int:
    knn_classifier = KNNClassifier()
    check_estimator(knn_classifier)

    print(
        "\nKNNClassifier class is a valid estimator compatible with scikit-learn API!"
    )

    return 0


if __name__ == "__main__":
    main()
