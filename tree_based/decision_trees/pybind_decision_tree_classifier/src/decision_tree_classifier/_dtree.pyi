from typing import List, Literal, overload

import numpy as np
from numpy.typing import NDArray

class DecisionTreeClassifier:
    """
    Implementation of a Decision Tree Classifier with multiclass support.
    """
    def __init__(
        self,
        max_depth: int = 3,
        min_samples_split: int = 2,
        criterion: Literal["gini", "entropy"] = "gini",
        min_impurity_decrease: float = 1e-7,
    ) -> None: ...
    @overload
    def fit(self, X: List[List[float]], y: List[int]) -> None: ...
    @overload
    def fit(self, X: NDArray[np.float64], y: NDArray[np.int32]) -> None: ...
    @overload
    def predict(self, X: List[List[float]]) -> List[int]: ...
    @overload
    def predict(self, X: NDArray[np.float64]) -> NDArray[np.int32]: ...
