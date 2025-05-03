import logging
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest
import tensorflow as tf
from IPython.display import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model_utils import (
    AugmentationModel,
    ErrorAnalyzer,
    TuningVisualizer,
    setup_logger,
)

# ------------------------- Tests for logger function ------------------------ #


class TestGetLogger:
    """
    Tests for the logger function.
    """

    def test_setup_logger(self, capsys):
        """
        Tests for name, level, and message
        """
        # Test that the logger has the correct name
        logger = setup_logger("test")
        assert logger.name == "test"

        # Test that the logger has the correct level
        assert logger.level == logging.INFO

        # Test that the logger has the correct handler
        assert isinstance(logger.handlers[0], logging.StreamHandler)

        # Test that the logger has the correct format
        assert (
            logger.handlers[0].formatter._fmt
            == "%(asctime)s %(levelname)s %(name)s: %(message)s"
        )

        # Test that the logger correctly logs a message
        logger.info("Info message")
        captured = capsys.readouterr()
        assert "Info message" in captured.out

        # Test that the logger does not log a debug message (because the level is set to INFO)
        logger.debug("Debug message")
        captured = capsys.readouterr()
        assert "Debug message" not in captured.out


# ----------------------- Tests for augmentation layer ----------------------- #


@pytest.fixture(scope="class")
def augmentation_layers():
    """
    Fixture for the augmentation layer.
    """
    return {
        "valid_dict_input": {
            "RandomFlip": {"mode": "horizontal"},
            "RandomRotation": {"factor": 0.5},
            "RandomZoom": {"height_factor": 0.5, "width_factor": 0.5},
            "RandomContrast": {"factor": 0.5},
        },
        "invalid_non_dict_input": (
            "RandomFlip",
            "RandomRotation",
            "RandomZoom",
            "RandomContrast",
        ),
        "invalid_layer_name": {"InvalidLayer": {"mode": "horizontal"}},
        "invalid_layer_config": {"RandomFlip": {"invalid_config": "horizontal"}},
    }


class TestAugmentationLayer:
    """
    Tests for the AugmentationModel class.
    """

    def test_constructor(self, augmentation_layers):
        """
        Tests for the constructor.
        """
        # Test that the constructor accepts a valid dictionary input
        augmentation_model = AugmentationModel(
            aug_params=augmentation_layers["valid_dict_input"]
        )
        assert isinstance(augmentation_model, AugmentationModel)
        assert isinstance(augmentation_model.base_model, tf.keras.Sequential)
        assert isinstance(augmentation_model.aug_params, dict)

    def test_exceptions(self, augmentation_layers):
        """
        Tests for exceptions.
        """
        # Non-dict object should raise TypeError when the setter method is called
        with pytest.raises(TypeError):
            AugmentationModel(
                aug_params=augmentation_layers["invalid_non_dict_input"]
            ).build_augmented_model()

        # Keras raises an exception if the layer name is invalid
        with pytest.raises(ValueError):
            AugmentationModel(
                aug_params=augmentation_layers["invalid_layer_name"]
            ).build_augmented_model()

        # Keras raises an exception if the layer config is invalid
        with pytest.raises(TypeError):
            AugmentationModel(
                aug_params=augmentation_layers["invalid_layer_config"]
            ).build_augmented_model()


# ------------------- Tests for the TuningVisualizer class ------------------- #


@pytest.fixture(scope="class")
def visualizer():
    """
    Valid TuningVisualizer object.
    """
    tune_data = pd.DataFrame(
        {
            "param1": ["low", "high", "low", "high"],
            "param2": [0.1, 0.2, 0.3, 0.4],
            "FinalObjectiveValue": [0.8, 0.9, 0.7, 0.6],
        }
    )
    num_params = ["param2"]
    cat_params = ["param1"]
    return TuningVisualizer(
        tune_data=tune_data, num_params=num_params, cat_params=cat_params
    )


class TestTuningVisualizer:
    @pytest.mark.parametrize(
        "tune_data, num_params, cat_params",
        [
            # Case 1: tune_data is not a DataFrame
            (pd.Series([1, 2, 3]), ["param1"], ["param1"]),
            # Case 2: num_params is not a list
            (
                pd.DataFrame({"param1": ["low", "high", "low", "high"]}),
                ("param1", "param2"),
                ["param2"],
            ),
            # Case 3: cat_params is not a list
            (
                pd.DataFrame({"param1": ["low", "high", "low", "high"]}),
                ["param1"],
                ("param1", "param2"),
            ),
            # Case 4: num_params accidently inputed as indices
            (
                pd.DataFrame({"param1": ["low", "high", "low", "high"]}),
                [1, 2, 3],
                ["param1"],
            ),
            # Case 5: cat_params accidently inputed as indices
            (
                pd.DataFrame({"param1": ["low", "high", "low, high", "high"]}),
                ["param1"],
                [1, 2, 3],
            ),
        ],
        scope="function",
    )
    def test_properties_type(self, tune_data, num_params, cat_params):
        with pytest.raises(TypeError):
            TuningVisualizer(
                tune_data=tune_data, num_params=num_params, cat_params=cat_params
            )

    def test_cat_mapping(self, visualizer):
        """
        Tests for the _cat_mapping method.
        """
        # Test that the method returns the correct mapping (in a list)
        assert visualizer._cat_mapping(cat_params=visualizer.cat_params) == [
            {"low": 0, "high": 1}
        ]

        # If user accidently inputed a numerical feature as categorical, the method should raise a TypeError
        with pytest.raises(TypeError):
            visualizer._cat_mapping(cat_params=visualizer.num_params)

    def test_plot_parallel_coordinate(self, visualizer):
        """
        Tests for the plot_parallel_coordinate method.
        """
        # Test that the method returns a plotly figure in interactive mode
        assert isinstance(
            visualizer.plot_parallel_coordinate(
                columns=visualizer.num_params + visualizer.cat_params
            ),
            go.Figure,
        )

        # Test that the method returns an Image object in non-interactive mode
        assert isinstance(
            visualizer.plot_parallel_coordinate(
                columns=visualizer.num_params + visualizer.cat_params, static=True
            ),
            Image,
        )


# ------------------------- Tests for error analyzer ------------------------- #


@pytest.fixture(scope="class")
def test_data():
    return np.load("tests/test_data/val_data.npz")


@pytest.fixture(scope="class")
def label_map():
    return {
        "backyard": 0,
        "bathroom": 1,
        "bedroom": 2,
        "diningRoom": 3,
        "frontyard": 4,
        "hall": 5,
        "kitchen": 6,
        "livingRoom": 7,
        "plan": 8,
    }


class TestErrorAnalyzer:
    """
    Test the plotting functionality of the Error Analyzer class.
    """

    @pytest.mark.parametrize(
        "class_label, path", [("bathroom", "test_plots/bathroom.png")], scope="function"
    )
    def test_plot(self, test_data, label_map, class_label, path):
        """
        Test the plot method.
        """
        # Test that the method returns a plotly figure in interactive mode
        error_analyzer = ErrorAnalyzer(
            test_data["y_val"], test_data["X_val"], test_data["y_pred"], label_map
        )
        error_analyzer.plot_mis_clf(class_label, sample_mis_clf=1)
        plt.savefig(path)
