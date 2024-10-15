import os
import joblib
import io

from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np

# -------------------------------- Load model -------------------------------- #


def model_fn(model_dir: str) -> Pipeline:
    """
    Function to load the model from the model_dir directory.

    Parameters
    ----------
    model_dir : str
        Directory where the model is saved.

    Returns
    -------
    Pipeline
        The model (Scikit-learn pipeline) object loaded in memory.
    """
    model_pipeline = joblib.load(os.path.join(model_dir, "best-model.joblib"))
    return model_pipeline


# -------------------------------- Input data -------------------------------- #


def input_fn(input_data: bytes, content_type: str) -> pd.DataFrame:
    """
    When an InvokeEndpoint operation is made against an Endpoint running a SageMaker Scikit-learn model server,
    the model server receives two pieces of information: 1) the request data body, a byte array and 2) the request
    content-type. The SageMaker Scikit-learn model server will invoke an “input_fn” function in this hosting script,
    passing in these two pieces of information.

    We implement this to override the default since the first step of the custom data preprocessing pipeline requires
    dataframe inputs (no support numpy).

    Parameters
    ----------
    input_data : bytes
        The payload of the incoming request.
    content_type : str
        The content type of the request.

    Returns
    -------
    pd.DataFrame
        Pandas DataFrame to be passed to predict_fn, which internally calls the hosted model pipeline's predict method.
    """
    # The default for SKLearnPredictor is to serialize input data to '.npy' format
    npy_array = np.load(io.BytesIO(input_data), allow_pickle=True)

    # Convert numpy array to pandas DataFrame
    data = pd.DataFrame(npy_array)

    # Feature names
    feature_names = [
        "AB",
        "AF",
        "AH",
        "AM",
        "AR",
        "AX",
        "AY",
        "AZ",
        "BC",
        "BD",
        "BN",
        "BP",
        "BQ",
        "BR",
        "BZ",
        "CB",
        "CC",
        "CD",
        "CF",
        "CH",
        "CL",
        "CR",
        "CS",
        "CU",
        "CW",
        "DA",
        "DE",
        "DF",
        "DH",
        "DI",
        "DL",
        "DN",
        "DU",
        "DV",
        "DY",
        "EB",
        "EE",
        "EG",
        "EH",
        "EJ",
        "EL",
        "EP",
        "EU",
        "FC",
        "FD",
        "FE",
        "FI",
        "FL",
        "FR",
        "FS",
        "GB",
        "GE",
        "GF",
        "GH",
        "GI",
        "GL",
    ]

    dtype_dict = {feature: "float64" for feature in feature_names if feature != "EJ"}
    dtype_dict["EJ"] = "category"

    # Restore feature names
    data.columns = feature_names

    # Datatypes
    data = data.astype(dtype=dtype_dict, copy=False)

    return data


# ---------------------------- Prediction function --------------------------- #


def predict_fn(data: pd.DataFrame, model: Pipeline) -> np.ndarray:
    """
    Predicts the probabilities for the given data using the given model. The prediction
    function takes the output of 'input_fn', which is a pandas dataframe.

    Parameters
    ----------
    data : pd.DataFrame
        The feature matrix as a pandas DataFrame.
    model : Pipeline
        The model loaded from model_dir to use for prediction.

    Returns
    -------
    np.ndarray
        The predicted uncalibrated probabilities vector.
    """
    # Obtain the uncalibrated probabilities
    return model.predict_proba(data)
