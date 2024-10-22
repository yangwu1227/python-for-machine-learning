import argparse
import io
import logging
import os

import boto3
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from custom_pipeline import create_pipeline
from custom_utils import S3Pickle, load_data, stratified_split, weighted_ap_score
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight

# --------------------- Parse argument from command line --------------------- #


def parser() -> argparse.Namespace:
    """
    Function that parses arguments from command line.

    Returns
    -------
    argparse.Namespace
        Namespace with arguments.
    """
    parser = argparse.ArgumentParser()
    # Default sagemaker arguments are set in the environment variables
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--train", type=str, default=os.environ["SM_CHANNEL_TRAIN"])
    parser.add_argument("--s3_key", "-s3_key", type=str, default="customer_churn")
    parser.add_argument(
        "--s3_bucket", "-s3_bucket", type=str, default="yang-ml-sagemaker"
    )
    args, _ = parser.parse_known_args()

    return args


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
    model_pipeline = joblib.load(os.path.join(model_dir, "model.joblib"))
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
    assert content_type == "application/x-npy"
    npy_array = np.load(io.BytesIO(input_data), allow_pickle=True)

    # Convert numpy array to pandas DataFrame
    data = pd.DataFrame(npy_array)

    # Feature names
    feature_names = [
        "gender",
        "age",
        "under_30",
        "senior_citizen",
        "married",
        "dependents",
        "number_of_dependents",
        "number_of_referrals",
        "tenure_in_months",
        "offer",
        "phone_service",
        "avg_monthly_long_distance_charges",
        "multiple_lines",
        "internet_service",
        "internet_type",
        "avg_monthly_gb_download",
        "online_security",
        "online_backup",
        "device_protection_plan",
        "premium_tech_support",
        "streaming_tv",
        "streaming_movies",
        "streaming_music",
        "unlimited_data",
        "contract",
        "paperless_billing",
        "payment_method",
        "monthly_charge",
        "total_charges",
        "total_refunds",
        "total_extra_data_charges",
        "total_long_distance_charges",
        "satisfaction_score",
        "cltv",
    ]

    # Feature types
    dtype_dict = {
        "gender": "category",
        "age": "int64",
        "under_30": "category",
        "senior_citizen": "category",
        "married": "category",
        "dependents": "category",
        "number_of_dependents": "int64",
        "number_of_referrals": "int64",
        "tenure_in_months": "int64",
        "offer": "category",
        "phone_service": "category",
        "avg_monthly_long_distance_charges": "float64",
        "multiple_lines": "category",
        "internet_service": "category",
        "internet_type": "category",
        "avg_monthly_gb_download": "int64",
        "online_security": "category",
        "online_backup": "category",
        "device_protection_plan": "category",
        "premium_tech_support": "category",
        "streaming_tv": "category",
        "streaming_movies": "category",
        "streaming_music": "category",
        "unlimited_data": "category",
        "contract": "category",
        "paperless_billing": "category",
        "payment_method": "category",
        "monthly_charge": "float64",
        "total_charges": "float64",
        "total_refunds": "float64",
        "total_extra_data_charges": "int64",
        "total_long_distance_charges": "float64",
        "satisfaction_score": "int64",
        "cltv": "int64",
    }

    # Check that the input dataframe has the correct number of columns
    if len(data.columns) != len(feature_names):
        raise AttributeError(
            "The input data does not have the right number of features"
        )

    # Restore feature names
    data.columns = feature_names

    # Datatypes
    data = data.astype(dtype=dtype_dict, copy=False)

    return data


# ---------------------------- Prediction function --------------------------- #


def predict_fn(data: pd.DataFrame, model: Pipeline) -> np.ndarray:
    """
    Predicts the target variable for the given data using the given model. The prediction
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
    # Obtain the uncalibrated probabilities instead of labels
    return model.predict_proba(data)


if __name__ == "__main__":
    args = parser()

    X_train, y_train = load_data(
        data_s3_url=os.path.join(args.train, "train.csv"), logger=None
    )

    # Sample weights
    sample_weights = compute_sample_weight("balanced", y_train)

    # Hyperparameters for processing
    hyperparams = {
        "colsample_bylevel": 0.921645063797166,
        "colsample_bynode": 0.9610978659904691,
        "colsample_bytree": 0.33889598945736665,
        "gamma": 3,
        "lambda": 25,
        "learning_rate": 0.22349223791882736,
        "max_delta_step": 1,
        "max_depth": 11,
        "num_boost_round": 1500,
        "num_feat": 60,
        "step": 0.1,
        "subsample": 0.6161079290690729,
    }

    # Preprocessing logic first
    model_pipeline = create_pipeline(
        num_feat=hyperparams["num_feat"], step=hyperparams["step"]
    )
    # Append xgbclassifier as a final step
    model_pipeline.steps.append(
        [
            "xgb_clf",
            xgb.XGBClassifier(
                n_estimators=hyperparams["num_boost_round"],
                max_depth=hyperparams["max_depth"],
                learning_rate=hyperparams["learning_rate"],
                objective="binary:logistic",
                booster="gbtree",
                tree_method="hist",
                n_jobs=-1,
                gamma=hyperparams["gamma"],
                max_delta_step=hyperparams["max_delta_step"],
                subsample=hyperparams["subsample"],
                colsample_bytree=hyperparams["colsample_bytree"],
                colsample_bylevel=hyperparams["colsample_bylevel"],
                colsample_bynode=hyperparams["colsample_bynode"],
                reg_alpha=None,
                reg_lambda=hyperparams["lambda"],
                scale_pos_weight=(y_train == 0).sum() / y_train.sum(),
            ),
        ]
    )

    model_pipeline.fit(X=X_train, y=y_train, xgb_clf__sample_weight=sample_weights)

    joblib.dump(model_pipeline, os.path.join(args.model_dir, "model.joblib"))
