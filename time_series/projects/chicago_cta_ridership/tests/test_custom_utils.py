import io

import boto3
import joblib
import pandas as pd
import pytest
from moto import mock_s3
from sktime.forecasting.compose import ColumnEnsembleForecaster
from src.custom_utils import S3Helper


@pytest.fixture(scope="class")
def sample_data():
    return pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})


@pytest.fixture(scope="class")
def sample_model():
    # Creating a dummy model for testing purposes
    from sktime.forecasting.naive import NaiveForecaster

    forecaster = NaiveForecaster(strategy="last")
    model = ColumnEnsembleForecaster([("naive", forecaster, 0)])
    return model


@pytest.fixture(scope="class")
def s3_helper():
    return S3Helper()


class TestS3Helper:
    """
    Unit tests for the S3Helper class.
    """

    def test_read_parquet(self, sample_data, s3_helper):
        """
        Test the read_parquet method.
        """
        with mock_s3():
            conn = boto3.resource("s3", region_name="us-east-1")
            conn.create_bucket(Bucket="yang-ml-sagemaker")

            # Save a in-memory data to the mock S3 bucket as a parquet file
            buffer = io.BytesIO()
            sample_data.to_parquet(buffer)
            conn.Object(
                "yang-ml-sagemaker", "chicago_cta_ridership/sample.parquet"
            ).put(Body=buffer.getvalue())

            # Test the read_parquet method by reading the parquet file back from the mock S3 bucket
            test_data = s3_helper.read_parquet("sample.parquet")
            pd.testing.assert_frame_equal(test_data, sample_data)

    def test_to_parquet(self, sample_data, s3_helper):
        """
        Test the to_parquet method.
        """
        with mock_s3():
            conn = boto3.resource("s3", region_name="us-east-1")
            conn.create_bucket(Bucket="yang-ml-sagemaker")

            # Test the to_parquet method by writing in-memory data to the mock S3 bucket
            s3_helper.to_parquet(sample_data, "test_output.parquet")

            # Read the saved data back from the mock S3 bucket
            obj = (
                conn.Object(
                    "yang-ml-sagemaker", "chicago_cta_ridership/test_output.parquet"
                )
                .get()
                .get("Body")
                .read()
            )
            buffer = io.BytesIO(obj)
            test_data = pd.read_parquet(buffer)

            pd.testing.assert_frame_equal(test_data, sample_data)

    def test_upload_joblib(self, sample_model, s3_helper):
        """
        Test the upload_joblib method.
        """
        with mock_s3():
            conn = boto3.resource("s3", region_name="us-east-1")
            conn.create_bucket(Bucket="yang-ml-sagemaker")

            # Test the upload_joblib method by saving a model to the mock S3 bucket
            s3_helper.upload_joblib(sample_model, "sample_model.joblib")

            # Check if the model file exists in the S3 bucket
            obj = conn.Object(
                "yang-ml-sagemaker", "chicago_cta_ridership/sample_model.joblib"
            )
            assert obj.get()

    def test_download_joblib(self, sample_model, s3_helper):
        """
        Test the download_joblib method.
        """
        with mock_s3():
            conn = boto3.resource("s3", region_name="us-east-1")
            conn.create_bucket(Bucket="yang-ml-sagemaker")

            # Save a model to the mock S3 bucket for testing the download
            buffer = io.BytesIO()
            joblib.dump(sample_model, buffer)
            conn.Object(
                "yang-ml-sagemaker", "chicago_cta_ridership/sample_model.joblib"
            ).put(Body=buffer.getvalue())

            # Test the download_joblib method
            downloaded_model = s3_helper.download_joblib("sample_model.joblib")

            assert type(downloaded_model) == type(sample_model)

            downloaded_model_params = downloaded_model.get_params()
            sample_model_params = sample_model.get_params()

            for param_key in downloaded_model_params:
                assert (
                    downloaded_model_params[param_key] == sample_model_params[param_key]
                )
