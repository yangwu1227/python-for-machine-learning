import os
from urllib.parse import ParseResult, urlparse

from botocore.exceptions import ClientError
from sagemaker.session import Session


def upload_video(sm_session: Session, video_path: str, bucket: str, key: str) -> str:
    """
    Uploads a video to S3 bucket at the specified key and returns the S3 URI.

    Parameters
    ----------
    sm_session : sagemaker.session.Session
        The SageMaker session.
    video_path : str
        The path to the video file to upload.
    s3_bucket : str
        The name of the S3 bucket.
    s3_key : str
        The key of the video file in the S3 bucket.

    Returns
    -------
    str
        The S3 URI of the uploaded video.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"File not found: {video_path}")

    s3_uri: str = sm_session.upload_data(
        path=video_path,
        bucket=bucket,
        key_prefix=f"{key}/input",
        extra_args={"ContentType": "video/mp4"},
    )
    return s3_uri


def read_prediction(sm_session: Session, output_s3_uri: str) -> str:
    """
    Read the prediction from the output S3 URI.

    Parameters
    ----------
    sm_session : sagemaker.session.Session
        The SageMaker session.
    s3_uri : str
        The S3 URI of the output prediction.

    Returns
    -------
    str
        The body of the s3 file as a string.
    """
    parsed_s3_uri: ParseResult = urlparse(output_s3_uri)
    s3_bucket: str = parsed_s3_uri.netloc
    s3_key: str = parsed_s3_uri.path.lstrip("/")
    try:
        response_str: str = sm_session.read_s3_file(bucket=s3_bucket, key_prefix=s3_key)
        return response_str
    except ClientError as error:
        raise error
    except Exception as error:
        raise error


def download_prediction(
    sm_session: Session, output_s3_uri: str, local_path: str
) -> None:
    """
    Download the prediction from the output S3 URI to the local path.

    Parameters
    ----------
    sm_session : sagemaker.session.Session
        The SageMaker session.
    output_s3_uri : str
        The S3 URI of the output prediction.
    local_path : str
        The local path to save the prediction.
    """
    parsed_s3_uri: ParseResult = urlparse(output_s3_uri)
    s3_bucket: str = parsed_s3_uri.netloc
    s3_key: str = parsed_s3_uri.path.lstrip("/")
    try:
        sm_session.download_data(
            bucket=s3_bucket,
            key_prefix=s3_key,
            path=local_path,
        )
    except ClientError as error:
        raise error
    except Exception as error:
        raise error
