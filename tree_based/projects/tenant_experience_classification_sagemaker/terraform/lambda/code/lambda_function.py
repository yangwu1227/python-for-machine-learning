import os
import sys
import uuid
from datetime import datetime
from logging import StreamHandler, getLogger
from typing import Any, Dict, Optional, Tuple

import boto3
import botocore
from mypy_boto3_s3 import S3Client
from mypy_boto3_s3.paginator import ListObjectsV2Paginator
from mypy_boto3_sagemaker import SageMakerClient

logger = getLogger("batch-transform-lambda")
logger.setLevel("INFO")
if not logger.hasHandlers():
    handler = StreamHandler(sys.stdout)
    logger.addHandler(handler)


def calculate_obj_size_and_count(
    s3_client: S3Client, s3_bucket: str, s3_key: str
) -> Tuple[int, int]:
    """
    Calculate the total size and count of objects in an S3 bucket
    with a given key prefix.

    Parameters
    ----------
    s3_client : S3Client
        The boto3 S3 client.
    s3_bucket : str
        The name of the S3 bucket.
    s3_key : str
        The key prefix to search for objects.

    Returns
    -------
    Tuple[int, int]
        A tuple containing the total size (bytes) and count of objects.

    Raises
    ------
    ValueError
        If no objects are found with the given prefix.
    """
    try:
        total_size: int = 0
        total_count: int = 0
        paginator: ListObjectsV2Paginator = s3_client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=s3_bucket, Prefix=s3_key):
            objects = page.get("Contents", [])
            total_count += len(objects)
            total_size += sum(obj["Size"] for obj in objects)
        if total_count == 1:
            raise ValueError(f"No objects found with prefix: {s3_key}")
        # Subtract 1 from the count to account for the prefix itself
        logger.info(f"Total size: {total_size} bytes, Total count: {total_count - 1}")
        return total_size, total_count - 1

    except ValueError as error:
        logger.error(f"Failed to calculate object size and count: {error}")
        raise error

    except botocore.exceptions.ClientError as error:
        logger.error(f"Failed to list objects in S3 bucket: {error}")
        raise error

    except Exception as error:
        logger.error(f"Unexpected error occurred: {error}")
        raise error


def optimize_payload_and_concurrency(
    total_size: int, total_count: int, max_payload_limit: int = 90 * 1024 * 1024
) -> Tuple[int, int]:
    """
    Optimize the MaxPayloadInMB and MaxConcurrentTransforms parameters.

    Parameters
    ----------
    total_size : int
        The total size of objects in bytes.
    total_count : int
        The total number of objects.
    limit : int
        The maximum allowed payload size in bytes (default is 90 MB),
        allowing a small buffer relative to the 100 MB limit to avoid
        any edges cases.

    Returns
    -------
    Tuple[int, int]
        The optimized MaxPayload (in MB) and MaxConcurrentTransforms.
    """
    # Calculate average object size in bytes
    avg_obj_size = total_size // total_count
    # Determine MaxPayload, constrained to max_payload_limit and converted to MB
    max_payload = min(max_payload_limit, avg_obj_size)
    # Even if max_payload // (1024 * 1024) is smaller than 1 mb, set it to 1 as a minimum
    max_payload_in_mb = max(1, max_payload // (1024 * 1024))
    # Ensure that (max_payload_mb * max_concurrent_transforms) is less than 100 mb
    max_concurrent_transforms = 100 // max_payload_in_mb
    logger.info(
        f"Optimized MaxPayloadInMB: {max_payload_in_mb}, MaxConcurrentTransforms: {max_concurrent_transforms}"
    )
    return max_payload_in_mb, max_concurrent_transforms


def lambda_handler(
    event: Optional[Dict[str, Any]], context: Optional[Any]
) -> Dict[str, Any]:
    sm_client: SageMakerClient = boto3.client("sagemaker")
    s3_client: S3Client = boto3.client("s3")

    env = os.environ.get("ENV", "dev")
    model_name: str = os.getenv("MODEL_NAME", "")
    s3_bucket: str = os.getenv("S3_BUCKET", "")
    s3_key_input: str = os.getenv("S3_KEY_INPUT", "")
    s3_key_output: str = os.path.join(
        os.getenv("S3_KEY_OUTPUT", ""), datetime.today().strftime("%Y-%m-%d")
    )
    instance_type = os.environ.get("INSTANCE_TYPE", "")
    instance_count = int(os.environ.get("INSTANCE_COUNT", 1))
    invocations_timeout_in_seconds = int(
        os.environ.get("INVOCATIONS_TIMEOUT_IN_SECONDS", 900)
    )
    content_type = os.environ.get("CONTENT_TYPE", "text/csv")
    accepted = os.environ.get("ACCEPT", "text/csv")
    invocations_max_retries = int(os.environ.get("INVOCATIONS_MAX_RETRIES", 3))
    transform_job_name = (
        f"tenant-experience-xgboost-batch-transform-{str(uuid.uuid4())[:6]}"
    )

    logger.info(f"Model Name: {model_name}")
    logger.info(f"S3 Bucket: {s3_bucket}")
    logger.info(f"S3 Key Input: {s3_key_input}")
    logger.info(f"S3 Key Output: {s3_key_output}")
    logger.info(f"Content Type: {content_type}")
    logger.info(f"Instance Type: {instance_type}")
    logger.info(f"Instance Count: {instance_count}")
    logger.info(f"Invocations Timeout (s): {invocations_timeout_in_seconds}")
    logger.info(f"Invocations Max Retries: {invocations_max_retries}")
    logger.info(f"Transform Job Name: {transform_job_name}")

    try:
        total_size, total_count = calculate_obj_size_and_count(
            s3_client, s3_bucket, s3_key_input
        )
        max_payload_in_mb, max_concurrent_transforms = optimize_payload_and_concurrency(
            total_size, total_count
        )
        response = sm_client.create_transform_job(
            TransformJobName=transform_job_name,
            ModelName=model_name,
            ModelClientConfig={
                "InvocationsTimeoutInSeconds": invocations_timeout_in_seconds,
                "InvocationsMaxRetries": invocations_max_retries,
            },
            MaxPayloadInMB=max_payload_in_mb,
            MaxConcurrentTransforms=max_concurrent_transforms,
            BatchStrategy="MultiRecord",
            TransformInput={
                "DataSource": {
                    "S3DataSource": {
                        "S3DataType": "S3Prefix",
                        "S3Uri": f"s3://{s3_bucket}/{s3_key_input}",
                    }
                },
                "ContentType": content_type,
            },
            TransformOutput={
                "S3OutputPath": f"s3://{s3_bucket}/{s3_key_output}",
                "Accept": accepted,
                "AssembleWith": "Line",
            },
            TransformResources={
                "InstanceType": instance_type,
                "InstanceCount": instance_count,
            },
        )
        return {
            "statusCode": 200,
            "body": f"SageMaker Batch Transform Job {transform_job_name} started successfully",
            "response": response,
        }

    except botocore.exceptions.ClientError as error:
        logger.error(f"Failed to start SageMaker batch transform job: {error}")
        return {
            "statusCode": 500,
            "body": f"Failed to start SageMaker batch transform job: {error}",
        }

    except botocore.exceptions.ParamValidationError as error:
        logger.error(f"Invalid parameters: {error}")
        return {"statusCode": 400, "body": f"Invalid parameters: {error}"}

    except Exception as error:
        logger.error(f"Internal error occurred: {error}")
        return {"statusCode": 500, "body": f"Internal error occurred: {error}"}


if __name__ == "__main__":
    lambda_handler(None, None)
