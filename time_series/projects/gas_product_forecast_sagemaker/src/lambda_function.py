import json
import os
from logging import StreamHandler, getLogger
from sys import stdout
from typing import Any, Dict, Union

import boto3
from botocore.exceptions import ClientError, ParamValidationError
from mypy_boto3_sagemaker_runtime import SageMakerRuntimeClient

runtime: SageMakerRuntimeClient = boto3.client("runtime.sagemaker")  # type: ignore
logger = getLogger("invoke_endpoint")
logger.setLevel("INFO")
logger.addHandler(StreamHandler(stdout))


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Union[int, str]]:
    # Event object dictionary contains request parameters, api config data, and request body
    payload = event["body"]
    try:
        endpoint_name: str = os.getenv("SAGEMAKER_SERVERLESS_ENDPOINT", "")
        if not endpoint_name:
            raise ValueError(
                "SAGEMAKER_SERVERLESS_ENDPOINT environment variable is not set"
            )
        response = runtime.invoke_endpoint(
            EndpointName=endpoint_name,
            Body=payload,  # No need to use json.dumps since the request body is already a JSON string
            ContentType="application/json",
            Accept="application/json",
        )
    except ClientError as error:
        logger.error(f"Client error invoking SageMaker endpoint: {error}")
        raise error
    except ParamValidationError as error:
        logger.error(f"Parameter validation error: {error}")
        raise error
    except Exception as error:
        logger.error(f"Internal error invoking SageMaker endpoint: {error}")
        raise error

    predictions = json.loads(response["Body"].read().decode(encoding="utf-8"))
    return {"statusCode": 200, "body": json.dumps(predictions)}
