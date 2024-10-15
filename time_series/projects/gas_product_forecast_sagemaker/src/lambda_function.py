import os
import boto3
import json

runtime = boto3.client("runtime.sagemaker")


def lambda_handler(event, context):
    # Event object dictionary contains request parameters, api config data, and request body
    payload = event["body"]

    response = runtime.invoke_endpoint(
        EndpointName=os.environ["SAGEMAKER_SERVERLESS_ENDPOINT"],
        Body=payload,  # No need to use json.dumps since the request body is already a JSON string
        ContentType="application/json",
        Accept="application/json",
    )

    predictions = json.loads(response["Body"].read().decode(encoding="UTF-8"))
    return {"statusCode": 200, "body": json.dumps(predictions)}
