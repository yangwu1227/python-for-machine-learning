import calendar
import datetime
import json
import logging
import sys
import time
import requests
from typing import Dict

import boto3
from botocore.exceptions import ClientError

from src.custom_utils import get_logger

class RestApiManager(object):
    """
    This class wraps the AWS APIGateway and Lambda clients and provides methods that simplify
    working with Rest API (i.e., creating, invoking, and deleting, etc.) using the AWS SDK for 
    Python (Boto3).
    """
    def __init__(self, api_name: str, api_base_path: str, api_stage: str, lambda_function_name: str):
        """
        Constructor for the RestApiManager class.

        Parameters
        ----------
        api_name : str
            The name of the REST API.
        api_base_path : str
            The base path part of the REST API URL.
        api_stage : str
            The deployment stage of the REST API.
        lambda_function_name : str
            The name of the AWS Lambda function called by Amazon API Gateway to handle REST requests.

        Returns
        -------
        None 
        """
        self.apigateway_client = boto3.client('apigateway')
        self.lambda_client = boto3.client('lambda')

        self.api_name = api_name
        self.api_base_path = api_base_path
        self.api_stage = api_stage
        self.api_id = None
        self._api_key_value = None
        self.api_key_id = None

        self.root_id = None
        self.resource_id = None
        self.usage_plan_id = None

        self.account_id = boto3.client('sts').get_caller_identity().get('Account')
        self.lambda_function_name = lambda_function_name
        self.lambda_function_arn = f'arn:aws:lambda:{self.lambda_client.meta.region_name}:{self.account_id}:function:{lambda_function_name}'

        self.logger = get_logger(__name__)

    def _create_rest_api(self) -> None:
        """ 
        Create a REST API in Amazon API Gateway. 

        Returns
        -------
        None
        """
        try:
            response = self.apigateway_client.create_rest_api(name=self.api_name)
            self.api_id = response['id']
            self.logger.info(f'Created REST API {self.api_name} with ID {self.api_id}')
        except ClientError as error:
            self.logger.exception(f'Cannot create REST API {self.api_name} due to {error.response["Error"]["Message"]}')
            raise

    def _get_root_resource_id(self) -> None:
        """
        Get the ID of the root resource of the REST API.

        Returns
        -------
        None
        """
        try:
            response = self.apigateway_client.get_resources(restApiId=self.api_id)
            # The items list contains a resource for each path part of the REST API, including '/' for the root
            self.root_id = next(item['id'] for item in response['items'] if item['path'] == '/')
            self.logger.info(f'Found root resource of the REST API with ID {self.root_id}')
        except ClientError as error:
            self.logger.exception('Cannot get the ID of the root resource of the REST API')
            raise

    def _create_resource(self) -> None:
        """
        Create a new resource under the root resource.

        Returns
        -------
        None
        """
        try:
            response = self.apigateway_client.create_resource(
                restApiId=self.api_id, 
                parentId=self.root_id, 
                pathPart=self.api_base_path
            )
            self.resource_id = response['id']
            self.logger.info(f'Created resource {self.api_base_path} under root resource with ID {self.resource_id}')
        except ClientError as error:
            self.logger.exception(f'Cannot create resource {self.api_base_path} due to {error.response["Error"]["Message"]}')
            raise

    def _create_post_method(self, apiKeyRequired: bool) -> None:
        """
        Create a POST method for the new resource.

        Parameters
        ----------
        apiKeyRequired : bool
            Specifies whether the method requires an API key.

        Returns
        -------
        None
        """
        try:
            self.apigateway_client.put_method(
                restApiId=self.api_id,
                resourceId=self.resource_id,
                httpMethod='POST',
                authorizationType='NONE',
                apiKeyRequired=apiKeyRequired
            )
            self.logger.info(f'Created POST method for resource {self.resource_id}')
        except ClientError as error:
            self.logger.exception(f'Cannot create POST method for resource {self.resource_id} due to {error.response["Error"]["Message"]}')
            raise

    def _setup_lambda_integration(self) -> None:
        """
        Set up integration for the POST method (https://docs.aws.amazon.com/apigateway/latest/developerguide/set-up-lambda-proxy-integrations.html).

        Returns
        -------
        None
        """
        # Format arn:aws:apigateway:{region}:{subdomain.service|service}:path|action/{service_api}
        lambda_uri = f'arn:aws:apigateway:{self.apigateway_client.meta.region_name}:lambda:path/2015-03-31/functions/{self.lambda_function_arn}/invocations'
        try:
            self.apigateway_client.put_integration(
                restApiId=self.api_id,
                resourceId=self.resource_id,
                httpMethod='POST',
                type='AWS_PROXY', # Set up Lambda proxy integration 
                integrationHttpMethod='POST',
                uri=lambda_uri
            )
            self.logger.info(f'Set up Lambda integration for POST method on resource {self.resource_id}')
        except ClientError as error:
            self.logger.exception(f'Cannot set up Lambda integration for POST method on resource {self.resource_id} due to {error.response["Error"]["Message"]}')
            raise

    def _deploy_rest_api(self) -> None:
        """
        Deploy the API.

        Returns
        -------
        None
        """
        try:
            self.apigateway_client.create_deployment(restApiId=self.api_id, stageName=self.api_stage)
            self.logger.info(f'Deployed REST API {self.api_id}')
        except ClientError as error:
            self.logger.exception(f'Cannot deploy REST API {self.api_id} due to {error.response["Error"]["Message"]}')
            raise

    def _grant_permission(self) -> None:
        """
        Update Lambda permission to allow the POST method with a Sourse ARN to invoke the Lambda function.

        Returns
        -------
        None
        """
        source_arn = (
            f'arn:aws:execute-api:{self.apigateway_client.meta.region_name}:{self.account_id}:{self.api_id}/*/POST/{self.api_base_path}'
        )
        try:
            self.lambda_client.add_permission(
                FunctionName=self.lambda_function_arn,
                StatementId='InvokeByApiGateway',
                Action='lambda:InvokeFunction',
                Principal='apigateway.amazonaws.com',
                SourceArn=source_arn
            )
            self.logger.info(f'Granted permission to let Amazon API Gateway invoke function {self.lambda_function_arn} from {source_arn}')
        except ClientError as error:
            self.logger.exception(f'Cannot add permission to let Amazon API Gateway invoke {self.lambda_function_arn} due to {error.response["Error"]["Message"]}')
            raise

    def _create_api_key(self, api_key_name: str, enabled: bool) -> None:
        """ 
        Create an API key for the REST API. The api key will be generated
        automatically by Amazon API Gateway when the `value` parameter is
        not specified in the `create_api_key` method.

        Parameters
        ----------
        api_key_name : str
            The name of the API key.
        enabled : bool
            Specifies whether the API key is enabled.

        Returns
        -------
        None
        """
        try:
            response = self.apigateway_client.create_api_key(
                name=api_key_name,
                enabled=enabled
            )
            self.api_key_id = response['id']
            self._api_key_value = response['value']
            self.logger.info(f'Created API key with ID {self.api_key_id}')
        except ClientError as error:
            self.logger.exception(f'Cannot create API key due to {error.response["Error"]["Message"]}')
            raise

    def get_api_key(self) -> str:
        """
        Securely retrieve the API key value.

        Returns
        -------
        str
            The API key value.
        """
        if self._api_key_value:
            return self._api_key_value
        else:
            raise ValueError('No API key value associated with this instance')

    def _create_usage_plan(self, usage_plan_name: str) -> None:
        """
        Create a (free) usage plan for the REST API

        Parameters
        ----------
        usage_plan_name : str
            The name of the usage plan.

        Returns
        -------
        None
        """
        try:
            response = self.apigateway_client.create_usage_plan(
                name=usage_plan_name,
                # Specify the API stages to associate with this usage plan
                apiStages=[
                    {
                        'apiId': self.api_id,
                        'stage': self.api_stage
                    }
                ],
                # Map containing throttling limits and quota limits
                throttle={
                    # Burst rate limit, which is the number of requests the API can handle concurrently
                    'burstLimit': 10,
                    # Number of allowed requests per second
                    'rateLimit': 10.0
                },
                quota={
                    # Maximum number of requests that can be made in a given time period
                    'limit': 10,
                    # The time period in which the limit applies
                    'period': 'DAY'
                }
            )
            self.usage_plan_id = response['id']
            self.logger.info(f'Created usage plan with ID {self.usage_plan_id}')
        except ClientError as error:
            self.logger.exception(f'Cannot create usage plan due to {error.response["Error"]["Message"]}')
            raise

    def _add_api_key_to_usage_plan(self) -> None:
        """
        Add the API key to the usage plan.

        Returns
        -------
        None
        """
        try:
            self.apigateway_client.create_usage_plan_key(
                usagePlanId=self.usage_plan_id,
                keyId=self.api_key_id,
                keyType='API_KEY'
            )
            self.logger.info(f'Added API key {self.api_key_id} to usage plan {self.usage_plan_id}')
        except ClientError as error:
            self.logger.exception(f'Cannot add API key {self.api_key_id} to usage plan {self.usage_plan_id} due to {error.response["Error"]["Message"]}')
            raise

    def _construct_api_url(self) -> str:
        """
        Constructs the URL of the REST API.

        Returns
        -------
        str
            The full URL of the REST API.
        """
        api_url = f'https://{self.api_id}.execute-api.{self.apigateway_client.meta.region_name}.amazonaws.com/{self.api_stage}/{self.api_base_path}'
        self.logger.info(f'Constructed REST API base URL: {api_url}')
        return api_url

    def delete_rest_api(self) -> None:
        """
        Deletes a REST API and all of its resources from Amazon API Gateway.

        Returns
        -------
        None
        """
        try:
            self.apigateway_client.delete_rest_api(restApiId=self.api_id)
            self.logger.info(f'Deleted REST API {self.api_id}')
        except ClientError as error:
            self.logger.exception(f'Cannot delete REST API {self.api_id} due to {error.response["Error"]["Message"]}')
            raise

    def _rollback_created_resources(self) -> None:
        """
        Rollback any resources created during the setup process. The order of deletion is important.

        1. Remove permission from the Lambda function, which allows the POST method to invoke the Lambda function.
        2. Delete (disasspciate) the API key from the usage plan, since usage plans cannot be deleted if an API key is associated with it.
        3. Delete the resource first, then the REST API, in the reverse order of creation.
        4. Delete the usage plan and API key.

        Returns
        -------
        None
        """
        self.logger.info('Rolling back created resources')

        try:
            self.lambda_client.remove_permission(
                FunctionName=self.lambda_function_arn,
                StatementId='InvokeByApiGateway'
            )
            self.logger.info(f'Removed permission "InvokeByApiGateway" from function {self.lambda_function_arn}')
        except ClientError as error:
            if error.response['Error']['Code'] == 'ResourceNotFoundException':
                # If the permission or the lambda function does not exist, then there is nothing to do
                self.logger.info('Nothing to remove as the specified Lambda function does not exist')
            else:
                self.logger.error(f'Failed to remove permission from Lambda function due to {error.response["Error"]["Message"]}')

        if self.usage_plan_id and self.api_key_id:
            try:
                self.apigateway_client.delete_usage_plan_key(usagePlanId=self.usage_plan_id, keyId=self.api_key_id)
                self.logger.info(f'Deleted API key {self.api_key_id} from usage plan {self.usage_plan_id}')
            except ClientError as error:
                self.logger.error(f'Failed to delete API key from usage plan: {error}')

        if self.resource_id:
            try:
                self.apigateway_client.delete_resource(restApiId=self.api_id, resourceId=self.resource_id)
                self.logger.info(f'Deleted resource {self.resource_id}')
            except ClientError as error:
                self.logger.error(f'Failed to delete resource: {error}')
            self.resource_id = None

        if self.api_id:
            try:
                self.apigateway_client.delete_rest_api(restApiId=self.api_id)
                self.logger.info(f'Deleted REST API {self.api_id}')
            except ClientError as error:
                self.logger.error(f'Failed to delete REST API: {error}')
            self.api_id = None

        if self.usage_plan_id:
            try:
                self.apigateway_client.delete_usage_plan(usagePlanId=self.usage_plan_id)
                self.logger.info(f'Deleted usage plan {self.usage_plan_id}')
            except ClientError as error:
                self.logger.error(f'Failed to delete usage plan: {error}')
            self.usage_plan_id = None

        if self.api_key_id:
            try:
                self.apigateway_client.delete_api_key(apiKey=self.api_key_id)
                self.logger.info(f'Deleted API key {self.api_key_id}')
            except ClientError as error:
                self.logger.error(f'Failed to delete API key: {error}')
            self.api_key_id = None

    def setup_rest_api(self, apiKeyRequired: bool, api_key_name: str = None, enabled: bool = None, usage_plan_name: str = None) -> None:
        """
        Set up a REST API with a POST method that invokes an AWS Lambda function.

        Parameters
        ----------
        apiKeyRequired : bool
            Specifies whether the POST method requires an API key.
        api_key_name : str, optional
            The name of the API key, by default None. This parameter is required if `apiKeyRequired` is True.
        enabled : bool, optional
            Specifies whether the API key is enabled, by default None. This parameter is required if `apiKeyRequired` is True.
        usage_plan_name : str
            The name of the usage plan, by default None. This parameter is required if `apiKeyRequired` is True.

        Returns
        -------
        None
        """
        error_occurred = False  # Flag to track if an error occurred

        try:
            if self.api_id:
                raise ValueError('This instance is already associated with a REST API')

            self._create_rest_api()
            self._get_root_resource_id()
            self._create_resource()
            self._create_post_method(apiKeyRequired=apiKeyRequired)
            self._setup_lambda_integration()
            self._deploy_rest_api()
            self._grant_permission()

            if apiKeyRequired:
                if not api_key_name or not enabled or not usage_plan_name:
                    raise ValueError('API key name, enabled, and usage plan name must be specified')
                self._create_api_key(api_key_name=api_key_name, enabled=enabled)
                self._create_usage_plan(usage_plan_name=usage_plan_name)
                self._add_api_key_to_usage_plan()
        except Exception as error:
            error_occurred = True
            self.logger.exception(f'Cannot set up REST API due to {error}')

        if error_occurred:
            self._rollback_created_resources()
        else:
            self.logger.info('Finished setting up REST API')
            

    def invoke_rest_api(self, payload: Dict[str, str]) -> Dict[str, str]:
        """
        Invoke the REST API.

        Parameters
        ----------
        payload : Dict[str, str]
            The payload to send to the REST API.

        Returns
        -------
        Dict[str, str]
            The response from the REST API.
        """
        api_url = self._construct_api_url()
        if self._api_key_value:
            headers = {'x-api-key': self._api_key_value}
            try:
                response = requests.post(api_url, json=payload, headers=headers)
                self.logger.info(f'Invoked REST API {self.api_id} with payload {payload} and API key {self.api_key_id}')
                return response.json()
            except requests.exceptions.RequestException as error:
                self.logger.exception(f'Cannot invoke REST API {self.api_id} due to {error}')
                raise
        else:
            # No API key required
            try:
                response = requests.post(api_url, json=payload)
                self.logger.info(f'Invoked REST API {self.api_id} with payload {payload}')
                return response.json()
            except requests.exceptions.RequestException as error:
                self.logger.exception(f'Cannot invoke REST API {self.api_id} due to {error}')
                raise

    def cleanup(self):
        """
        Clean up any resources created during the setup process.

        Returns
        -------
        None
        """
        self.logger.info('Cleaning up resources created during the setup process')
        self._rollback_created_resources()