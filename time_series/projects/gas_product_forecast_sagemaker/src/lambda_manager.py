import io
import json
import zipfile
from typing import Dict, Optional, Tuple, Union

from botocore.exceptions import ClientError
from mypy_boto3_iam import IAMServiceResource
from mypy_boto3_iam.service_resource import Role
from mypy_boto3_lambda import LambdaClient
from mypy_boto3_lambda.literals import RuntimeType
from mypy_boto3_lambda.type_defs import (
    FunctionConfigurationResponseTypeDef,
    GetFunctionResponseTypeDef,
    InvocationResponseTypeDef,
)

from src.model_utils import setup_logger


class LambdaManager(object):
    """
    This class wraps the AWS Lambda client and provides methods that simplify
    working with Lambda functions (i.e., creating, invoking, and deleting, etc.)
    using the AWS SDK for Python (Boto3).

    Attributes
    ----------
    lambda_client : LambdaClient
        The AWS Lambda client instance.
    iam_resource : IAMServiceResource
        The AWS IAM resource instance.
    logger : logging.Logger
        The logger used by this class.
    """

    def __init__(self, lambda_client: LambdaClient, iam_resource: IAMServiceResource):
        """
        Constructor for LambdaManager class.

        Parameters
        ----------
        lambda_client : LambdaClient
            The AWS Lambda client instance.
        iam_resource : IAMServiceResource
            The AWS IAM resource instance.

        Returns
        -------
        None
        """
        self.lambda_client = lambda_client
        self.iam_resource = iam_resource
        self.logger = setup_logger(__name__)

    @staticmethod
    def create_deployment_package(source_file: str, destination_file: str) -> bytes:
        """
        Creates a Lambda deployment package in .zip format in an in-memory buffer.

        Parameters
        ----------
        source_file : str
            The name of the file that contains the Lambda handler function.
        destination_file : str
            The name to give the file when it's deployed to Lambda.

        Returns
        -------
        bytes
            The deployment package in bytes.
        """
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as zipped:
            zipped.write(source_file, destination_file)
        # Move to the start of the buffer, so we can read from the start of the lambda handler source code
        buffer.seek(0)
        return buffer.read()

    def get_iam_role(self, iam_role_name: str) -> Optional[Role]:
        """
        Get an AWS Identity and Access Management (IAM) role, which should be
        the execution role we created for SageMaker. This role gives SageMaker
        permission to create, invoke, and perform other actions related to
        AWS Lambda.

        Parameters
        ----------
        iam_role_name : str
            The name of the role to retrieve.

        Returns
        -------
        Optional[Role]
            The IAM role if found, otherwise None.
        """
        role: Optional[Role] = None
        try:
            temp_role = self.iam_resource.Role(iam_role_name)
            temp_role.load()
            role = temp_role
            self.logger.info(f"Found IAM role {iam_role_name}")
        except ClientError as error:
            if error.response["Error"]["Code"] == "NoSuchEntity":
                self.logger.info(f"IAM role {iam_role_name} does not exist")
            else:
                self.logger.exception(
                    f"Cannot find IAM role {iam_role_name} due to {error.response['Error']['Message']}"
                )
                raise error
        return role

    def create_iam_role_for_lambda(self, iam_role_name: str) -> Tuple[Role, bool]:
        """
        Creates an IAM role for Lambda function. If the role already exists, it is
        returned. In this project, we created the `forecast-lambda-execution-role`
        that gives Lambda basic execution permissions and permission to invoke SageMaker
        endpoints with the 'forecast-' prefix.

        Parameters
        ----------
        iam_role_name : str
            The name of the role to create.

        Returns
        -------
        Tuple[Role, bool]
            The role and a boolean indicating if the role was newly created.
        """
        # If the role already exists, return it, and indicate that it wasn't newly created.
        role = self.get_iam_role(iam_role_name)
        if role is not None:
            return role, False

        # Create the role and attach the basic execution policy to it
        lambda_assume_role_policy = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": {"Service": "lambda.amazonaws.com"},
                    "Action": "sts:AssumeRole",
                }
            ],
        }
        policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"

        try:
            role = self.iam_resource.create_role(
                RoleName=iam_role_name,
                AssumeRolePolicyDocument=json.dumps(lambda_assume_role_policy),
            )
            self.logger.info(f"Created IAM role {iam_role_name} for Lambda")
            role.attach_policy(PolicyArn=policy_arn)
            self.logger.info(f"Attached basic execution policy to role {role.name}")
        except ClientError as error:
            if error.response["Error"]["Code"] == "EntityAlreadyExists":
                role = self.iam_resource.Role(iam_role_name)
                role.load()
                self.logger.warning(
                    f"The role {iam_role_name} already exists, using it"
                )
            else:
                self.logger.exception(
                    f"Cannot create IAM role {iam_role_name} due to {error.response['Error']['Message']}"
                )
                raise error

        return role, True

    def get_function(self, function_name: str) -> Optional[GetFunctionResponseTypeDef]:
        """
        Gets meta-data about a Lambda function.

        Parameters
        ----------
        function_name : str
            The name of the function.

        Returns
        -------
        Optional[GetFunctionResponseTypeDef]
            The function data if found, otherwise None.
        """
        response: Optional[GetFunctionResponseTypeDef] = None
        try:
            response = self.lambda_client.get_function(FunctionName=function_name)
        except ClientError as error:
            if error.response["Error"]["Code"] == "ResourceNotFoundException":
                self.logger.info(f"Function {function_name} does not exist")
            else:
                self.logger.exception(
                    f"Cannot get function {function_name} due to {error.response['Error']['Message']}"
                )
                raise error
        return response

    def create_function(
        self,
        function_name: str,
        function_description: str,
        time_out: int,
        python_runtime: RuntimeType,
        iam_role: Role,
        handler_name: str,
        deployment_package: bytes,
        publish: bool,
        env_vars: Dict[str, str],
    ) -> str:
        """
        Deploys a Lambda function.

        Parameters
        ----------
        function_name : str
            The name of the Lambda function.
        function_description : str
            The description of the Lambda function.
        time_out : int
            The amount of time (in seconds) that Lambda allows a function to run before stopping it.
        python_runtime : RuntimeType
            The Python runtime to use for the function.
        iam_role : Role
            The IAM (execution) role to use for the function.
        handler_name : str
            The fully qualified name of the handler function.
        deployment_package : bytes
            The base64-encoded contents of the deployment package.
        publish : bool
            Set to true to publish the first version of the function during creation.
        env_vars : Dict[str, str]
            The environment variables to set for the function as key -> value pairs.

        Returns
        -------
        str
            The Amazon Resource Name (ARN) of the newly created function.
        """
        try:
            response = self.lambda_client.create_function(
                FunctionName=function_name,
                Role=iam_role.arn,
                Runtime=python_runtime,
                Description=function_description,
                Timeout=time_out,
                Handler=handler_name,
                Code={"ZipFile": deployment_package},
                Publish=publish,
                Environment={"Variables": env_vars},
            )

            function_arn = response["FunctionArn"]
            # This returns an object that can wait for some condition
            waiter = self.lambda_client.get_waiter("function_active_v2")
            # Wait for the function's state to be Active
            waiter.wait(FunctionName=function_name)
            self.logger.info(
                f"Function {function_name} is active with ARN {function_arn}"
            )
        except ClientError as error:
            self.logger.exception(
                f"Cannot create function due to {error.response['Error']['Message']}"
            )
            raise error
        else:
            return function_arn

    def delete_function(self, function_name: str) -> None:
        """
        Deletes a Lambda function.

        Parameters
        ----------
        function_name : str
            The name of the function to delete.
        """
        try:
            self.lambda_client.delete_function(FunctionName=function_name)
            self.logger.info(f"Deleted function {function_name}")
        except ClientError as error:
            self.logger.exception(
                f"Cannot delete function {function_name} due to {error.response['Error']['Message']}"
            )
            raise error

    def invoke_function(
        self,
        function_name: str,
        payload: Dict[str, Union[str, int]],
        include_log: bool = False,
    ) -> InvocationResponseTypeDef:
        """
        Invokes a Lambda function based on the function name and parameters. This
        method is useful for testing a Lambda function.

        Parameters
        ----------
        function_name : str
            The name of the function to invoke.
        payload : Dict[str, Union[str, int]]
            The payload to send to the function.
        include_log : bool, optional
            Include execution log in response, by default False.

        Returns
        -------
        InvocationResponseTypeDef
            The response from the function invocation.
        """
        try:
            response = self.lambda_client.invoke(
                FunctionName=function_name,
                # This is the default, which invokes the function synchronously and keeps the connection open until return or timeout
                InvocationType="RequestResponse",
                Payload=json.dumps(payload),
                LogType="Tail" if include_log else "None",
            )
            self.logger.info(f"Invoked function {function_name}")
        except ClientError as error:
            self.logger.exception(
                f"Cannot invoke function {function_name} due to {error.response['Error']['Message']}"
            )
            raise error
        return response

    def update_function_code(
        self, function_name: str, deployment_package: bytes
    ) -> FunctionConfigurationResponseTypeDef:
        """
        Updates the code for a Lambda function by submitting a .zip archive that contains
        the refactored or updated code for the function. This is useful for interactive
        development and testing of Lambda functions.

        Parameters
        ----------
        function_name : str
            The name of the function to update.
        deployment_package : bytes
            The function code in bytes.

        Returns
        -------
        FunctionConfigurationResponseTypeDef
            Data about the update, including the status.
        """
        try:
            response = self.lambda_client.update_function_code(
                FunctionName=function_name, ZipFile=deployment_package
            )
        except ClientError as error:
            self.logger.exception(
                f"Cannot update function {function_name} due to {error.response['Error']['Message']}"
            )
            raise error
        else:
            return response

    def update_function_configuration(
        self, function_name: str, env_vars: Dict[str, str]
    ) -> FunctionConfigurationResponseTypeDef:
        """
        Updates the environment variables for a Lambda function.

        Parameters
        ----------
        function_name : str
            The name of the function to update.
        env_vars : Dict[str, str]
            A dictionary of environment variables.

        Returns
        -------
        FunctionConfigurationResponseTypeDef
            Data about the update, including the status.
        """
        try:
            response = self.lambda_client.update_function_configuration(
                FunctionName=function_name, Environment={"Variables": env_vars}
            )
        except ClientError as error:
            self.logger.exception(
                f"Cannot update function {function_name} configurations due to {error.response['Error']['Message']}"
            )
            raise error
        else:
            return response

    def list_functions(self) -> None:
        """
        Lists the Lambda functions for the current account.
        """
        try:
            func_paginator = self.lambda_client.get_paginator("list_functions")
            for func_page in func_paginator.paginate():
                for func in func_page["Functions"]:
                    print(func["FunctionName"])
                    desc = func.get("Description")
                    if desc:
                        print(f"\t{desc}")
                    print(f"\t{func['Runtime']}: {func['Handler']}")
        except ClientError as error:
            self.logger.exception(
                f"Cannot list functions due to {error.response['Error']['Message']}"
            )
            raise error
