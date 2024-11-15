variable "project_prefix" {
  type        = string
  description = "Prefix to use when naming all resources for the project"
}

variable "region" {
  description = "AWS region"
  type        = string
}

variable "account_id" {
  description = "AWS account ID"
  type        = string
}

variable "s3_bucket_prefix" {
  type        = string
  description = "Name of the S3 bucket that will trigger the Lambda function"
}

variable "s3_filter_prefix" {
  type        = string
  description = "Prefix filter for objects in the S3 bucket that will trigger the Lambda function"
}

variable "lambda_handler" {
  type        = string
  description = "Handler for the Lambda function (e.g., 'index.handler' for Node.js or 'main.lambda_handler' for Python)"
}

variable "lambda_runtime" {
  type        = string
  description = "Runtime environment for the Lambda function (e.g., 'python3.9', 'nodejs14.x')"
}

variable "lambda_filename" {
  type        = string
  description = "Path to the deployment package for the Lambda function"
}

variable "lambda_environment_variables" {
  type        = map(string)
  description = "Key-value map of environment variables for the Lambda function"
  default     = {}
}

variable "lambda_timeout" {
  type        = number
  description = "Timeout for the Lambda function in seconds"
}

variable "env" {
  type        = string
  description = "Environment (e.g., 'dev', 'prod')"
}
