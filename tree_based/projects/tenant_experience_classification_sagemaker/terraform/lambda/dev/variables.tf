variable "region" {
  type        = string
  description = "AWS region where resources will be deployed"
}

variable "profile" {
  type        = string
  description = "AWS configuration profile with all required permissions"
}

variable "project_prefix" {
  type        = string
  description = "Prefix to use when naming all resources for the project"
}

variable "account_id" {
  description = "AWS Account ID"
  type        = string
}

variable "env" {
  description = "Environment (e.g., dev, prod)"
  type        = string
}

variable "s3_bucket_prefix" {
  description = "S3 bucket prefix"
  type        = string
}

variable "s3_filter_prefix" {
  description = "S3 filter prefix"
  type        = string
}

variable "lambda_handler" {
  description = "Lambda function handler"
  type        = string
}

variable "lambda_runtime" {
  description = "Lambda runtime"
  type        = string
}

variable "lambda_filename" {
  description = "Path to Lambda function ZIP file"
  type        = string
}

variable "lambda_timeout" {
  description = "Lambda function timeout"
  type        = number
}

variable "lambda_environment_variables" {
  description = "Lambda function environment variables"
  type        = map(string)
}
