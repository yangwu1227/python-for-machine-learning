terraform {
  backend "s3" {
    bucket  = "yang-templates"
    key     = "terraform-states/tenant-experience/dev/terraform.tfstate" # Development 
    region  = "us-east-1"
    profile = "admin" # The credentials profile in ~/.aws/config with permissions to interact with the S3 bucket
  }

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region  = "us-east-1"
  profile = "admin"
}

module "batch_transform" {
  source           = "../../modules"
  project_prefix   = "tenant_experience"
  region           = "us-east-1"
  account_id       = "722696965592"
  env              = "dev" # Development
  s3_bucket_prefix = "tenant-experience"
  s3_filter_prefix = "batch-transform-input-data/"
  lambda_handler   = "lambda_function.lambda_handler"
  lambda_runtime   = "python3.12"
  lambda_filename  = "../code/lambda_function.zip"
  lambda_timeout   = 180
  lambda_environment_variables = {
    "MODEL_NAME"                     = "tenant-experience-xgboost-v44bcb1"
    "S3_BUCKET"                      = "tenant-experience-dev" # Development
    "S3_KEY_INPUT"                   = "batch-transform-input-data/"
    "S3_KEY_OUTPUT"                  = "batch-transform-output/"
    "INSTANCE_TYPE"                  = "ml.c5.xlarge"
    "INSTANCE_COUNT"                 = "1"
    "INVOCATIONS_TIMEOUT_IN_SECONDS" = "900"
    "CONTENT_TYPE"                   = "application/x-parquet"
    "ACCEPTED"                       = "text/csv"
    "INVOCATIONS_MAX_RETRIES"        = "3"
    "ENV"                            = "dev" # Development
  }
}
