terraform {
  backend "s3" {
  }

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region  = var.region
  profile = var.profile
}

module "batch_transform" {
  source                       = "../../modules"
  project_prefix               = var.project_prefix
  region                       = var.region
  account_id                   = var.account_id
  env                          = var.env
  s3_bucket_prefix             = var.s3_bucket_prefix
  s3_filter_prefix             = var.s3_filter_prefix
  lambda_handler               = var.lambda_handler
  lambda_runtime               = var.lambda_runtime
  lambda_filename              = var.lambda_filename
  lambda_timeout               = var.lambda_timeout
  lambda_environment_variables = var.lambda_environment_variables
}

