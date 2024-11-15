terraform {
  backend "s3" {
    bucket  = "yang-templates"
    key     = "terraform-states/tenant-experience/github-action/terraform.tfstate" # Path to the state file in the S3 bucket
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
  region  = var.region
  profile = var.profile
}

locals {
  environments = ["dev", "prod"]
}

data "terraform_remote_state" "environments" {
  for_each = toset(local.environments)
  backend  = "s3"
  config = {
    bucket  = "yang-templates"
    key     = "terraform-states/tenant-experience/${each.key}/terraform.tfstate"
    region  = "us-east-1"
    profile = "admin"
  }
}
