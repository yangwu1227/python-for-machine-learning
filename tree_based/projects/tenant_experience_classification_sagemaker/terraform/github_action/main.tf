provider "aws" {
  region  = var.region
  profile = var.profile
}
terraform {
  backend "s3" {}
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}
locals {
  environments = ["dev", "prod"]
}

data "terraform_remote_state" "environments" {
  for_each = toset(local.environments)
  backend  = "s3"
  config = {
    bucket  = "tf-cf-templates"
    key     = "terraform-states/tenant-experience/${each.key}/terraform.tfstate"
    region  = "us-east-1"
    profile = "admin"
  }
}
