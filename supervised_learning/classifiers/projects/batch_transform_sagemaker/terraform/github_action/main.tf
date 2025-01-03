terraform {
  backend "s3" {}
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
    bucket  = var.s3_remote_state_bucket
    key     = "${var.s3_remote_state_key}/${each.key}/terraform.tfstate"
    region  = var.region
    profile = var.profile
  }
}
