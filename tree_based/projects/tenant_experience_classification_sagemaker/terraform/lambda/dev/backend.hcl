bucket  = "tf-cf-templates"
key     = "terraform-states/tenant-experience/dev/terraform.tfstate" # Path to the state file in the S3 bucket
region  = "us-east-1"
profile = "admin" # The credentials profile in ~/.aws/config with permissions to interact with the S3 bucket
