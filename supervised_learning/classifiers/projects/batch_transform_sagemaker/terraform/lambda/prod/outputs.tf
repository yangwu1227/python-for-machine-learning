output "s3_bucket" {
  value       = var.lambda_environment_variables["S3_BUCKET"]
  description = "The S3 bucket used for batch transform"
}

output "s3_bucket_arn" {
  value       = module.batch_transform.s3_bucket_arn
  description = "The ARN of the S3 bucket for batch transform"
}

output "s3_key_input" {
  value       = var.lambda_environment_variables["S3_KEY_INPUT"]
  description = "The input S3 key for batch transform"
}

output "s3_key_output" {
  value       = var.lambda_environment_variables["S3_KEY_OUTPUT"]
  description = "The output S3 key for batch transform"
}

output "lambda_function_arn" {
  value       = module.batch_transform.lambda_function_arn
  description = "The ARN of the Lambda function for batch transform"
}
