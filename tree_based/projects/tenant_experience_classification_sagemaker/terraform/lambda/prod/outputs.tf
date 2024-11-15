output "lambda_function_arn" {
  value = module.batch_transform.lambda_function_arn
}

output "s3_bucket_arn" {
  value = module.batch_transform.s3_bucket_arn
}
