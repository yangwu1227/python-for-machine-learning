output "lambda_function_arn" {
  value       = aws_lambda_function.batch_transform_trigger.arn
  description = "The ARN of the Lambda function"
}

output "s3_bucket_arn" {
  value       = aws_s3_bucket.batch_transform_input_output.arn
  description = "The ARN of the S3 bucket"
}
