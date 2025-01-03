output "s3_bucket" {
  value       = aws_s3_bucket.s3_bucket.bucket
  description = "The name of the S3 bucket for storing the model artifacts"
}

output "sagemaker_execution_role_arn" {
  value       = aws_iam_role.sagemaker_execution_role.arn
  description = "The ARN of the execution role for SageMaker"
}

output "ecr_repository" {
  value       = aws_ecr_repository.ecr_repo.name
  description = "The name of the ECR repository for storing the Docker image"
}
