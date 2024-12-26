# VPC and subnets
output "vpc" {
  description = "A reference to the created VPC"
  value       = aws_vpc.main.id
}

output "public_subnets" {
  description = "A list of the public subnets"
  value       = join(",", [aws_subnet.public["0"].id, aws_subnet.public["1"].id])
}

output "private_subnets" {
  description = "A list of the private subnets"
  value       = join(",", [aws_subnet.private["0"].id, aws_subnet.private["1"].id])
}

# Security groups
output "sagemaker_security_group" {
  description = "Security group for SageMaker notebook instance / training container"
  value       = aws_security_group.sagemaker_sg.id
}

# RDS
output "rds_cluster_endpoint" {
  description = "Cluster endpoint"
  value       = "${aws_rds_cluster.db_cluster.endpoint}:${aws_rds_cluster.db_cluster.port}"
}

output "rds_cluster_name" {
  description = "Name of cluster"
  value       = aws_rds_cluster.db_cluster.cluster_identifier
}

output "rds_creds_secret_arn" {
  description = "AWS Secrets Manager secret name for RDS/Aurora"
  value       = aws_secretsmanager_secret.db_secret.arn
}

output "rds_database_name" {
  description = "Database name in RDS/Aurora"
  value       = aws_rds_cluster.db_cluster.database_name
}
