# Lifecycle configuration scripts
resource "aws_sagemaker_notebook_instance_lifecycle_configuration" "lifecycle_config" {
  name      = "${replace(var.project_prefix, "_", "-")}-lifecycle-config"
  on_create = filebase64("${path.module}/lifecycle_scripts/install_codeserver.sh")
  on_start  = filebase64("${path.module}/lifecycle_scripts/setup_codeserver.sh")
}

# Private code repository to attach to noteboook instance
resource "aws_sagemaker_code_repository" "code_repo" {
  code_repository_name = "${replace(var.project_prefix, "_", "-")}-code-repo"
  git_config {
    repository_url = var.git_repo_url
    secret_arn     = aws_secretsmanager_secret.git_pat.arn
  }
  depends_on = [aws_secretsmanager_secret.git_pat]
}

# SageMaker notebook instance
resource "aws_sagemaker_notebook_instance" "notebook_instance" {
  name          = "${replace(var.project_prefix, "_", "-")}-notebook"
  instance_type = var.sagemaker_instance_type
  role_arn      = aws_iam_role.sagemaker_execution_role.arn
  # Place the notebook instance in the public subnet for internet access
  subnet_id               = aws_subnet.public["0"].id
  security_groups         = [aws_security_group.sagemaker_sg.id]
  volume_size             = var.sagemaker_volume_size
  lifecycle_config_name   = aws_sagemaker_notebook_instance_lifecycle_configuration.lifecycle_config.name
  default_code_repository = aws_sagemaker_code_repository.code_repo.code_repository_name
  tags = {
    Name = "${var.project_prefix}_sagemaker_notebook_instance"
  }
  depends_on = [
    aws_subnet.public,
    aws_iam_role.sagemaker_execution_role,
    aws_security_group.sagemaker_sg,
    aws_sagemaker_notebook_instance_lifecycle_configuration.lifecycle_config,
    aws_sagemaker_code_repository.code_repo
  ]
}
