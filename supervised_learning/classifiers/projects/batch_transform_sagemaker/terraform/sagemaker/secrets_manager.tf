resource "random_string" "secret_suffix" {
  length  = 4 # Desired length of the string
  special = false
  upper   = false
  lower   = true
  numeric = true
}

# Git personal access token for connecting to private repositories
resource "aws_secretsmanager_secret" "git_pat" {
  name        = "${var.project_prefix}_git_pat_${random_string.secret_suffix.result}"
  description = "GitHub (classic) personal access token for SageMaker"
  tags = {
    Name = "${var.project_prefix}_git_pat"
  }
  depends_on = [random_string.secret_suffix]
}

resource "aws_secretsmanager_secret_version" "git_pat_version" {
  secret_id     = aws_secretsmanager_secret.git_pat.id
  secret_string = jsonencode({ username = var.git_username, password = var.git_pat })
  lifecycle {
    ignore_changes = [secret_string]
  }
  depends_on = [aws_secretsmanager_secret.git_pat]
}
