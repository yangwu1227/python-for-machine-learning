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

# Random password for connecting to rds
data "aws_secretsmanager_random_password" "db_password" {
  password_length            = 16
  exclude_characters         = "\"@/\\{}"
  require_each_included_type = true
}

resource "aws_secretsmanager_secret" "db_secret" {
  name        = "${var.project_prefix}_db_credentials_${random_string.secret_suffix.result}"
  description = "Credentials for the RDS cluster"
  tags = {
    Name = "${var.project_prefix}_db_secret"
  }
  depends_on = [random_string.secret_suffix]
}

resource "aws_secretsmanager_secret_version" "db_secret_version" {
  secret_id = aws_secretsmanager_secret.db_secret.id
  secret_string = jsonencode({
    username = "admin"
    password = data.aws_secretsmanager_random_password.db_password.random_password
  })
  lifecycle {
    ignore_changes = [secret_string]
  }
  depends_on = [
    aws_secretsmanager_secret.db_secret,
    data.aws_secretsmanager_random_password.db_password
  ]
}
