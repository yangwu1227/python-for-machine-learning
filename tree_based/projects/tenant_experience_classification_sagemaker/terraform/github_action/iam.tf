# OIDC provider
resource "aws_iam_openid_connect_provider" "github_oidc_provider" {
  count = var.create_github_oidc_provider == true ? 1 : 0
  url   = "https://token.actions.githubusercontent.com"
  client_id_list = [
    "sts.amazonaws.com"
  ]
  # https://github.blog/changelog/2023-06-27-github-actions-update-on-oidc-integration-with-aws/
  thumbprint_list = [
    "1c58a3a8518e8759bf075b76b750d4f2df264fcd",
    "6938fd4d98bab03faadb97b34396831e3780aea1"
  ]
  tags = {
    Name = "github_oidc_provider"
  }
}

locals {
  github_oidc_provider_arn = var.create_github_oidc_provider == true ? aws_iam_openid_connect_provider.github_oidc_provider[0].arn : var.existing_oidc_provider_arn
}

# IAM role for workflow
resource "aws_iam_role" "github_actions_role" {
  name = "${var.project_prefix}_iam_github_actions_role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17",
    Statement = [
      {
        Effect = "Allow",
        Principal = {
          Federated = local.github_oidc_provider_arn
        },
        Action = "sts:AssumeRoleWithWebIdentity",
        Condition = {
          StringEquals = {
            "token.actions.githubusercontent.com:aud" = "sts.amazonaws.com"
          },
          StringLike = {
            "token.actions.githubusercontent.com:sub" = "repo:${var.github_username}/${var.github_repo_name}:*"
          }
        }
      }
    ]
  })
  tags = {
    Name = "${var.project_prefix}_iam_github_actions_role"
  }
}

resource "aws_iam_policy" "github_lambda_deploy_policy" {
  name        = "${var.project_prefix}_github_lambda_deploy_policy"
  description = "Policy to grant least privilege access for deploying Lambda functions from GitHub Actions"
  policy = jsonencode({
    "Version" : "2012-10-17",
    "Statement" : [
      # Lambda
      {
        "Effect" : "Allow",
        "Action" : [
          "lambda:UpdateFunctionCode",
          "lambda:GetFunction"
        ],
        "Resource" : [
          data.terraform_remote_state.environments["dev"].outputs.lambda_function_arn,
          data.terraform_remote_state.environments["prod"].outputs.lambda_function_arn
        ]
      },
      {
        "Effect" : "Allow",
        "Action" : [
          # S3
          "s3:GetObject",
          "s3:PutObject",
        ],
        "Resource" : [
          "${data.terraform_remote_state.environments["dev"].outputs.s3_bucket_arn}/*",
          "${data.terraform_remote_state.environments["prod"].outputs.s3_bucket_arn}/*"
        ]
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "attach_github_lambda_policy" {
  role       = aws_iam_role.github_actions_role.name
  policy_arn = aws_iam_policy.github_lambda_deploy_policy.arn
}
