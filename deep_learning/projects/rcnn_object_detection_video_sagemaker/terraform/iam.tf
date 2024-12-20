resource "aws_iam_role" "sagemaker_execution_role" {
  name = "${var.project_prefix}_sagemaker_execution_role"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action    = "sts:AssumeRole"
        Effect    = "Allow"
        Principal = { Service = "sagemaker.amazonaws.com" }
      }
    ]
  })
}

resource "aws_iam_policy" "s3_policy" {
  name = "${var.project_prefix}_s3_policy"
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "s3:*"
        Effect = "Allow"
        Resource = [
          "arn:aws:s3:::${var.s3_bucket}",
          "arn:aws:s3:::${var.s3_bucket}/*"
        ]
      }
    ]
  })
}

resource "aws_iam_policy" "sns_policy" {
  name = "${var.project_prefix}_sns_policy"
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sns:Publish"
        Effect = "Allow"
        Resource = [
          "arn:aws:sns:${var.region}:${data.aws_caller_identity.current.account_id}:${replace(var.project_prefix, "_", "-")}-success",
          "arn:aws:sns:${var.region}:${data.aws_caller_identity.current.account_id}:${replace(var.project_prefix, "_", "-")}-failure"
        ]
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "policies_attachments" {
  for_each = {
    s3_policy             = aws_iam_policy.s3_policy.arn
    sns_policy            = aws_iam_policy.sns_policy.arn
    sagemaker_full_access = "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess"
  }
  role       = aws_iam_role.sagemaker_execution_role.name
  policy_arn = each.value
  depends_on = [
    aws_iam_role.sagemaker_execution_role,
  ]
}
