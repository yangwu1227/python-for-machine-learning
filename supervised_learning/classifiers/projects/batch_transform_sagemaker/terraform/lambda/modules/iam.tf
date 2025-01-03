# IAM Role for Lambda Function
resource "aws_iam_role" "lambda_execution_role" {
  name = "${var.project_prefix}_lambda_execution_role_${var.env}"
  assume_role_policy = jsonencode({
    "Version" : "2012-10-17",
    "Statement" : [
      {
        "Effect" : "Allow",
        "Principal" : {
          "Service" : "lambda.amazonaws.com"
        },
        "Action" : "sts:AssumeRole"
      }
    ]
  })
}

# IAM Policy for Lambda Function
resource "aws_iam_policy" "lambda_execution_policy" {
  name        = "${var.project_prefix}_lambda_execution_policy_${var.env}"
  description = "IAM policy for Lambda function to interact with S3, SageMaker, and CloudWatch"
  policy = jsonencode({
    "Version" : "2012-10-17",
    "Statement" : [
      # S3 
      {
        "Effect" : "Allow",
        "Action" : [
          "s3:ListBucket"
        ],
        "Resource" : [
          "arn:aws:s3:::${var.s3_bucket_prefix}-${var.env}",
          "arn:aws:s3:::${var.s3_bucket_prefix}-${var.env}/*"
        ]
      },
      # Sagemaker
      {
        "Effect" : "Allow",
        "Action" : [
          "sagemaker:CreateTransformJob"
        ],
        "Resource" : "*"
      },
      # CloudWatch 
      {
        "Effect" : "Allow",
        "Action" : "logs:CreateLogGroup",
        "Resource" : "arn:aws:logs:${var.region}:${var.account_id}:*"
      },
      {
        "Effect" : "Allow",
        "Action" : [
          "logs:CreateLogStream",
          "logs:PutLogEvents"
        ],
        "Resource" : [
          "arn:aws:logs:${var.region}:${var.account_id}:log-group:/aws/lambda/*"
        ]
      }
    ]
  })
}

# Attach the Policy to the Role
resource "aws_iam_role_policy_attachment" "lambda_execution_policy_attachment" {
  role       = aws_iam_role.lambda_execution_role.name
  policy_arn = aws_iam_policy.lambda_execution_policy.arn
}
