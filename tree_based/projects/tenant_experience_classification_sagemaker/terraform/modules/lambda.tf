resource "aws_lambda_function" "batch_transform_trigger" {
  function_name    = "${var.project_prefix}_batch_transform_trigger_${var.env}"
  role             = aws_iam_role.lambda_execution_role.arn
  handler          = var.lambda_handler
  runtime          = var.lambda_runtime
  filename         = var.lambda_filename
  timeout          = var.lambda_timeout
  source_code_hash = filebase64sha256(var.lambda_filename)

  environment {
    variables = var.lambda_environment_variables
  }

  depends_on = [aws_iam_role_policy_attachment.lambda_execution_policy_attachment]

  lifecycle {
    ignore_changes = [
      # Ignore changes to the Lambda function code and hash
      filename,
      source_code_hash
    ]
  }

}
