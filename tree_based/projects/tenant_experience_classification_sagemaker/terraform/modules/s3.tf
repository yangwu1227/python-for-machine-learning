resource "aws_s3_bucket" "batch_transform_input_output" {
  bucket = "${var.s3_bucket_prefix}-${var.env}"
  tags = {
    Environment = var.env
    Project     = var.project_prefix
  }
}

resource "aws_lambda_permission" "allow_s3" {
  statement_id  = "AllowS3Invoke"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.batch_transform_trigger.arn
  principal     = "s3.amazonaws.com"
  source_arn    = aws_s3_bucket.batch_transform_input_output.arn
}

resource "aws_s3_bucket_notification" "s3_trigger_notification" {
  bucket = aws_s3_bucket.batch_transform_input_output.id
  lambda_function {
    lambda_function_arn = aws_lambda_function.batch_transform_trigger.arn
    events              = ["s3:ObjectCreated:*"]
    filter_prefix       = var.s3_filter_prefix
  }

  depends_on = [aws_lambda_function.batch_transform_trigger, aws_lambda_permission.allow_s3]
}
