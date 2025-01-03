resource "aws_s3_bucket" "s3_bucket" {
  bucket        = var.s3_bucket
  force_destroy = true

  tags = {
    Name = "${var.project_prefix}"
  }
}
