# Security group for sagemaker allowing outbound traffic to the internet
resource "aws_security_group" "sagemaker_sg" {
  name_prefix = "${var.project_prefix}_sagemaker_sg"
  vpc_id      = aws_vpc.main.id
  tags = {
    Name = "${var.project_prefix}_sagemaker_sg"
  }
  depends_on = [aws_vpc.main]
}

resource "aws_vpc_security_group_egress_rule" "sagemaker_outbound" {
  security_group_id = aws_security_group.sagemaker_sg.id
  cidr_ipv4         = "0.0.0.0/0"
  ip_protocol       = "-1"
  depends_on        = [aws_security_group.sagemaker_sg]
  tags = {
    Name = "${var.project_prefix}_sagemaker_sg_outbound"
  }
}
