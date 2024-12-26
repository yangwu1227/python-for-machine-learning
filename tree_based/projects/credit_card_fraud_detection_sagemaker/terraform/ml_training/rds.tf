# DB subnet group
resource "aws_db_subnet_group" "db_subnet_group" {
  name        = "${var.project_prefix}_db_subnet_group"
  description = "Subnet group for RDS cluster"
  subnet_ids  = [aws_subnet.private["0"].id, aws_subnet.private["1"].id]
  tags = {
    Name = "${var.project_prefix}_db_subnet_group"
  }
  depends_on = [aws_subnet.private["0"], aws_subnet.private["1"]]
}

# DB cluster parameter group
resource "aws_rds_cluster_parameter_group" "db_cluster_param_group" {
  name        = "${replace(var.project_prefix, "_", "-")}-db-cluster-param-group"
  family      = var.db_cluster_parameter_group_family
  description = "Cluster parameter group for ${var.db_engine}"
  parameter {
    name  = "time_zone"
    value = "US/Eastern"
  }
  tags = {
    Name = "${var.project_prefix}_db_cluster_param_group"
  }
}

# DB parameter group
resource "aws_db_parameter_group" "db_param_group" {
  name        = "${replace(var.project_prefix, "_", "-")}-param-group"
  family      = var.db_cluster_parameter_group_family
  description = "Parameter group for ${var.db_engine}"
  tags = {
    Name = "${var.project_prefix}_db_param_group"
  }
}

# RDS cluster
resource "aws_rds_cluster" "db_cluster" {
  cluster_identifier     = "${replace(var.project_prefix, "_", "-")}-db-cluster"
  engine                 = var.db_engine
  engine_version         = var.db_engine_version
  master_username        = jsondecode(aws_secretsmanager_secret_version.db_secret_version.secret_string).username
  master_password        = jsondecode(aws_secretsmanager_secret_version.db_secret_version.secret_string).password
  database_name          = var.database_name
  db_subnet_group_name   = aws_db_subnet_group.db_subnet_group.name
  vpc_security_group_ids = [aws_security_group.rds_sg.id]
  port                   = var.db_port
  skip_final_snapshot    = true
  tags = {
    Name = "${var.project_prefix}_db_cluster"
  }
  depends_on = [
    aws_secretsmanager_secret_version.db_secret_version,
    aws_db_subnet_group.db_subnet_group,
    aws_security_group.rds_sg
  ]
}

# RDS cluster instance
resource "aws_rds_cluster_instance" "db_instance" {
  count                = 1
  identifier           = "${replace(var.project_prefix, "_", "-")}-db-instance-${count.index + 1}"
  cluster_identifier   = aws_rds_cluster.db_cluster.id
  instance_class       = var.db_instance_type
  engine               = var.db_engine
  db_subnet_group_name = aws_db_subnet_group.db_subnet_group.name
  tags = {
    Name = "${var.project_prefix}_db_instance"
  }

  depends_on = [
    aws_rds_cluster.db_cluster,
    aws_db_subnet_group.db_subnet_group
  ]
}
