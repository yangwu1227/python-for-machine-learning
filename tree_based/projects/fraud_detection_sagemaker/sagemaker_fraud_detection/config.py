import os
import boto3
import sagemaker

# -------------------- Define aws configuration variables -------------------- #


# Get execution role
default_role = sagemaker.get_execution_role()

# A session stores configuration state and allows us to create service clients and resources
boto3_session = boto3.session.Session()
region_name = boto3_session.region_name

# Target s3 paths and prefixes
s3_prefix = "fraud_detection_project"
s3_bucket = "yang-ml-sagemaker"
s3_raw_data_prefix = "raw_data"
s3_processing_output = "preprocessed_data"
s3_train_output = "training_output"

role = default_role
