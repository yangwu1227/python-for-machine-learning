#!/bin/bash

############################################################################
# This script should be executed in the same directory that contains 'src' #
############################################################################

# Check if argument is provided, where [-z string]: True if the string is null (an empty string)
if [ -z "$1" ] || [ -z "$2" ] || [ -z "$3" ]; then
  echo "Please provide the entry point python script name, the custom image tag name, and the mode ('train' or 'serve')"
  exit 1
fi

# Variables
account_id=$(aws sts get-caller-identity --query Account --output text)
region=$(aws configure get region)
entry_point="$1"
image_tag="$2"
mode="$3"
# The image name must follow the following format, where the repository should already be created
image_name="$account_id.dkr.ecr.$region.amazonaws.com/ml-sagemaker:23.04-cuda11.8-base-ubuntu22.04-py3.10-sagemaker-$image_tag"

# Login to ECR
aws ecr get-login-password --region "$region" | docker login --username AWS --password-stdin "$account_id.dkr.ecr.$region.amazonaws.com"

# Location of 'src' is out of the Docker context so we need to run this bash script from the directory that contains 'src'
if [ "$mode" == "train" ]; then
    docker build --build-arg SAGEMAKER_PROGRAM="$entry_point" . -t "$image_name" -f ./docker/train.Dockerfile
elif [ "$mode" == "serve" ]; then
    docker build . -t "$image_name" -f ./docker/serve.Dockerfile
else
    echo "Invalid mode: $mode"
    exit 1
fi

docker tag "$image_name" "$image_name"

# Push to ECR
docker push "$image_name"
