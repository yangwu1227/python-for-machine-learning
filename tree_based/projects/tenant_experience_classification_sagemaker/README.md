## Tenant Experience Classification with SageMaker

This repository provides a framework for an end-to-end machine learning pipeline that processes customer communications data. The project involves unsupervised clustering followed by a classification model to refine predictions and insights. The training is performed using SageMaker training jobs, and the model is hosted using batch transform with a FastAPI-based web server.

### Project Overview

1. **Unsupervised Clustering**
   - The initial step involves applying unsupervised clustering to the customer communications data to identify business-interpretable groups.
   - The clustering results are used to generate cluster labels.
        1. A bad customer experience group with longer response times, large number of maintainance requests, and lower answer rates.
        2. A good customer experience group with shorter response times, fewer maintainance requests, and higher answer rates.

2. **Classification Model**
   - The cluster labels generated from the unsupervised clustering step are used as targets for a Gradient Boosting Machine (GBM) classification model.
   - The classification model is trained to predict the cluster labels based on the input features.

3. **Training on SageMaker**
   - The training of the classification model is performed using Amazon SageMaker training jobs.
   - The training pipeline includes data preprocessing, model training, and model evaluation.

4. **Batch Transform and Hosting**
   - The trained model is deployed using SageMaker Batch Transform for batch inference.
   - A FastAPI-based web server is used to handle inference requests and serve the predictions.

### Repository Structure

<center>

| Directory | Description |
|-----------|-------------|
| **Source Code** | |
| `src/` | Contains the source code for data processing, model training, and evaluation. |
| `src/plot_utils.py` | Utility functions for plotting and data visualization. |
| `src/xgboost_entry.py` | Entry point for training the XGBoost classification model. |
| `src/config/` | Configuration files for the project. |
| `src/config/main.yaml` | Configuration settings for AWS resources and model training. |
| `src/config/xgboost/xgboost.yaml` | Configuration settings for XGBoost training. |
| **Web Server** | |
| `server/` | Contains the FastAPI-based web server code. |
| `server/main.py` | Main application file for the FastAPI server. |
| `server/api/endpoints.py` | API endpoints for handling inference requests. |
| `server/start_server.py` | This is the entrypoint that dynamically selects the number of uvicorn workers based on the number of CPU cores available at runtime. |
| **Infrastructure** | |
| `terraform/` | Contains Terraform scripts for infrastructure setup related to the lambda function used to trigger SageMaker Batch Transform jobs. |
| `terraform/github_action/` | GitHub Actions role for updating the Lambda function using workflows. |
| `terraform/github_action/iam.tf` | IAM roles and policies for GitHub Actions. |
| `terraform/lambda/` | Lambda function code for triggering SageMaker Batch Transform jobs. |
| `terraform/lambda/code/` | Source code for the Lambda function. |
| `terraform/lambda/code/lambda_function.py` | Main Lambda function handler. |
| `terraform/lambda/code/package_and_zip.sh` | Script to package and zip the Lambda function code. |
| `terraform/lambda/dev/` | Terraform configuration for the development environment. |
| `terraform/lambda/prod/` | Terraform configuration for the production environment. |
| `terraform/modules/` | Reusable Terraform modules for setting up AWS resources. |
| `terraform/modules/iam.tf` | IAM roles and policies for the Lambda function. |
| `terraform/modules/s3.tf` | S3 bucket creation and notification configuration for batch transform input and output. |
| `terraform/modules/lambda.tf` | Lambda function configuration. |
| **Docker** | |
| `docker/` | Docker-related files for training and serving the model. |
| `docker/serve.Dockerfile` | Dockerfile for serving the trained model. |
| `docker/train.Dockerfile` | Dockerfile for training the model. |
| `docker/build_and_push.sh` | Utility script to build and push Docker images to ECR. |
| **CI/CD** | |
| `.github/workflows/` | GitHub Actions workflows for CI/CD. |
| `.github/workflows/lambda_deploy.yml` | Reusable workflow for deploying the Lambda function from S3. |
| `.github/workflows/lambda_deploy_dev.yml` | Workflow for deploying the Lambda function to the development environment. |
| `.github/workflows/lambda_deploy_prod.yml` | Workflow for deploying the Lambda function to the production environment. |

</center>

### Secrets Used in GitHub Actions

This project uses several secrets in the GitHub Actions workflows to securely manage sensitive information. Below is a list of the secrets and their descriptions:

- `AWS_ACCESS_KEY_ID`: The access key ID for AWS, used to authenticate and authorize actions performed by the GitHub Actions workflows.
- `AWS_SECRET_ACCESS_KEY`: The secret access key for AWS, used in conjunction with the access key ID to authenticate and authorize AWS actions.
- `GITHUB_TOKEN`: A token provided by GitHub to authenticate and authorize actions performed by the GitHub Actions workflows, such as triggering other workflows or accessing repository data.
- `DOCKERHUB_USERNAME`: The username for DockerHub, used to authenticate and authorize Docker image pushes.
- `DOCKERHUB_PASSWORD`: The password for DockerHub, used in conjunction with the username to authenticate and authorize Docker image pushes.
- `AWS_REGION`: The AWS region where resources will be deployed.
- `AWS_GITHUB_ACTIONS_ROLE_ARN`: The Amazon Resource Name (ARN) of the role to assume for GitHub Actions.
- `S3_BUCKET_PREFIX`: The prefix for the S3 bucket used in the project.
- `LAMBDA_FUNCTION_PREFIX`: The prefix for the Lambda function names used in the project.

The GitHub Actions workflows use OpenID Connect (OIDC) to securely authenticate and authorize actions with AWS, eliminating the need for long-lived AWS credentials.
