## Cost

The cost of running this demo is based on a few factors:

1. **Instance Type**: The most significant cost factor is the instance type used for the SageMaker batch transform job. The cost of the instance type is calculated per hour of usage.

    - For this project, I used [ml.c5.xlarge](https://instances.vantage.sh/aws/ec2/c5.xlarge), which has an on-demand price of $0.204 per hour of batch transform usage in the US East (N. Virginia) region.

    - If only a single batch transform job is made, then the cost is negligible. If the code blocks are run exactly as in the notebook, i.e. ~ 1 to 2 jobs, the cost would be at most around $\$0.2$ Hrs  $\times \$0.204 = \$0.04$.

2. **Notebook Instance**: If a notebook instance is used to run the demo, the cost is also calculated per hour of usage. 

    - For this proejct, I used [ml.t3.xlarge](https://instances.vantage.sh/aws/ec2/t3.xlarge), which has an on-demand price of $0.20 per hour of usage in the US East (N. Virginia) region.

    - The entire demo, including provisioning the resources, building the Docker image, creating the model, IO operations with S3, and running the batch transform job, can take as little as 30 minutes to a few hours. The cost may therefore range from $0.10 (for 30 minutes) to $0.60 (for 3 hours), depending on the duration of the process.

3. Other costs including S3, Cloudwatch, and lambda would be practically negligible.

Mileage may vary depending on the region, instance type, job duration, and the actualy number of jobs triggered. But if the exact steps in the notebook are followed, the cost should be around $\$3-\$5$.

See the [SageMaker pricing page](https://aws.amazon.com/sagemaker-ai/pricing/) for more details on the instance types and pricing.

Expore the [AWS Pricing Calculator](https://docs.aws.amazon.com/pricing-calculator/latest/userguide/what-is-pricing-calculator.html) for additional cost estimation.

---

## AWS CLI Setup

Follow the instructions in the [official AWS CLI documentation](https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-getting-started.html) to install and configure the AWS CLI.

### Configuring Credentials

Set up the AWS CLI with a named profile specifically for this project. For demonstration purposes only, this profile can be assigned the [AdministratorAccess](https://docs.aws.amazon.com/aws-managed-policy/latest/reference/AdministratorAccess.html) policy. 

> This does not adhere to the principle of least privilege, but it will have all required permissions for this demo.

If fine-tuned, the profile should at least have permissions to perform the following tasks:

- Configure, deploy, and delete SageMaker models, endpoints, and batch transform jobs. Start with the [AmazonSageMakerFullAccess](https://docs.aws.amazon.com/aws-managed-policy/latest/reference/AmazonSageMakerFullAccess.html) policy.

- S3 buckets permissions for uploading, downloading, and deleting objects

- Lambda permissions for creating, updating, and deleting functions

--- 

## Terraform 

Install Terraform by following the instructions in the [official Terraform documentation](https://developer.hashicorp.com/terraform/tutorials/aws-get-started/install-cli).

### Backend Configuration

> All `variables.tfvars` and `backend.hcl` files are provided as examples by adding the `-example` suffix; remove the suffix to use them.

To use a remote backend for Terraform state management, refer to `backend.hcl-example`, which configures the remote state storage in an S3 bucket and optionally specifies a DynamoDB table for state locking. This creates a chicken-and-egg situation: to manage this project's resources with Terraform using a remote state file, the S3 bucket for the state must already exist. For guidance on setting up remote state and state locking, see [this Stack Overflow answer](https://stackoverflow.com/a/48362341/12923148).

```hcl
bucket         = "s3-bucket-for-remote-state"
key            = "s3/key/to/remote-state.tfstate"
region         = "us-east-1"
profile        = "aws-profile"
dynamodb_table = "optional-dynamodb-table-name-for-state-locking"
```

Too much work for a demo? Use local state management, update the `backend` block in `main.tf`:

```hcl
terraform {
  backend "local" {
    path = "relative/path/to/terraform.tfstate"
  }
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}
```

### Variables

Variables are declared in `variables.tf` and values are set in `variables.tfvars-example`.

### Commands

```bash
terraform fmt
terraform init --backend-config="backend.hcl"
terraform validate
terraform apply --var-file="variables.tfvars"
terraform destroy --var-file="variables.tfvars"
```

### Overview of Terraform Modules

1. **SageMaker Module**  

   The SageMaker module provisions a stack of resources centered around Amazon SageMaker to demonstrate batch transform jobs. 

   - **Network Environment**: Configures the necessary networking setup for SageMaker like vpc, subnets (public only), and security groups.

   - **Notebook Instance**: Provides an interactive environment for running the demo. This may not be necessary if one's local machine is configuered and set up with AWS CLI and SageMaker SDK.

   - **Secrets Manager**: Clones a private repository to the notebook instance for version control. Again, this may not be necessary if all execution is done locally.

   - **ECR (Elastic Container Registry)**: Stores Docker images used during the demo. In this demo, we build a custom serving container for running the inference server code.

   - **S3**: Creates an S3 bucket to store the model artifacts. Typically, if any training is done, the training input and output data would also be stored here.

   - **IAM Role**: Grants permissions to interact with services such as S3 and ECR required for the demonstration. This execution role will be assumed by the sagemaker notebook as well as the model and batch transform jobs.

2. **Lambda Module**  

   The Lambda module sets up an event-driven workflow to demonstrate the triggering of batch transform jobs.  

   - **S3 Buckets with Notifications**: Configured to monitor uploads to specific S3 key patterns.  

   - **Lambda Function**: Automatically triggered by S3 notifications to start SageMaker batch transform jobs, allowing for seamless integration between S3, Lambda, and SageMaker.

   Useful links:

   - [Tutorial: Using an Amazon S3 trigger to invoke a Lambda function](https://docs.aws.amazon.com/lambda/latest/dg/with-s3-example.html)

3. **GitHub Action Module**  

   This module enables continuous deployment of the Lambda function code directly from the repository using GitHub Actions.

    **Benefits of Using OIDC with GitHub Actions**

    Implementing OIDC with GitHub Actions offers a secure method for authenticating workflows with AWS. It minimizes the risk of credential exposure and simplifies secret management. For additional details, refer to the following resources:

    - [Configuring OpenID Connect in Amazon Web Services](https://docs.github.com/en/actions/security-for-github-actions/security-hardening-your-deployments/configuring-openid-connect-in-amazon-web-services)
    - [GitHub Actions: Update on OIDC integration with AWS](https://github.blog/changelog/2023-06-27-github-actions-update-on-oidc-integration-with-aws/)

---

## Development Environment

I used [uv](https://docs.astral.sh/uv/getting-started/installation/) to manage the dependencies:

```bash
$ uv sync --frozen --all-groups
```

This is very optional and a `requirements.txt` file is provided for other setups.
