# Cost

The cost of running this demo is based on a few factors:

1. **Instance Type**: The most significant cost factor is the instance type used for the SageMaker endpoint. The cost of the instance type is calculated per hour of usage.

    - For this project, I used the [g4dn.xlarge](https://instances.vantage.sh/aws/ec2/g4dn.xlarge), which has an on-demand price of $0.736 per hour of asynchronous inference usage in the US East (N. Virginia) region.

    - If only a single request is made, then the cost is negligible. If the autocaling sections are run exactly as in the notebook, i.e. ~ 100 invocations, the cost would be around $\$6$ Hrs  $\times \$0.736$= $\$4.42$.

2. Other costs including S3, Cloudwatch, and SNS (First 1,000 Amazon SNS Email/Email-JSON Notifications per month are free) are practically negligible.

Mileage may vary depending on the region, instance type, and actualy number of invocations. But if the exact steps in the notebook are followed, the cost should be around $\$3-\$5$.

See the [SageMaker pricing page](https://aws.amazon.com/sagemaker-ai/pricing/) for more details on the instance types and pricing.

Expore the [AWS Pricing Calculator](https://docs.aws.amazon.com/pricing-calculator/latest/userguide/what-is-pricing-calculator.html) for additional cost estimation.

---

# Prerequisites

## AWS CLI Setup

Follow the instructions in the [official AWS CLI documentation](https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-getting-started.html) to install and configure the AWS CLI.

### Configuring Credentials

Set up the AWS CLI with a named profile specifically for this project. For demonstration purposes only, this profile can be assigned the [AdministratorAccess](https://docs.aws.amazon.com/aws-managed-policy/latest/reference/AdministratorAccess.html) policy. 

> This does not adhere to the principle of least privilege, but it will have all required permissions for this demo.

This is read in at the top of `notebooks/model_endpoint_setup.ipynb`:

```python
profile = os.getenv("AWS_PROFILE", "default")
```

If fine-tuned, the profile should at least have permissions to perform the following tasks:

- Configure, deploy, and delete SageMaker endpoints
- [CreateTopic](https://docs.aws.amazon.com/sns/latest/api/API_CreateTopic.html) and [DeleteTopic](https://docs.aws.amazon.com/sns/latest/api/API_DeleteTopic.html) for SNS
- S3 buckets permissions for uploading, downloading, and deleting objects
- Application Auto Scaling permissions
    - [DeregisterScalableTarget](https://docs.aws.amazon.com/autoscaling/application/APIReference/API_DeregisterScalableTarget.html) and [RegisterScalableTarget](https://docs.aws.amazon.com/autoscaling/application/APIReference/API_RegisterScalableTarget.html)
    - [PutScalingPolicy](https://docs.aws.amazon.com/autoscaling/application/APIReference/API_PutScalingPolicy.html) and [DeleteScalingPolicy](https://docs.aws.amazon.com/autoscaling/application/APIReference/API_DeleteScalingPolicy.html)
    - [DescribeScalingActivities](https://docs.aws.amazon.com/autoscaling/application/APIReference/API_DescribeScalingActivities.html)
    - [TagResource](https://docs.aws.amazon.com/autoscaling/application/APIReference/API_TagResource.html) and [UntagResource](https://docs.aws.amazon.com/autoscaling/application/APIReference/API_UntagResource.html)

## S3 Bucket and IAM Role for SageMaker

An S3 bucket is required to:

1. Store the model artifact
2. Store the inference input video file
3. Store the inference output video files

In addition, an IAM role for SageMaker with the following permissions is required:

### S3 Access

Grant the IAM role access to the S3 bucket for managing model artifacts and inference files. E.g., for the bucket `async-inference-od-demo`:

```json
{
    "Statement": [
        {
            "Action": ["s3:*"]
            "Effect": "Allow",
            "Resource": [
                "arn:aws:s3:::async-inference-od-demo",
                "arn:aws:s3:::async-inference-od-demo/*"
            ]
        }
    ],
    "Version": "2012-10-17"
}
```

### SNS Permissions

Allow SageMaker to publish to SNS topics for status notifications. These are created in the `notebooks/model_endpoint_setup.ipynb` notebook; ensure that the ARNs match the SNS topics created in the notebook.

```json
{
    "Statement": [
        {
            "Action": "sns:*",
            "Effect": "Allow",
            "Resource": [
                "arn:aws:sns:{region}:{account-id}:async-inference-od-demo-success",
                "arn:aws:sns:{region}:{account-id}:async-inference-od-demo-failure"
            ]
        }
    ],
    "Version": "2012-10-17"
}
```

### SageMaker Permissions

Attach the [AmazonSageMakerFullAccess](https://docs.aws.amazon.com/aws-managed-policy/latest/reference/AmazonSageMakerFullAccess.html) policy to the IAM role.

### Trust Relationship

Ensure the IAM role has a trust relationship with SageMaker:

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": {
                "Service": "sagemaker.amazonaws.com"
            },
            "Action": "sts:AssumeRole"
        }
    ]
}
```

The IAM role and S3 bucket can be created manually or automated using Terraform.

---

## Terraform (Optional)

Install Terraform by following the instructions in the [official Terraform documentation](https://developer.hashicorp.com/terraform/tutorials/aws-get-started/install-cli).

### Backend Configuration

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

---

# Development Environment

I used [uv](https://docs.astral.sh/uv/getting-started/installation/) to manage the dependencies:

```bash
$ uv sync --frozen --group notebook --group lint-fmt 
```

This is very optional and a `requirements.txt` file is provided for other setups.

---

# Model Deployment Overview

The model deployment process is detailed in the `notebooks/model_endpoint_setup.ipynb` notebook. The notebook covers the following steps:

1. Download the pre-trained [model](https://pytorch.org/vision/main/models/generated/torchvision.models.detection.maskrcnn_resnet50_fpn.html#torchvision.models.detection.MaskRCNN_ResNet50_FPN_Weights) weights

    - We won't do any fine-tuning in this demo, but simply use the pre-trained weights for inference; the model should be able to detect object classes it was trained on (e.g., [COCO datasets](https://cocodataset.org/#home))

2. Create the model artifact and upload it to S3; see more details in [Using PyTorch with the SageMaker Python SDK](https://sagemaker.readthedocs.io/en/v1.50.4/using_pytorch.html#using-third-party-libraries)

3. Create the SNS topics for status notifications

4. Create the model entity in SageMaker

5. Create the endpoint configuration and deploy the model as an endpoint

6. Test the endpoint with an inference request

    - Upload an example video file to S3
    - Send an inference request to the endpoint
    - Download the inference output video file from S3
    - Generate a GIF with bounding boxes

7. Optionally, experiment with autoscaling the endpoint

8. Clean up the resources

## Source Code

The `src` directory contains the source code for model deployment. This code is copied onto the container when the model is deployed as an endpoint.

The model used in this demo is a [Mask R-CNN](https://arxiv.org/abs/1703.06870) model implemented in PyTorch.

For details on the input and output formats for this specific model in this demo, refer to the [Pytorch Mask R-CNN documentation](https://pytorch.org/vision/main/models/generated/torchvision.models.detection.maskrcnn_resnet50_fpn.html#torchvision.models.detection.maskrcnn_resnet50_fpn).

### Inference Script

The `inference.py` script is the entry point for model inference.

Since we used a pre-built docker image already set up with [torch](https://pytorch.org/), [torchvision](https://pytorch.org/vision/stable/index.html), and [torchserve](https://pytorch.org/serve/) (as supposed to building our custom inference server), SageMaker has requirements for which functions must be implemented in the `inference.py` script.

Documentations for the required functions can be found [here](https://docs.aws.amazon.com/sagemaker/latest/dg/neo-deployment-hosting-services-prerequisites.html).

At a high-level, the following functions are required in the `inference.py` script:

- `model_fn(model_dir)`: When an endpoint is created and passes the health check, SageMaker calls this function to load the model into memory. The model artifact is downloaded from S3 to the container at the path specified by `model_dir`.

- `transform_fn(model, request_body, request_content_type, response_content_type`: This function is called when an inference request is made to the endpoint. The input data is passed in the `request_body` parameter, and the function should return the inference output.

### Model Utils

Some logic is abstracted into the `model_utils.py` script:

- **`get_logger(name: str)`:** Sets up and returns a logger with a specified name and standardized format for logging messages.

- **`get_device()`:** Dynamically determines and returns the available device (GPU or CPU) for computation.

- **`preprocess_frames(frames: List[np.ndarray])`:** Converts a list of image frames into a **normalized** tensor suitable for model input.

- **`predict(batch_frames: torch.Tensor, model: MaskRCNN)`:** Runs inference on a batch of preprocessed frames using a MaskRCNN model and returns raw predictions.

- **`postprocess_predictions(predictions: List[Dict[str, torch.Tensor]])`:** Extracts and formats bounding boxes, labels, and scores from the model's raw predictions.

- **`batch_generator(temp_file: _TemporaryFileWrapper, frame_width: int, frame_height: int, frame_interval: int, batch_size: int)`:** Reads and processes video frames in batches, resizing and skipping frames as specified.

### Generate GIF

The `generate_gif.py` script generates a GIF from the inference output video file. The logic is adapted from this [notebook](https://github.com/aws-samples/amazon-sagemaker-asynchronous-inference-computer-vision/blob/main/visualization/generate_gif.ipynb).

1. **Frame Extraction**: Extracts frames at specified intervals, resizes them, and saves them as images.

2. **Annotation**: Annotates extracted frames using bounding boxes, labels, and scores from the JSON predictions, filtering based on a configurable score threshold. For instance, I set this to 0.9 by default.

3. **GIF Creation**: Combines the annotated frames into an animated GIF for visualizing detection results.

### Requirements.txt

This `requirements.txt` file is packaged with the source code and is used to install additional dependencies, e.g. `numpy`, `opencv-python` not included in the pre-built docker image.
