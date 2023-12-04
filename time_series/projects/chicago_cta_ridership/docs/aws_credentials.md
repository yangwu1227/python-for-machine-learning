## AWS Credentials

Data and model artifacts can be too large to store on github; local storage can make collaboration harder. For this project, I propose we use AWS [S3](https://docs.aws.amazon.com/AmazonS3/latest/userguide/Welcome.html) as a single source of truth for saving data and model artifacts. I have created the following s3-bucket and key:

* Bucket name: `yang-ml-sagemaker`
* Key name for the project: `chicago_cta_ridership`

The project key `chicago_cta_ridership` has the following structure:

```bash
├── chicago_cta_ridership
    ├── data
        ├── processed
        ├── raw
    ├── eda
    ├── models
```

To interact (read & write) with objects stored in the S3 bucket, we will need temporary credentials, which are refreshed every 12 hours:

* `AWS_ACCESS_KEY_ID`
* `AWS_SECRET_ACCESS_KEY`
* `AWS_SESSION_TOKEN`

These credentials are automatically updated every 12 hours and are made available as an encrypted artifact in the GitHub Action `Fetch and Encrypt AWS Temporary Credentials` of this repository.

Steps to Access the Credentials:

1. Navigate to the GitHub Actions tab of the repository.
2. Look for the most recent run of the `Fetch and Encrypt AWS Temporary Credentials` action.
3. Download the `encrypted-credentials` artifact in the `summary`.
4. Use the `decrypt_credentials.sh` script located in the `scripts` directory to decrypt the credentials:

```bash
$ cd project_root
$ bash ./scripts/decrypt_credentials.sh
```

Follow the on-screen prompts, entering the decryption passphrase when asked. The decrypted credentials will be saved to a .env file in your project's root directory.

### Additional Set Up

If you are cloning the project for the first time, you may need to run the following command:

```bash
# Activate the conda environment if necessary
$ poetry self add poetry-dotenv-plugin
```

### Accessing Credentials in Python

With this setup, the credentials can be accessed in Python as follows:

```python
import os

os.environ['AWS_ACCESS_KEY_ID']
os.environ['AWS_SECRET_ACCESS_KEY']
os.environ['AWS_SESSION_TOKEN']
```

**Important**: We should never need to hard-code these credentials in any script or notebook.