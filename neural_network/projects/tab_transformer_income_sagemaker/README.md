## Census Income Classification

This project tackles a binary classification (structure data) problem; the task is to predict whether a person is likely to be making over USD 50,000 a year. We use the [TabTransformer](https://arxiv.org/abs/2012.06678) architecture, which is an advanced modeling structure designed for both supervised and semi-supervised learning, especially for structured data. It leverages the power of Transformer layers, which are grounded in self-attention mechanisms. These layers enhance the embeddings of categorical variables, producing context-rich embeddings.

The data can be loaded using the `ingest_upload.py` script from the URLs specified in the configuration files.

## Environment

To reproduce the development environment in SageMaker:

```
$ source activate tensorflow2_p310
# Or
$ source ~/anaconda3/etc/profile.d/conda.sh
$ conda activate tensorflow2_p310
$ pip install -r ./src/requirements.txt
``` 

## Structure of Source Directory

The src directory contains entry point scripts, utilities, and the `requirements.txt` required to successfully run the project on AWS SageMaker. The src directory contains the following files:

```
src
├── __init__.py
├── config
│   ├── main.yaml
│   └── tf_keras
│       └── tf_keras.yaml
├── custom_utils.py
├── ingest_upload.py
├── requirements.txt
├── serve
└── tf_keras_entry.py
```

* The `config` directory contains the `hydra` configuration yaml files. The typical configurations include:

    - AWS configurations: S3 bucket, region, framework version, etc.
    - Meta data for training: categorical and numerical features, unique categories in each categorical feature, etc.
    - Other configurations for training: computing resources (instance types), spot instance set up, etc.
 
* The `ingest_data.py` script loads the raw data from the web and uploads the train-val-test splits to s3.

* The `tf_keras_entry.py` script is entry point that are used for SageMaker training jobs and hyperparameter jobs. The implementation of the TabTransformer is contained within and we also use [Optuna](https://optuna.org/) for hyperparameter tuning. The CloudFormation yaml file that sets up the resources required for running Optuna on AWS can be found [here](https://github.com/aws-samples/amazon-sagemaker-optuna-hpo-blog).
 
All the modeling is carried out using tensorflow 2.13.0 and we take advantage of the new focal loss function, which is an improved loss function for handling class imbalance. This option can be toggled using the `use_focal_loss` hyperparameter.

* The `custom_utils.py` module contains utility functions for training and analysis.

When training begins, the files located in the `src` directory (including the `requirements.txt` file) will be copied onto the training docker image.