## Structure of Source Directory

The src directory contains entry point scripts, utilities, and the `requirements.txt` required to sucessfully run the image classification project on AWS SageMaker. The src directory contains the following files:

```
.
├── config
│   ├── baseline
│   │   └── baseline.yaml
│   ├── fine_tune
│   │   └── fine_tune.yaml
│   └── main.yaml
├── __init__.py
├── custom_utils.py
├── ingest_data.py
├── baseline_entry.py
├── fine_tune_entry.py
└── requirements.txt
```

* The `config` directory contains the `hydra` configuration yaml files. The typical configurations include:

    - AWS configurations: S3 bucket, region, framework version, etc.
    - Meta data for training: class labels, number of channels, image size, validation size, etc.
    - Other configurations for training: computing resources (instance types), spot instance set up, etc.
 
* The `ingest_data.py` script loads the raw data zip files from s3, reshapes the images, splits the data into train-val-test, and uploads the images as multi-dimensional (numpy) arrays. This part of the workflow is more ad-hoc and can benefit from more iterations and scoping as we move to better ways to store and access new images.

* The `baseline_entry.py` and `fine_tune_entry.py` scripts are entry points that are used for SageMaker training jobs and hyperparameter jobs.

    - For the baseline, the model trained is a 7-layer (5 convolutional and 2 fully-connected) cnn with batch normalization, max pooling,       and dropout.
    - For the fine-tuning, the model is first initialized and trained at the top (dense layers) and then fine-tuned with more layers released. More documentations can be found on confluence [here](https://hyperionpath.atlassian.net/wiki/spaces/ANALYTICS/pages/2680389640/Computer+Vision+Workflow+on+AWS+SageMaker).
 
    A note on the implementation is that all the modeling is carried out using tensorflow 2.8 to avoid a bug related to data augmentation in newer versions of the framework. The related github issue and stackoverflow post can be tracked [here](https://github.com/keras-team/keras-cv/issues/581) and [here](https://stackoverflow.com/questions/73304934/tensorflow-data-augmentation-gives-a-warning-using-a-while-loop-for-converting). Once the issue is resolved sufficiently, we can switch to newer versions of Tensorflow in the next iteration to take advantage of the new additions to Keras such as the focal loss function, which is an improved loss function for handling class imbalance.

* The `custom_utils.py` module contains utility functions for training and analysis.

When training begins, the files located in the `src` directory (including the `requirements.txt` file) will be copied onto the training docker image.

---

## Custom Utils

### Functions

---

*  `get_logger`: this is used to log training status updates and information to AWS CloudWatch.

    Example

    ```python
    from custom _utils import get_logger
    logger = get_logger(name='example')
    accuracy = 0.98
    logger.info(f'The accuray score {accuracy} will be logged to cloudwatch')
    ```

---

* `parser`: this is used to parser command line arguments passed to the entry point scripts. In hyperparameter tuning, SageMaker spawns many training jobs, each with a different combination of hyperparameters, which are passed to the training entry point via the command line. This `parser` function allows us to parse command line arguments that are *shared* between different scripts as well as those that are script-specific, e.g. `baseline`, `fine_tune`, and potentially other scripts. It is usually going to be used in conjunction with the `add_additional_args` decorator to allow it to parser arguments specific to the script at hand.

    Example

    ```python
    from custom_utils import parser, add_additiona_args

    additional_args = {
        'argument1': 1,
        'argument2': 'str'
    }

    # The 'parser' contains shared args for all scripts, but we decorate it to allow for additional script-dependent args
    args = add_additional_args(parser_func=parser, additional_args=additional_args)()

    assert args.argument1 == 1
    assert args.argument2 == 'str'
    ```

---

* `classification_report`: this function computes model performance metrics given the target, predictions, and a list of class label strings.

    Example

    ```python
    from custom_utils import classification_report

    assert y_train.shape == (num_samples, )
    assert y_pred.shape == (num_samples, )

    clf_report, agg_metrics = custom_utils.classification_report(y_train, y_pred, labels=['class_1', 'class_2', 'class_4'])

    clf_report
    ```

---

### Classes

* `AugmentationModel`: this class creates a parametrized stack of data augmentation layers, which is essentially a sequetial model. This class can be extended to include more data augmentation layers, which the user can specify using 'layer_name' and **kwargs pairs. The accepted 'layer_name' must be a valid keras class from `tf.keras.layers` and its kwargs must be passed as a dictionary, i.e., `{'argument_1: value, 'argument2': value}`.

    Example

    ```python
    from custom_utils import AugmentationModel

    data_augmentation = AugmentationModel(aug_params={
        'RandomFlip': {'mode': 'horizontal'},
        'RandomRotation': {'factor': 0.3},
        'RandomZoom': {'height_factor': 0.2, 'width_factor': 0.3},
        'RandomContrast': {'factor': 0.3}
    }).build_augmented_model()

    inputs = tf.keras.Input(shape=(10, ))
    x = data_augmentation(inputs)
    ```

---

* `TuningVisualizer`: this class implements a plotting function (perhaps more in the future) that creates parallel coordinate plot for visualizing hyperparamter tuning results.

    Example

    ```python
    from custom_utils import TuningVisualizer
    import pandas as pd

    assert isinstance(fine_tune_hpo_results, pd.DataFrame)

    categorical_params = ['cat_param1', 'cat_param2']
    numerical_params = ['num_param1', 'num_param2']

    viz = TuningVisualizer(tune_data=fine_tune_hpo_results, cat_params=categorical_params, num_params=numerical_params)
    
    viz.plot_parallel_coordinate(columns=, static=True, figsize=(1100, 700))
    ```

---

* `ErrorAnalyzer`: this class implements utilities for conducting error analysis. Specifically, we wish to inspect visually the types of incorrect predictions for a specific class. There are two main arguments, `sample_mis_clf: int = 10` and `sample_correct_clf: int = 5`, which sample the incorrect and correct predictions, respectively, for plotting. If left as default, the method automatically determines the number of samples to use when the number of incorrect and correct predictions are less than `10` or `5`, respectively. Empty plots should be taken to mean that there are no incorrect or correct prediction.

    Example

    ```python
    from custom_utils import ErrorAnalyzer

    error_analyzer = ErrorAnalyzer(
        y_true=y_train, # Shape (num_samples, )
        images=X_train, # Shape (num_samples, height, width, num_chanels)
        y_pred=train_predicted_prob_matrix, # Shape (num_smaples, num_classes)
        label_mapping=class_labels # Mapping from 'class_name' -> int
    )

    error_analyzer.plot_mis_clf(class_label='class_name', figsize=(12, 10));
    ```