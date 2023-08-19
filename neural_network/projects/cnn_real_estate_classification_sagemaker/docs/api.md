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