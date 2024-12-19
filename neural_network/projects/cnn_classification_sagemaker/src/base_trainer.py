import logging
import os
from typing import Any, Dict, List

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Nopep8
import tensorflow as tf

# ---------------------------- Base trainer class ---------------------------- #


class BaseTrainer(object):
    """
    This is the base trainer class, which contains initialization and model persistence methods. Both
    baseline model and transfer learning model classes in the entry point scripts inherit from this class.
    """

    def __init__(
        self,
        hyperparameters: Dict[str, Any],
        config: Dict[str, Any],
        job_name: str,
        train_dataset: tf.data.Dataset,
        val_dataset: tf.data.Dataset,
        train_class_weights: Dict[str, float],
        distributed: bool,
        strategy: tf.distribute.Strategy,
        model_dir: str,
        logger: logging.Logger,
    ) -> None:
        """
        Constructor for the BaselineTrainer class.

        Parameters
        ----------
        hyperparameters : Dict[str, any]
            A dictionary containing the hyperparameters for model training.
        config : Dict[str, Any]
            A dictionary containing the configuration for model training.
        job_name : str
            The name of the job.
        train_dataset : tf.data.Dataset
            A tf.data.Dataset object that contains the training data.
        val_dataset : tf.data.Dataset
            The validation data is recommend to be a repeated dataset.
        train_class_weights : Dict[str, float]
            Class weights for the training data.
        distributed : bool
            A boolean that specifies whether to use distributed training.
        strategy : tf.distribute.Strategy
            A tf.distribute.Strategy object that specifies the strategy for distributed training.
        model_dir : str
            Path to the directory where the model will be saved.
        logger : logging.Logger
            A logger object.

        Returns
        -------
        None
        """
        self.hyperparameters = hyperparameters
        self.config = config
        self.job_name = job_name
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.train_class_weights = train_class_weights
        self.distributed = distributed
        self.strategy = strategy
        self.model_dir = model_dir
        self.logger = logger

        return None

    def _create_optimizer(self, learning_rate: float) -> tf.keras.optimizers.Optimizer:
        """
        This function creates an Adam optimizer based on the hyperparameters.

        Parameters
        ----------
        learning_rate : float
            The learning rate for the optimizer, which must be reduced for transfer learning.
            So we need to pass the learning rate as a parameter.

        Returns
        -------
        tf.keras.optimizers.Optimizer
            An Adam optimizer.
        """
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=learning_rate,
            beta_1=self.hyperparameters["adam_beta_1"],
            beta_2=self.hyperparameters["adam_beta_2"],
            clipnorm=self.hyperparameters["adam_clipnorm"],
        )

        return optimizer

    def _create_loss_fn(self) -> tf.keras.losses.Loss:
        """
        This function creates a loss function based on the hyperparameters.
        The loss function can either be categorical cross entropy or focal loss.

        Returns
        -------
        tf.keras.losses.Loss
            A loss function.
        """
        if self.hyperparameters["use_focal_loss"]:
            loss_fn = tf.keras.losses.CategoricalFocalCrossentropy(
                alpha=self.hyperparameters["loss_alpha"],
                gamma=self.hyperparameters["loss_gamma"],
                from_logits=True,
                name="loss",
            )
        else:
            loss_fn = tf.keras.losses.CategoricalCrossentropy(
                from_logits=True, name="loss"
            )

        return loss_fn

    def _create_metrics(self) -> List[tf.keras.metrics.Metric]:
        """
        This function creates a list of metrics for model evaluation.

        Returns
        -------
        List[tf.keras.metrics.Metric]
            A list of metrics--- accuracy, precision, recall, and area under PR curve.
        """
        metrics = [
            tf.keras.metrics.CategoricalAccuracy(name="accuracy"),
            tf.keras.metrics.Recall(thresholds=0, name="recall"),
            tf.keras.metrics.Precision(thresholds=0, name="precision"),
            tf.keras.metrics.AUC(curve="PR", from_logits=True, name="auc_pr"),
        ]

        return metrics

    def _is_chief(self, task_type: str, task_id: int) -> bool:
        """
        This function checks if the current worker is the chief worker.

        Parameters
        ----------
        task_type : str
            The type of the task, which can be something other than 'worker'. But we only care about the 'worker' type, specifically, the chief worker.
        task_id : int
            The id of the task, which is customary to be 0 for the chief worker.

        Returns
        -------
        bool
            A boolean that specifies whether the current process is the chief worker.
        """
        return task_type == "worker" and task_id == 0

    def _create_temp_dir(self, dir: str, task_id: int) -> str:
        """
        This function creates a temporary directory for a given worker, specified by the task id. The temporary directories
        on the worker need to be unique to prevent errors resulting from multiple workers trying to write to the same location.

        Parameters
        ----------
        dir : str
            The path to the directory where the temporary directory will be created.
        task_id : int
            The id of the task to uniquely name the temporary directory.

        Returns
        -------
        str
            The path to the temporary directory.
        """
        # Unique temporary directory for worker with task id
        base_temp_dir = "worker_temp_" + str(task_id)
        full_temp_dir = os.path.join(dir, base_temp_dir)
        # This creates a directory and all parent/intermediate directories
        tf.io.gfile.makedirs(full_temp_dir)
        return full_temp_dir

    def _create_model_dir(self, model_dir: str, task_type: str, task_id: int) -> str:
        """
        This function creates a model directory for the given worker. If the worker is the chief worker,
        then the model directory will be returned as a string. Otherwise, a temporary directory will be
        created and returned as a string. This is so that the chief worker can save the model to the
        model directory, while the other workers can save the model to their respective temporary directories.

        Parameters
        ----------
        model_dir : str
            The path to the directory where the model directories will be created.
        task_type : str
            The type of the task.
        task_id : int
            The id of the task.

        Returns
        -------
        str
            The path to the model directory.
        """
        # Break up model_dir into its components (e.g. opt/ml/model -> 'opt/ml' and 'model')
        base_model_dir = os.path.dirname(model_dir)
        base_name = os.path.basename(model_dir)

        # If not the chief worker, create a temporary directory
        if not self._is_chief(task_type, task_id):
            base_model_dir = self._create_temp_dir("/tmp", task_id)

        # The variable 'base_model_dir' is now either the original model_dir ('opt/ml') or a temporary directory (f'worker_temp_{task_id}')
        return os.path.join(base_model_dir, base_name)

    def __del__(self) -> None:
        """
        Destructor for the BaselineTrainer class.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        # Delete the temporary directories
        os.system("rm -rf /tmp/worker_temp_*")

        return None
