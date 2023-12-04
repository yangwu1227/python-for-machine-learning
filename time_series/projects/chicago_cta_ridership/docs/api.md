## Important API

### Custom Utils

#### S3Helper

Data read and write usage:

```python
from src.custom_utils import S3Helper
s3_helper = S3Helper()
# Read from s3
data = s3_helper.read_parquet(obj_key='data/raw/chicago_cta_ridership.parquet')
# Write to s3
s3_helper.to_parquet(data=data, obj_key='data/processed/processed_data.parquet')
```

To list all objects in the project S3 bucket:

```python
from src.custom_utils import S3Helper
s3_helper = S3Helper()
s3_helper.list_objects()
```

---

#### SetUp

A utility class to set up a logger and read configuration parameters for each entry-point script, such as training, preprocessing, or prediction scripts.

Initialize the class with logger and configuration names and paths:

```python
from src.custom_utils import SetUp

# Initialize the SetUp class
setup = SetUp(
    logger_name='training_logger',
    config_name='training_config',
    config_path='config'
)
# Set up logger and configuration
logger, config = setup.setup()

# Now we can use logger to log messages
logger.info('Logger is set up')

# And use config as a dictionary containing your configuration parameters
print(config['parameter_key'])
```

---

#### CVHelper

`CVHelper` is a utility class providing static methods for calculating sliding window sizes, the number of splits for time-series cross-validation, and plotting cross-validation windows.

Calculate the maximum window size for cross-validation:

```python
from src.custom_utils import CVHelper
window_size = CVHelper.calculate_window_size(num_splits=5, n=100, h=10, s=5)
```

Calculate the number of splits for a dataset based on sliding window parameters:

```python
from src.custom_utils import CVHelper
num_splits = CVHelper.calculate_num_splits(n=100, h=10, s=5, w=20)
```

Plot the sliding windows for cross-validation given the window size, step size, and forecast horizon:

```python
from src.custom_utils import CVHelper
CVHelper.plot_cv_windows(w=20, s=5, h=10, start_date='2020-01-01', end_date='2020-12-31', freq='D')
```

---

### BaseTrainer

The `BaseTrainer` class serves as the foundational class for creating forecasting models in a machine learning project. It is designed to be inherited by other trainer classes that specify model behaviors for different forecasting horizons (short, medium, or long). It integrates with S3 for data storage and provides methods for data processing, cross-validation, forecasting, and diagnostics.

---

#### Main Methods

* `load_and_process_data`: Loads and processes the training and testing data from S3.

* `setup_cross_validation`: Prepares the cross-validation strategy based on the provided data.

* `cross_validate`: Placeholder for cross-validation logic, to be implemented in child classes.

* `forecast`: Placeholder for the forecasting logic, to be implemented in child classes.

* `refit_and_forecast`: Placeholder for refitting and forecasting logic, to be implemented in child classes.

* `diagnostics`: Placeholder for model diagnostic tests, to be implemented in child classes.

* `upload_trainer`: Uploads the trainer state to S3 for persistence.

* `download_trainer`: Downloads the trainer state from S3 for later use.