import pandas as pd
import numpy as np

from sktime.forecasting.base import ForecastingHorizon
from sktime.split import SlidingWindowSplitter

from src.custom_utils import S3Helper
from src.base_trainer import BaseTrainer

class LongTrainer(BaseTrainer):
    """ 
    This class subclasses the BaseTrainer class and overrides the setup_cross_validation method to set up the cross-validation parameters for long-term forecasting. The two differences are as follows:
    
    1. We select either the 'original' or 'counterfactual' data depending on the `data_type` attribute
    2. We use the pandas PeriodIndex to represent the monthly periods in the data.
    
    The `data_type` attribute is used to select either the 'original' or 'counterfactual' data. The `data_type` attribute is set to 'original' by default, but can be set to 'counterfactual' to select the counterfactual data.
    """
    def __init__(self, horizon: str, config_path: str, logger_name: str, config_name: str, s3_helper: S3Helper = None, data_type: str = 'original') -> None:
        super().__init__(
            horizon=horizon, 
            config_path=config_path, 
            logger_name=logger_name, 
            config_name=config_name, 
            s3_helper=s3_helper
        )
        self.best_forecaster = None
        self.grid_search = None
        self.y_pred = None
        self.y_forecast = None
        self.oos_fh = None
        self.data_type = data_type
        
    def setup_cross_validation(self) -> None:
        """
        Override the base class method to set up the cross-validation parameters for long-term forecasting.
        """
        if (self._attribute_is_none('y_train')) and (self._attribute_is_none('y_test')):
            self.logger.info('Ingesting data...')
            data_dict = self.load_and_process_data()
            
            # Select either the original or counterfactual data
            data_cols = [col for col in data_dict['train'].columns if self.data_type in col]
            
            # Rename the columns to remove the data type prefix, e.g. 'original' or 'counterfactual'
            self.y_train = data_dict['train'][data_cols].rename(columns={col: col.replace(f'{self.data_type}_', '') for col in data_cols})
            self.y_test = data_dict['test'][data_cols].rename(columns={col: col.replace(f'{self.data_type}_', '') for col in data_cols})
            self.y_full = pd.concat([self.y_train, self.y_test], axis=0)
            
            # Set index to period
            self.y_train.index = pd.PeriodIndex(self.y_train.index, freq=self.config[self.horizon]['freq'])
            self.y_test.index = pd.PeriodIndex(self.y_test.index, freq=self.config[self.horizon]['freq'])
            self.y_full.index = pd.PeriodIndex(self.y_full.index, freq=self.config[self.horizon]['freq'])
        else:
            self.logger.info('Data already ingested, skipping ingestion...')
            
        if self._attribute_is_none('cv'):
            self.logger.info('Creating cross-validation splitter...')
            self.test_fh = ForecastingHorizon(self.y_test.index, is_relative=False)
            self.cv = SlidingWindowSplitter(
                fh=np.arange(1, len(self.test_fh)), 
                window_length=self.config[self.horizon]['window_length'], 
                step_length=self.config[self.horizon]['step_length']
            )
        else:
            self.logger.info('Cross-validation splitter already created, skipping creation...')