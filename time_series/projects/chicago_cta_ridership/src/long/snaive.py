import warnings
from typing import Dict, List, Tuple, Union, Optional, Any

import pandas as pd
import numpy as np

from sktime.transformations.series.boxcox import LogTransformer
from sktime.transformations.compose import OptionalPassthrough
from sktime.transformations.series.detrend import Detrender
from sktime.transformations.series.detrend import Deseasonalizer
from sktime.forecasting.naive import NaiveForecaster

from sktime.forecasting.compose import TransformedTargetForecaster
from sktime.forecasting.compose import ColumnEnsembleForecaster
from sktime.forecasting.model_selection import ForecastingGridSearchCV

from src.custom_utils import S3Helper
from src.long.long_trainer import LongTrainer

class SNaiveTrainer(LongTrainer):
    """ 
    This class implements methods for forecasting using the seasonal naive method.
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

    def _create_model(self) -> ColumnEnsembleForecaster:
        """
        Create the seasonal naive modeling pipeline with the following steps:
        
            1. Log transform the target
            2. Tunable steps to detrend and deseasonalize the target
            3. Forecast using seasonal naive, tuning the 'strategy' parameter e.g., 'last', 'mean', 'drift', etc.
        
        Returns
        -------
        ColumnEnsembleForecaster
            A forecaster containing seasonal naive model pipeline for each target.
        """
        forecasters = []
        for target in self.config['targets']:
            
            target_pipeline = TransformedTargetForecaster([
                ('log_transform', LogTransformer()),
                ('detrend', OptionalPassthrough(transformer=Detrender(), passthrough=True)),
                ('deseasonalize', OptionalPassthrough(transformer=Deseasonalizer(sp=self.config[self.horizon]['m']), passthrough=True)),
                ('snaive', NaiveForecaster(sp=self.config[self.horizon]['m'], window_length=None))
            ])       
            
            forecasters.append(
                (target, target_pipeline, target)
            )
            
        ensemble_forecaster = ColumnEnsembleForecaster(forecasters=forecasters)
        
        return ensemble_forecaster
    
    def cross_validate(self, verbose: int = 1, n_jobs: int = -1, refit: bool = True):
        self.setup_cross_validation()
        
        if self.model is None:
            self.logger.info('Creating model...')
            self.model = self._create_model()
        else:
            self.logger.info('Model already created, skipping creation...')
            
        # Hyperparameter tuning
        grid = {}
        for target in self.config['targets']:
            grid[f'{target}__detrend__passthrough'] = [True, False]
            grid[f'{target}__deseasonalize__passthrough'] = [True, False]
            grid[f'{target}__snaive__strategy'] = ['last', 'mean', 'drift']
            
        grid_search = ForecastingGridSearchCV(
            forecaster=self.model, 
            cv=self.cv,
            strategy='refit', # Refit on each train-val split (more computationally expensive)
            param_grid=grid, 
            scoring=self.metric,
            n_jobs=n_jobs,
            refit=True, # Refit on entire data with the best params
            verbose=verbose,
            error_score='raise'
        )
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            warnings.simplefilter('ignore', category=FutureWarning)
            warnings.simplefilter('ignore', category=UserWarning)
            grid_search.fit(y=self.y_train)
            
        self.logger.info(f'Best params: {grid_search.best_params_}')
        self.logger.info(f'Best score: {grid_search.best_score_}')
        
        self.grid_search = grid_search
        self.best_forecaster = grid_search.best_forecaster_.clone()
        
        return None
    
    def forecast(self, level: float = 0.95) -> Dict[str, Union[pd.DataFrame, Dict[str, pd.DataFrame]]]:
        if self._attribute_is_none('grid_search'):
            raise ValueError('The grid search attribute is None, please run `cross_validate` method first')
        
        self.logger.info('Making predictions on test set...')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            warnings.simplefilter('ignore', category=FutureWarning)
            warnings.simplefilter('ignore', category=UserWarning)
            self.y_pred = self.grid_search.predict(fh=self.test_fh)
            pi = self.__class__.extract_prediction_intervals(
                pi=self.grid_search.predict_interval(fh=self.test_fh, coverage=level),
                level=level
        )
        rmse = self.metric(y_true=self.y_test, y_pred=self.y_pred)
        metric_name = self.config[self.horizon]['metric'].upper()
        self.logger.info(f'Test {metric_name}: {rmse}')
        
        return {
            'y_train': self.y_train,
            'y_test': self.y_test,
            'y_pred': self.y_pred,
            'pi': pi
        }
        
    def refit_and_forecast(self, level: float = 0.95) -> Dict[str, Union[pd.DataFrame, Dict[str, pd.DataFrame]]]:
        if self._attribute_is_none('best_forecaster'):
            raise ValueError('Best forecaster attribute is None, please run `cross_validate` method first')
        
        self.logger.info('Refitting model on entire data...')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            warnings.simplefilter('ignore', category=FutureWarning)
            warnings.simplefilter('ignore', category=UserWarning)
            self.best_forecaster.fit(y=self.y_full)
            
        # Make out-of-sample forecasts
        self.logger.info('Making out-of-sample forecasts...')
        self.oos_fh = np.arange(1, self.config[self.horizon]['forecast_horizon'] + 1) 
        if self.best_forecaster.is_fitted:
            self.y_forecast = self.best_forecaster.predict(fh=self.oos_fh)
        pi = self.__class__.extract_prediction_intervals(
            pi=self.best_forecaster.predict_interval(fh=self.oos_fh, coverage=level),
            level=level
        )
    
        return {
            'y_train': self.y_full,
            'y_pred': self.y_forecast,
            'pi': pi
        }
        
    def diagnostics(self, 
                    full_model: bool,
                    lags: int = None, 
                    auto_lag: bool = None) -> pd.DataFrame:
        if full_model:
            if self._attribute_is_none('best_forecaster'):
                raise ValueError('Best forecaster attribute is None, please run `cross_validate` method first')
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                warnings.simplefilter('ignore', category=FutureWarning)
                warnings.simplefilter('ignore', category=UserWarning)
                residuals = self.best_forecaster.predict_residuals(y=self.y_full)
        else:
            if self._attribute_is_none('grid_search'):
                raise ValueError('The grid search attribute is None, please run `cross_validate` method first')
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                warnings.simplefilter('ignore', category=FutureWarning)
                warnings.simplefilter('ignore', category=UserWarning)
                residuals = self.grid_search.best_forecaster_.predict_residuals(y=self.y_train)
        
        return self.__class__._diagnostic_tests(
            residuals=residuals,
            lags=lags,
            auto_lag=auto_lag
        )