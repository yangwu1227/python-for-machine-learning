import sys
import os
import joblib
from contextlib import asynccontextmanager
from typing import Dict, Any

import uvicorn
from fastapi import FastAPI, status, Request, Response, HTTPException

import pandas as pd
from sktime.forecasting.compose import TransformedTargetForecaster
from sktime.transformations.series.fourier import FourierFeatures

from hydra import compose, initialize, core
from omegaconf import OmegaConf

def main():

    logger = get_logger(__name__)

    # ---------------------- Application lifespan management --------------------- #

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """ 
        A context manager to manage the startup and shutdown of the FastAPI application.
        The startup logic includes creating the logger, loading the configuration, model,
        transformer, and the data. The shutdown logic simply logs a message.

        Parameters
        ----------
        app : FastAPI
            The FastAPI app instance.
        """
        core.global_hydra.GlobalHydra.instance().clear()
        initialize(version_base='1.2', config_path='config', job_name='serve')
        app.state.config = OmegaConf.to_container(compose(config_name='main'), resolve=True)

        logger.info('Starting up: Loading model and transformer...')
        # Sagemaker copies model artifacts from S3 to this directory
        model_dir = app.state.config['model_dir']

        app.state.target_pipeline = joblib.load(os.path.join(model_dir, 'model_full_target_pipeline.joblib'))
        app.state.fourier_transformer = joblib.load(os.path.join(model_dir, 'model_full_fourier_transformer.joblib'))
        y_full = pd.read_csv(os.path.join(model_dir, 'model_full_data.csv'), index_col=0)
        y_full.index = pd.to_datetime(y_full.index)
        y_full.index.freq = app.state.config['freq']
        app.state.y_full = y_full

        logger.info('Model and transformer loaded...')

        yield  # This yield separates startup and shutdown logic

        logger.info('Shutting down the application...')

    # ----------------------------- Forecasting logic ---------------------------- #

    def forecast(periods: int, conf: float, app: FastAPI) -> str:
        """
        Forecast the gas production for the next fh periods. Note that the model pipeline,
        fourier transformer, and data are loaded into memory when the application starts,
        so they are scoped from the enclosing environment of this function.

        Parameters
        ----------
        periods : int
            The number of periods to forecast.
        conf : float
            The prediction interval coverage.
        app : FastAPI
            The FastAPI app instance containing the model, transformer, and data in its state.

        Returns
        -------
        str
            The forecasted gas production with prediction intervals. The json string contains
            a list of dictionaries, where each dictionary has the keys 'date', 'lower', 'prediction',
            and 'upper'.
        """

        try: 
            target_pipeline = app.state.target_pipeline
            fourier_transformer = app.state.fourier_transformer
            y_full = app.state.y_full
            config = app.state.config

            logger.info(f'Forecasting {periods} periods ahead...')
            fh = range(0, periods)
            conf = round(conf, 2)

            # Generate out-of-sample fourier features
            max_date = y_full.index.max()
            # Create a dummy series with fh steps ahead of the last in-sample date
            dummy_series = pd.Series(data=0, index=pd.period_range(start=max_date, periods=len(fh), freq=y_full.index.freq))
            # Convert back to period index, which is what the Fourier features transformer expects
            dummy_series.index = pd.PeriodIndex(data=dummy_series.index, freq=y_full.index.freq)
            X_oos = fourier_transformer.transform(dummy_series)

            # Forecast
            y_pred = target_pipeline.predict(fh=fh, X=X_oos)
            # The prediction interval returned by sktime has multi-level column names, so we need to index into the columns to get (lower, upper)
            logger.info(f'Using prediction interval: {conf * 100}%')
            pi = target_pipeline.predict_interval(fh=fh, X=X_oos, coverage=conf)[0][conf]

            # Create a dataframe with the predictions and prediction intervals
            predictions = pd.DataFrame(
                data={
                    f'lower_pi_{conf}': pi['lower'],
                    'predictions': y_pred,
                    f'upper_pi_{conf}': pi['upper']
                },
                index=dummy_series.index
            )
            # Convert index back to datetime
            predictions.index = predictions.index.to_timestamp(freq=y_full.index.freq)
            # Convert index back to string column
            predictions.reset_index(inplace=True, names=['date'])
            # Convert date to string
            predictions['date'] = predictions['date'].dt.strftime('%Y-%m-%d')

            return predictions.to_json()
        
        except Exception as e:
            logger.error(f'Error during forecasting: {str(e)}')
            raise HTTPException(status_code=500, detail='Error during forecasting')

    # ----------------------------- Application setup ---------------------------- #

    app = FastAPI(
        title='Gas Production Time Series Forecasting',
        lifespan=lifespan
    )

    @app.get('/ping')
    async def ping() -> Dict[str, str]:
        """
        Sagemaker sends a periodic GET request to /ping endpoint to check if the container is healthy. 

        Returns
        -------
        Dict[str, str]
            A dictionary with a single key, 'status', and value 'ok'.
        """
        return {'message': 'ok'}

    @app.post('/invocations')
    async def invocations(request: Request) -> Response:
        """
        Endpoint for Sagemaker to send POST requests to for inference. The
        request mimetype should be 'application/json' and the dictionary 
        should include two keys, 'periods' and 'conf', with values that are
        a positive integer and a float between 0 and 1, respectively.

        Parameters
        ----------
        request : Request
            The request object containing the payload.

        Returns
        -------
        Response
            The response object containing the predictions.
        """
        logger.info('Invoked with request...')
        # The body method is an asynchronous operation that reads the request body, so we need to await it
        body = await request.json()
        periods_str = body.get('periods')
        conf = body.get('conf')

        try:
            periods = int(periods_str)
            if periods <= 0:
                raise ValueError('Number of periods must be a positive integer')
            conf = float(conf)
            if conf <= 0 or conf >= 1:
                raise ValueError('Prediction interval coverage must be a float between 0 and 1')

            predictions = forecast(periods=periods, conf=conf, app=app)
            return Response(
                content=predictions,
                media_type='application/json',
                status_code=status.HTTP_200_OK
            )

        except ValueError as e:
            logger.error(f'Validation error: {str(e)}')
            raise HTTPException(status_code=400, detail=str(e))

        except Exception as e:
            logger.error(f'Error during invocation: {str(e)}')
            raise HTTPException(status_code=500, detail='Error during invocation')


    uvicorn.run(app, port=8080, host='0.0.0.0', log_level='info')


if __name__ == '__main__':

    from custom_utils import get_logger
    
    main()