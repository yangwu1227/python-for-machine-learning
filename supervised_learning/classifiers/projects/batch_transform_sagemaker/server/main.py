import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import joblib
from fastapi import FastAPI

from server.api import endpoints, ping
from server.utils import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    A context manager to manage the startup and shutdown of the FastAPI application.
    The startup logic includes creating the logger and loading the model pipeline.
    The shutdown logic simply logs a message.

    Parameters
    ----------
    app : FastAPI
        The FastAPI app instance.

    Yields
    ------
    None
    """
    # Model directory and file path for SageMaker
    model_dir = "/opt/ml/model"
    model_path = os.path.join(model_dir, "model.joblib")

    try:
        logger.info("Starting up: Loading the model...")
        # Load the model and assign it to app state
        app.state.model_pipeline = joblib.load(model_path)
        logger.info("Model loaded successfully")
    except FileNotFoundError as error:
        logger.error(f"Model file not found at {model_path}: {error}")
        raise error
    except Exception as error:
        logger.error(f"Error loading model: {error}")
        raise error

    yield  # Separate startup and shutdown logic

    logger.info("Shutting down the server...")


app = FastAPI(lifespan=lifespan)
app.include_router(ping.router)
app.include_router(endpoints.router)
