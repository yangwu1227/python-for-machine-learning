from io import BytesIO

import polars as pl
from fastapi import APIRouter, HTTPException, Request, Response, status
from server.utils import get_logger

router = APIRouter()
logger = get_logger(__name__)


@router.post("/invocations")
async def invocations(request: Request) -> Response:
    """
    Handles inference requests from SageMaker batch transform.

    Parameters
    ----------
    request : Request
        The request object containing payload for inference.

    Returns
    -------
    Response
        The response object containing inference results.
    """
    try:
        logger.info("Inference request received")
        # Determine the content type and accept type
        content_type = request.headers.get("Content-Type", "").lower()
        accept_type = request.headers.get("Accept", "text/csv").lower()

        # Read the input data based on the content type
        input_data = await request.body()
        if not input_data:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="Empty request body"
            )
        if content_type == "text/csv":
            data = pl.read_csv(BytesIO(input_data))
        elif content_type == "application/x-parquet":
            data = pl.read_parquet(BytesIO(input_data))
        else:
            raise HTTPException(
                # Unsupported content type status code
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail=f"Unsupported content type: {content_type}; supported types: text/csv, application/x-parquet",
            )
        logger.info(f"Input data shape: {data.shape}")
        logger.info(f"Input data columns: {data.columns}")

        # Access the model pipeline from the app state
        model_pipeline = request.app.state.model_pipeline
        predicted_probs = model_pipeline.predict_proba(data)
        predictions = pl.DataFrame(
            {
                "predicted_class": predicted_probs.argmax(axis=1),
                "predicted_probability": predicted_probs.max(axis=1),
            }
        )
        logger.info(f"Successfully generated predictions: {predictions.shape}")

        if accept_type == "text/csv":
            response_content = predictions.write_csv().encode("utf-8")
        elif accept_type == "application/x-parquet":
            buffer = BytesIO()
            predictions.write_parquet(buffer)
            response_content = buffer.getvalue()
        else:
            raise HTTPException(
                # Unsupported accept type status code
                status_code=status.HTTP_406_NOT_ACCEPTABLE,
                detail=f"Unsupported accept type: {accept_type}; supported types: text/csv, application/x-parquet",
            )

        return Response(
            content=response_content,
            media_type=accept_type,
            status_code=status.HTTP_200_OK,
        )

    except HTTPException as error:
        logger.error(f"Client error during inference: {str(error)}")
        raise error

    except Exception as error:
        logger.error(f"Internal error during inference: {str(error)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(error)
        )
