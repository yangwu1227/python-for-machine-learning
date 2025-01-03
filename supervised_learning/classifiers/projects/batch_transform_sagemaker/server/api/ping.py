from typing import Dict

from fastapi import APIRouter

router = APIRouter()


@router.get("/ping")
async def ping() -> Dict[str, str]:
    """
    Sagemaker sends a periodic GET request to /ping endpoint to check if the inference
    container is healthy.

    Returns
    -------
    Dict[str, str]
        A dictionary with a single key, 'status', and value 'ok'.
    """
    return {"status": "ok"}
