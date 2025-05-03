import io
import json
import os
from tempfile import NamedTemporaryFile, _TemporaryFileWrapper
from typing import Dict, List, Tuple, Union

import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.mask_rcnn import MaskRCNN

from model_utils import (
    BoxesType,
    LabelsType,
    ScoresType,
    batch_generator,
    get_device,
    postprocess_predictions,
    predict,
    preprocess_frames,
    setup_logger,
)

logger = setup_logger("async_inference")


def model_fn(model_dir: str) -> MaskRCNN:
    """
    Load the model from the model directory.

    Parameters
    ----------
    model_dir : str
        The directory where the model artifacts are stored.

    Returns
    -------
    MaskRCNN
        A MaskRCNN model instance.
    """
    logger.info(f"Loading model from: {model_dir}")
    device: torch.device = get_device()
    model: MaskRCNN = maskrcnn_resnet50_fpn()
    model.load_state_dict(
        torch.load(
            f"{model_dir}/model.pth",
            weights_only=True,
            map_location=device,
        )
    )
    logger.info("Model loaded successfully")
    return model


def transform_fn(
    model: MaskRCNN,
    request_body: bytes,
    request_content_type: str,
    response_content_type: str,
) -> Tuple[str, str]:
    """
    Perform inference on the input data using the model. This function
    combines `input_fn`, `predict_fn`, and `output_fn` into a single
    function.

    Parameters
    ----------
    model : MaskRCNN
        The model instance to use for inference.
    request_body : bytes
        The input data to perform inference on.
    request_content_type : str
        The content type of the input data; not used except for matching
        the SageMaker inference API signature.
    response_content_type : str
        The content type of the output data.

    Returns
    -------
    str
        The JSON-encoded predictions.
    """
    frame_interval: int = int(os.getenv("FRAME_INTERVAL", 30))
    frame_width: int = int(os.getenv("FRAME_WIDTH", 1024))
    frame_height: int = int(os.getenv("FRAME_HEIGHT", 1024))
    batch_size: int = int(os.getenv("BATCH_SIZE", 24))
    logger.info(f"Sampling every {frame_interval} frames")
    logger.info(f"Frames will be resized to {frame_width} x {frame_height}")
    logger.info(f"Batch size: {batch_size}")

    bytes_io: io.BytesIO = io.BytesIO(request_body)
    temp_file: _TemporaryFileWrapper = NamedTemporaryFile(delete=False)
    temp_file.write(bytes_io.read())

    logger.info("Starting inference")
    predictions: List[Dict[str, List[Dict]]] = []
    for batch, batch_frames in enumerate(
        batch_generator(
            temp_file,
            frame_width,
            frame_height,
            frame_interval,
            batch_size,
        ),
        1,
    ):
        logger.info(f"Batch {batch}: processing {len(batch_frames)} frames " + "-" * 50)
        preprocessed_frames: torch.Tensor = preprocess_frames(batch_frames)
        logger.info(f"Batch {batch}: successfully preprocessed frames")
        batch_predictions: List[Dict[str, torch.Tensor]] = predict(
            preprocessed_frames, model
        )
        logger.info(f"Batch {batch}: successfully predicted on frames")
        postprocessed_predictions: List[
            Dict[str, Union[BoxesType, LabelsType, ScoresType]]
        ] = postprocess_predictions(batch_predictions)
        logger.info(f"Batch {batch}: successfully postprocessed predictions")
        predictions.extend(postprocessed_predictions)
        logger.info(f"Batch {batch}: Finished processing " + "-" * 50)

    logger.info(f"Completed inference for {len(predictions)} frames")
    return json.dumps(predictions), response_content_type
