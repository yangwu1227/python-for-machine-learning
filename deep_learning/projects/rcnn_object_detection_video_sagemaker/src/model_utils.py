import logging
import sys
from collections.abc import Callable
from tempfile import _TemporaryFileWrapper
from typing import Dict, Generator, List, TypeAlias, TypedDict

import numpy as np
import torch
from cv2 import INTER_AREA, VideoCapture, resize
from torchvision.models.detection.mask_rcnn import MaskRCNN
from torchvision.transforms import ToTensor

BoxesType: TypeAlias = List[List[float]]
LabelsType: TypeAlias = List[float]
ScoresType: TypeAlias = List[float]


class PredictionDict(TypedDict):
    boxes: BoxesType
    labels: LabelsType
    scores: ScoresType


def setup_logger(name: str) -> logging.Logger:
    """
    Parameters
    ----------
    name : str
        A string that specifies the name of the logger.

    Returns
    -------
    logging.Logger
        A logger with the specified name.
    """
    logger = logging.getLogger(name)

    log_format = "%(asctime)s %(levelname)s %(name)s: %(message)s"
    formatter = logging.Formatter(log_format)
    if not any(
        isinstance(handler, logging.StreamHandler) for handler in logger.handlers
    ):
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.setLevel(logging.INFO)

    return logger


def get_device() -> torch.device:
    """
    Dynamically get the device (GPU or CPU) based on the availability of GPU.

    Returns
    -------
    torch.device
        A torch.device instance.
    """
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


def preprocess_frames(frames: List[np.ndarray]) -> torch.Tensor:
    """
    Preprocess a list of frames to a torch tensor. The input frames are
    numpy arrays with shape (height, width, channels) in the range [0, 255].
    The output tensor has shape (batch_size, channels, height, width) in
    the range [0.0, 1.0].

    Parameters
    ----------
    frames : List[np.ndarray]
        A list of frames (numpy arrays) to preprocess.

    Returns
    -------
    torch.Tensor
        A tensor of shape (batch_size, channels, height, width).
    """
    preprocessor: ToTensor = ToTensor()
    transformed_frames: torch.Tensor = torch.stack(
        [preprocessor(frame) for frame in frames]
    )
    return transformed_frames


def predict(
    batch_frames: torch.Tensor, model: MaskRCNN
) -> List[Dict[str, torch.Tensor]]:
    """
    Predict for a batch of frames using a MaskRCNN model.

    Parameters
    ----------
    batch_frames : torch.Tensor
        A tensor of frames with shape (batch_size, channels, height, width).
    model : MaskRCNN
        A MaskRCNN model instance.

    Returns
    -------
    List[Dict[str, torch.Tensor]]
        A list of dictionaries `{boxes: torch.Tensor, labels: torch.Tensor,
        scores: torch.Tensor, masks: torch.Tensor}`.
    """
    with torch.no_grad():
        device: torch.device = get_device()
        model.to(device)  # This modifies the model in-place
        batch_frames_tensor: torch.Tensor = batch_frames.to(device)
        model.eval()
        predictions: List[Dict[str, torch.Tensor]] = model(batch_frames_tensor)
    return predictions


def postprocess_predictions(
    predictions: List[Dict[str, torch.Tensor]],
) -> List[PredictionDict]:
    """
    Postprocess the model predictions to extract the bounding boxes, labels, and scores.

    Parameters
    ----------
    predictions : List[Dict[str, torch.Tensor]]
        A list of dictionaries `{boxes: torch.Tensor, labels: torch.Tensor,
        scores: torch.Tensor, masks: torch.Tensor}`.

    Returns
    -------
    List[PredictionDict]
        A list of dictionaries `{boxes: List[List[float]], labels: List[float],
        scores: List[float]}`.
    """
    # See https://stackoverflow.com/questions/63582590/why-do-we-call-detach-before-calling-numpy-on-a-pytorch-tensor
    tensor_to_list: Callable = lambda tensor: tensor.detach().cpu().numpy().tolist()
    postprocessed_predictions: List[PredictionDict] = []
    for prediction in predictions:
        postprocessed_predictions.append(
            {
                "boxes": tensor_to_list(prediction["boxes"]),
                "labels": tensor_to_list(prediction["labels"]),
                "scores": tensor_to_list(prediction["scores"]),
            }
        )
    return postprocessed_predictions


def batch_generator(
    temp_file: _TemporaryFileWrapper,
    frame_width: int,
    frame_height: int,
    frame_interval: int,
    batch_size: int,
) -> Generator[List[np.ndarray], None, None]:
    """
    Generate batches of resized video frames at specified intervals.

    This function processes a video file, resizes frames, and yields them in batches.
    Frames are sampled based on a given interval and resized to the desired dimensions.

    Any remaining frames in the buffer are yielded when the video ends; when this
    happens, the buffer is guranteed to have fewer frames than the batch size.

    Parameters
    ----------
    temp_file : _TemporaryFileWrapper
        Temporary file wrapper object containing the video file.
    frame_width : int
        Desired width of each resized frame.
    frame_height : int
        Desired height of each resized frame.
    frame_interval : int
        Number of frames to skip between processed frames. For example,
        if `frame_interval=2`, every second frame is processed.
    batch_size : int
        Number of frames to include in each yielded batch.

    Yields
    ------
    List[np.ndarray]
        A list of frames (numpy arrays), each resized to (frame_height, frame_width, channels).

    Raises
    ------
    Exception
        If the video capture fails for the given file.
    """
    video_cap: VideoCapture = VideoCapture(temp_file.name)
    frame_index: int = 0
    frame_buffer: List[np.ndarray] = []

    while video_cap.isOpened():
        # Each frame is a numpy array with shape (height, width, channels)
        success, frame = video_cap.read()

        if not success:
            # Release capture if the video has ended
            video_cap.release()
            # Yield remaining frames in the buffer
            if frame_buffer:
                yield frame_buffer
            return None

        # For every (frame_interval)th frame, resize and add it to the buffer
        if frame_index % frame_interval == 0:
            frame_resized: np.ndarray = resize(
                frame, (frame_width, frame_height), interpolation=INTER_AREA
            )
            frame_buffer.append(frame_resized)

        # Once the buffer reaches the desired batch size, yield it
        if len(frame_buffer) == batch_size:
            yield frame_buffer
            frame_buffer.clear()

        # Increment the frame index to keep track of the frames processed
        frame_index += 1
    else:
        raise Exception(f"Video capture failed for {temp_file.name}")
