import argparse
import glob
import json
import os
from typing import List

import numpy as np
import torch
from cv2 import INTER_AREA, VideoCapture, imwrite, resize
from PIL import Image
from torchvision.io import read_image
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import draw_bounding_boxes

from model_utils import get_logger

logger = get_logger("generate_gif")

COCO_LABELS: List[str] = [
    "__background__",
    "person",
    "bicycle",
    "car",
    "motorbike",
    "aeroplane",
    "bus",
    "train",
    "truck",
    "boat",
    "trafficlight",
    "firehydrant",
    "streetsign",
    "stopsign",
    "parkingmeter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "hat",
    "backpack",
    "umbrella",
    "shoe",
    "eyeglasses",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sportsball",
    "kite",
    "baseballbat",
    "baseballglove",
    "skateboard",
    "surfboard",
    "tennisracket",
    "bottle",
    "plate",
    "wineglass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hotdog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "sofa",
    "pottedplant",
    "bed",
    "mirror",
    "diningtable",
    "window",
    "desk",
    "toilet",
    "door",
    "tvmonitor",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cellphone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "blender",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddybear",
    "hairdrier",
    "toothbrush",
    "hairbrush",
]


def video2frame(
    video_path: str,
    frame_width: int,
    frame_height: int,
    interval: int,
    output_path: str,
) -> List[np.ndarray]:
    """
    Extract frames from a video file at a specified interval and resize them. The interval
    determines how many frames are skipped before saving the next frame, resulting in a
    fps reduction.

    Parameters
    ----------
    video_path : str
        Path to the video file.
    frame_width : int
        Desired frame width.
    frame_height : int
        Desired frame height.
    interval : int
        Interval to extract frames; every `interval`-th frame is saved.
    output_path : str
        Directory to save extracted frames.

    Returns
    -------
    List[np.ndarray]
        List of extracted frames as NumPy arrays.
    """
    logger.info(f"Attempting to open video file: {video_path}")

    # Clear the output directory to avoid leftover files
    for file in glob.glob(os.path.join(output_path, "*.jpg")):
        os.remove(file)
        logger.info(f"Removed leftover file: {file}")

    video_frames: List[np.ndarray] = []
    video_cap: VideoCapture = VideoCapture(video_path)
    if not video_cap.isOpened():
        logger.error(f"Error opening video file: {video_path}")
        return []

    logger.info(f"Video file opened successfully: {video_path}")
    frame_index: int = 0
    frame_count: int = 0

    while video_cap.isOpened():
        success, frame = video_cap.read()
        if not success:
            break

        if frame_index % interval == 0:
            if frame is None or frame.size == 0:
                logger.warning(
                    f"Empty frame encountered at index {frame_index}. Skipping."
                )
                continue
            resize_frame = resize(
                frame, (frame_width, frame_height), interpolation=INTER_AREA
            )
            video_frames.append(resize_frame)
            imwrite(os.path.join(output_path, f"image-{frame_count}.jpg"), resize_frame)
            frame_count += 1

        frame_index += 1

    video_cap.release()
    logger.info(f"Number of frames extracted: {frame_count}")
    return video_frames


def save(imgs: List[torch.Tensor], img_num: int, output_path: str) -> None:
    """
    Save a list of images (tensor) to disk.

    Parameters
    ----------
    imgs : List[torch.Tensor]
        List of images as PyTorch tensors.
    img_num : int
        Image index for file naming.
    output_path : str
        Directory to save images.
    """
    if not imgs:
        logger.warning(f"No images to save for frame {img_num}")
        return

    for img in imgs:
        img_tensor: torch.Tensor = img.detach()
        img_pil: Image.Image = to_pil_image(img_tensor)
        img_pil.save(os.path.join(output_path, f"image-{img_num}.jpg"))


def annotate_frames(
    frames_path: str, predictions_path: str, score_threshold: float
) -> None:
    """
    Annotate frames with bounding boxes and save the results.

    Parameters
    ----------
    frames_path : str
        Path to the directory containing extracted frames.
    predictions_path : str
        Path to the JSON file with predictions.
    score_threshold : float
        Minimum score threshold for displaying bounding boxes.
    """
    if not os.path.exists(predictions_path):
        logger.error(f"Predictions file not found: {predictions_path}")
        return

    with open(predictions_path, "r") as file:
        predictions = json.load(file)

    annotated_frames_path = os.path.join(frames_path, "annotated")
    os.makedirs(annotated_frames_path, exist_ok=True)

    num_frames = len(predictions)
    for i in range(num_frames):
        pred = predictions[i]
        frame_path = os.path.join(frames_path, f"image-{i}.jpg")
        if not os.path.exists(frame_path):
            logger.error(f"Frame file not found: {frame_path}")
            continue

        image = read_image(frame_path)

        if not pred.get("boxes") or not pred.get("scores"):
            logger.warning(f"Frame {i} has no valid predictions.")
            annotated_image = to_pil_image(image)
            annotated_image.save(os.path.join(annotated_frames_path, f"image-{i}.jpg"))
            continue

        scores = torch.tensor(pred["scores"])
        boxes = torch.tensor(pred["boxes"])
        labels = torch.tensor(pred.get("labels", []))

        # Filter predictions
        valid_indices = scores > score_threshold
        valid_boxes = boxes[valid_indices]
        valid_labels = labels[valid_indices]
        valid_scores = scores[valid_indices]

        if len(valid_boxes) == 0:
            logger.info(f"Frame {i}: No predictions above threshold.")
            annotated_image = to_pil_image(image)
            annotated_image.save(os.path.join(annotated_frames_path, f"image-{i}.jpg"))
            continue

        class_labels = [
            f"{COCO_LABELS[l]}: {s:.2f}" for l, s in zip(valid_labels, valid_scores)
        ]

        annotated_image = draw_bounding_boxes(
            image,
            boxes=valid_boxes,
            labels=class_labels,
            width=2,
            colors="red",
        )

        # Save the annotated image
        save([annotated_image], i, annotated_frames_path)


def create_gif(input_path: str, output_path: str, duration: int = 200) -> None:
    """
    Create a GIF from a sequence of image files.

    Parameters
    ----------
    input_path : str
        Directory containing input images.
    output_path : str
        Path to save the output GIF.
    duration : int, optional
        Duration of each frame in milliseconds, by default 200.
    """
    image_files = sorted(
        glob.glob(os.path.join(input_path, "image-*.jpg")),
        key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split("-")[1]),
    )

    if not image_files:
        logger.error("No frames found for GIF creation.")
        return

    frames = [Image.open(f) for f in image_files]
    frames[0].save(
        output_path,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0,
    )
    logger.info(f"GIF saved at: {output_path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Video Frame Extraction and Annotation"
    )
    parser.add_argument(
        "--video_path", type=str, required=True, help="Path to the video file"
    )
    parser.add_argument("--frame_width", type=int, default=1024, help="Frame width")
    parser.add_argument("--frame_height", type=int, default=1024, help="Frame height")
    parser.add_argument(
        "--frame_interval",
        type=int,
        default=30,
        help="Frame extraction interval; extract every n-th frame",
    )
    parser.add_argument(
        "--score_threshold", type=float, default=0.9, help="Detection score threshold"
    )
    parser.add_argument(
        "--predictions_path", type=str, required=True, help="Path to JSON predictions"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="visualization/annotated_frames",
        help="Directory to save annotated frames",
    )
    args = parser.parse_args()

    if args.frame_interval <= 0:
        parser.error("Frame interval must be greater than zero.")
    if not (0 <= args.score_threshold <= 1):
        parser.error("Score threshold must be between 0 and 1.")

    os.makedirs(args.output_path, exist_ok=True)

    logger.info(f"Extracting frames from video: {args.video_path}")
    video2frame(
        args.video_path,
        args.frame_width,
        args.frame_height,
        args.frame_interval,
        args.output_path,
    )

    logger.info("Annotating frames with predictions")
    annotate_frames(args.output_path, args.predictions_path, args.score_threshold)

    logger.info("Creating GIF from annotated frames")
    annotated_frames_path = os.path.join(args.output_path, "annotated")
    create_gif(
        annotated_frames_path,
        os.path.join(args.output_path, "annotated_frames.gif"),
    )

    return 0


if __name__ == "__main__":
    main()
