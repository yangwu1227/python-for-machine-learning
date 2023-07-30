import os
import argparse
import json
import subprocess
import shutil
import random
import functools
from typing import Tuple, Dict, List
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor

import sagemaker
import boto3
from sagemaker.session import Session

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

from hydra import core, initialize, compose
from omegaconf import OmegaConf

# -------------------------------- XML parser -------------------------------- #

def parse_file(filename: str, image_id: int, mappings: Dict[str, int]) -> Tuple[Dict[str, str], List[Dict[str, int]]]:
    """
    Parse an XML file and extract image data and annotations.

    Parameters
    ----------
    filename : str
        The path to the XML file.
    image_id : int
        The unique identifier of the image.
    mappings : Dict[str, int]
        A dictionary mapping class label names to category IDs, which are integers.

    Returns
    -------
    Tuple[Dict[str, str], List[Dict[str, int]]]
        A tuple containing a dictionary with image meta data and a list of dictionaries with annotations data.
    """
    # Parse XML file into element tree
    tree = ET.parse(filename)
    # Get root element
    root = tree.getroot()

    # Obtain image meta data
    file_name = root.find('filename').text
    width = int(root.find('size/width').text)
    height = int(root.find('size/height').text)
    image_meta_data = {
        'file_name': file_name,
        'height': height,
        'width': width,
        'id': image_id
    }

    annotations_data = []
    # For each object in the image
    for obj in root.iter('object'):
        # Get the label name
        name = obj.find('name').text
        # Map the string label to integer
        category_id = mappings[name]
        # Get the bounding box coordinates for the object
        bbox = [
            int(obj.find('bndbox/xmin').text),
            int(obj.find('bndbox/ymin').text),
            int(obj.find('bndbox/xmax').text),
            int(obj.find('bndbox/ymax').text),
        ]
        # Append bbox to the list of annotations with the same image_id
        annotations_data.append({
            'image_id': image_id,
            'bbox': bbox,
            'category_id': category_id
        })

    return image_meta_data, annotations_data

# --------------- Function for generating annotations json file -------------- #

def generate_annotations(directories: List[str], output_files: List[str], mappings: Dict[str, int]) -> None:
    """
    Generate annotations from XML files in multiple directories and write them to separate JSON files.

    Parameters
    ----------
    directories : List[str]
        The list of paths to the directories with XML files.
    output_files : List[str]
        The list of paths to the JSON files that will be written.
    mappings : Dict[str, int]
        A dictionary mapping class label names to category IDs, which are integers.

    Returns
    -------
    None
    """
    with ProcessPoolExecutor() as executor:
        for directory, output_file in zip(directories, output_files):
            # Structure of the json file
            data = {'images': [], 'annotations': []}

            # List of xml files
            xml_files = [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith('.xml')]
            # List of tuples that are pairs of (path_to_xml, image_id)
            args = [(xml_file, i) for i, xml_file in enumerate(xml_files)]
            # Partial function application to fix the mapping argument (image_id and filename are not fixed)
            fixed_parse_file = functools.partial(parse_file, mappings=mappings)

            # The *zip(*[tuple1, tuple2, ...]) unpacks the list of tuples into separate tuples
            # The *args unpacks the tuples into separate arguments 'xml_file' and 'image_id'
            for result in executor.map(fixed_parse_file, *zip(*args)):
                image_data, annotations_data = result
                if annotations_data:
                    data['images'].append(image_data)
                    data['annotations'].extend(annotations_data)

            with open(output_file, 'w') as f:
                json.dump(data, f)

# ----------------------------- Directory set up ----------------------------- #

def create_directories(data_dir: str, image_format: str = 'jpeg', val_split: bool = False) -> None:
    """
    Randomly sample 20% of the training images and move them to the validation directory.
    This function also sets up the directory structure for the built-in algorithm. For example:

    ```
    train/
    |--images
        |--abc.png
        |--def.png
    |--annotations.json
    ```

    Parameters
    ----------
    data_dir : str
        The path to the directory with the unzipped data.
    image_format : str
        The format of the images. Default: 'jpeg'.
    val_split: bool
        Whether to split the data into train-val-test.
    

    Returns
    -------
    None
    """
    # ------------------- Ensure images have corresponding xml ------------------- #

    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')
    # Initial images and xml files (use absolute paths)
    train_images = [os.path.join(train_dir, file) for file in os.listdir(train_dir) if file.endswith(image_format)]
    test_images = [os.path.join(test_dir, file) for file in os.listdir(test_dir) if file.endswith(image_format)]
    train_xmls = [os.path.join(train_dir, file) for file in os.listdir(train_dir) if file.endswith('.xml')]
    test_xmls = [os.path.join(test_dir, file) for file in os.listdir(test_dir) if file.endswith('.xml')]

    # Keep only the images that have corresponding xml files
    train_images = [train_image for train_image in train_images if train_image.replace(image_format, 'xml') in train_xmls]
    test_images = [test_image for test_image in test_images if test_image.replace(image_format, 'xml') in test_xmls]

    # Update xml files to match the images
    train_xmls = [train_xml for train_xml in train_xmls if train_xml.replace('xml', image_format) in train_images]
    test_xmls = [test_xml for test_xml in test_xmls if test_xml.replace('xml', image_format) in test_images]

    if val_split:
        
        # ------------------- Randomly sample 20% of the training images ------------------- #

        # Randomly sample 20% of the training images
        val_images = random.sample(train_images, int(len(train_images) * 0.2))
        # Take the set difference between the training and validation images
        train_images = list(set(train_images) - set(val_images))

        # Create xml files for the validation images
        val_xmls = [train_xml for train_xml in train_xmls if train_xml.replace('xml', image_format) in val_images]
        # The remaining xml files are for training
        train_xmls = list(set(train_xmls) - set(val_xmls))

    # ------------------- Create directories ------------------- #

    sub_dirs = ['train_data', 'val_data', 'test_data'] if val_split else ['train_data', 'test_data']
    list_of_image_files = [train_images, val_images, test_images] if val_split else [train_images, test_images]
    list_of_xml_files = [train_xmls, val_xmls, test_xmls] if val_split else [train_xmls, test_xmls]

    for sub_dir, image_files, xml_files in zip(sub_dirs, list_of_image_files, list_of_xml_files):
        # Within data_dir, create sub-directories train_data, validation_data, and test_data
        image_sub_dir = os.path.join(data_dir, sub_dir, 'images')
        xml_sub_dir = os.path.join(data_dir, sub_dir, 'annotations')
        os.makedirs(image_sub_dir, exist_ok=True)
        os.makedirs(xml_sub_dir, exist_ok=True)

        for image_file, xml_file in zip(image_files, xml_files):
            # Move images to the images subdirectory
            shutil.move(image_file, image_sub_dir)
            # Move xml files to the annotations subdirectory
            shutil.move(xml_file, xml_sub_dir)

    return None

# ------------ Function that plots the bounding boxes on the image ----------- #

def plot_bbox(data_dir: str, image_name: str) -> None:
    """
    Plot an image with its bounding boxes.

    Parameters
    ----------
    data_dir : str
        The path to the input directory containing the images and annotations.json file.
    image_name : str
        The name of the image file.

    Returns
    -------
    None
    """
    # Load annotations
    with open(os.path.join(data_dir, 'annotations.json'), 'r') as f:
        annotations = json.load(f)

    # Get image data
    image_data = next((image for image in annotations['images'] if image['file_name'] == image_name), None)
    if image_data is None:
        print(f'No image data found for {image_file_name}')
        return

    # Get bounding boxes for the image
    bounding_boxes = [annotation['bbox'] for annotation in annotations['annotations'] if annotation['image_id'] == image_data['id']]

    # Load image
    im = Image.open(os.path.join(data_dir, 'images', image_name))

    # Create figure and axes
    fig, ax = plt.subplots(1)
    ax.imshow(im)
    for bbox in bounding_boxes:
        xmin, ymin, xmax, ymax = bbox
        rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    plt.show()

# ------------------------------- Main program ------------------------------- #

def main() -> None:

    # ---------------------------------- Set up ---------------------------------- #

    parser = argparse.ArgumentParser()
    parser.add_argument('--val_split', action='store_true')
    args, _ =parser.parse_known_args()

    core.global_hydra.GlobalHydra.instance().clear()
    initialize(version_base='1.2', config_path='config', job_name='xml_to_json_annotations')
    config = OmegaConf.to_container(compose(config_name='main'), resolve=True)

    logger = get_logger(name=__name__)

    # ------------------------ Download zip files from s3 ------------------------ #

    logger.info('Downloading raw data zip files from s3...')

    # Create raw data directory
    raw_data_dir = os.path.join(
        os.path.dirname(  # Get the parent directory of the script directory
            os.path.dirname(  # Get the parent directory of the current script
                os.path.abspath(__file__)  # Get the absolute path of the current script
            )
        ),
        'data' 
    )
    if not os.path.exists(raw_data_dir):
        os.makedirs(raw_data_dir)

    # Unzipping
    file_names = ['train.zip', 'test.zip']
    for file_name in file_names:
        # Download the zip file from s3
        subprocess.run(
            f'aws s3 cp s3://{config["s3_bucket"]}/{config["s3_key"]}/raw-data/{file_name} {raw_data_dir}/{file_name}',
            shell=True
        )
        # Unzip the file
        subprocess.run(
            f'unzip -q {raw_data_dir}/{file_name} -d {raw_data_dir}',
            shell=True
        )

    logger.info('Clean-up by removing the zip files...')
    for file_name in file_names:
        os.remove(os.path.join(raw_data_dir, file_name))

    # ------------------------ Create directories ------------------------ #

    logger.info('Creating directories and setting up directories...')

    create_directories(data_dir=raw_data_dir, val_split=args.val_split)

    logger.info('Clean-up by removing original images and xml files...')
    # Get all files that are not 'train_data', 'val_data', or 'test_data' under raw_data_dir
    sub_dirs = ['train_data', 'val_data', 'test_data'] if args.val_split else ['train_data', 'test_data']
    files_to_remove = [os.path.join(raw_data_dir, file) for file in os.listdir(raw_data_dir) if file not in sub_dirs]
    # Remove the files
    for file in files_to_remove:
        if os.path.isdir(file):
            shutil.rmtree(file)
        else:
            os.remove(file)

    # ------------------------ Convert xml files to json ------------------------ #

    logger.info('Converting xml files to json...')

    xml_dirs = [os.path.join(raw_data_dir, sub_dir, 'annotations') for sub_dir in sub_dirs]
    output_dirs = [os.path.join(raw_data_dir, sub_dir, 'annotations.json') for sub_dir in sub_dirs]

    generate_annotations(
        directories=xml_dirs,
        mappings=config['class_label_map'],
        output_files=output_dirs
    )

    logger.info('Clean-up by removing xml files in the original annotations folders...')
    for xml_dir in xml_dirs:
        shutil.rmtree(xml_dir)

    # ------------------------ Upload data to s3 ------------------------ #

    logger.info('Uploading data to s3...')

    for sub_dir in sub_dirs:
        subprocess.run(
            f'aws s3 cp {os.path.join(raw_data_dir, sub_dir)} s3://{config["s3_bucket"]}/{config["s3_key"]}/input-data/{sub_dir} --recursive',
            shell=True
        )

    logger.info('Clean-up by removing the raw data directory...')
    shutil.rmtree(raw_data_dir)

    return None

if __name__ == '__main__':

    from custom_utils import get_logger

    main()