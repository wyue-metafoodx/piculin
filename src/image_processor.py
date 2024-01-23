import os
from PIL import Image
import logging
import keras
from keras.applications import EfficientNetB0
from keras.applications.efficientnet import preprocess_input

import tensorflow as tf
print(tf.__version__)

try:
    model = EfficientNetB0(weights='imagenet')
    
    # Configure logging
    log_folder = './log'
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)

    # Create a logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Create a file handler for writing logs to a file
    file_handler = logging.FileHandler(os.path.join(log_folder, 'image_processing.log'))
    file_handler.setLevel(logging.INFO)
    file_format = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')
    file_handler.setFormatter(file_format)

    # Create a stream handler for printing logs to the console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter('%(levelname)s - %(message)s')
    console_handler.setFormatter(console_format)

    # Add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    def crop_and_resize_images(
            source_folder, target_folder, target_size=(224, 224)):
        for dir, _, files in os.walk(source_folder):
            for filename in files:
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(dir, filename)
                    img = Image.open(img_path)

                    # Cropping to the center of the image, the origin point is left
                    # top corner
                    width, height = img.size
                    target_width, target_height = target_size
                    left = (width - target_width) / 2
                    top = (height - target_height) / 2
                    right = (width + target_width) / 2
                    bottom = (height + target_height) / 2

                    # If the image is smaller than the target size, resize without
                    # cropping
                    if width < target_width or height < target_height:
                        img = img.resize(target_size, Image.Resampling.LANCZOS)
                        logger.info(f'Resized smaller image: {filename}')
                    else:
                        img = img.crop((left, top, right, bottom))
                        img = img.resize(target_size, Image.Resampling.LANCZOS)
                        logger.info(f'Cropped and resized larger image: {filename}')

                    # Construct a new path to save the processed image
                    relative_path = os.path.relpath(dir, source_folder)
                    save_dir = os.path.join(target_folder, relative_path)
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)

                    save_path = os.path.join(save_dir, filename)
                    img.save(save_path)


    source_folder = './input_images'  # Replace with your source folder path
    target_folder = './processed_images'  # Replace with your target folder path

    crop_and_resize_images(source_folder, target_folder)

except Exception as e:
    print(f"Error: {e}")
