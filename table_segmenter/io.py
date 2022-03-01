from typing import Text, Tuple, List
import os
import cv2
import numpy as np


def list_images(path: Text) -> List[Text]:
    """Lists all jpg files in a folder."""
    files = os.listdir(path)
    allowed_extensions = {".jpeg", ".jpg"}
    return [f for f in files if os.path.splitext(f)[1] in allowed_extensions]


def load_images(data_path: Text) -> Tuple[List[Text], List[np.ndarray]]:
    """Loads inputs with their names from data path."""
    image_names = list_images(data_path)
    images = [read_image(os.path.join(data_path, image_name))
              for image_name in image_names]
    return image_names, images


def read_image(image_path: Text) -> np.ndarray:
    """Reads an image from disc."""
    return cv2.imread(image_path)


def write_image(file_path: Text, image: np.ndarray) -> None:
    """Writes image to disc."""
    cv2.imwrite(file_path, image)


def read_targets_for_image(image_path: Text) -> List[int]:
    """Reads the pixel target for the given task image."""
    target_path = os.path.splitext(image_path)[0] + ".txt"
    with open(target_path, mode="r", encoding="utf-8") as f:
        return [int(i) for i in f.read().split(",")]


def load_targets(data_path: Text, image_names: List[Text]) -> List[List[int]]:
    """Loads all targets for the provided image_names."""
    return [read_targets_for_image(os.path.join(data_path, image_name))
            for image_name in image_names]
