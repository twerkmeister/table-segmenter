from typing import List, Tuple
import random
import cv2
import numpy as np
import table_segmenter.conf as conf


def augment_multi(preprocessed_images: np.ndarray, targets: List[List[int]],
                  original_image_shapes: List[Tuple[int]]) -> Tuple[np.ndarray,
                                                                    List[List[int]]]:
    """Manages augmenting multiple images."""
    augmented_images = []
    augmented_targets = []
    for preprocessed_image, task_targets, original_image_shape in zip(
        preprocessed_images, targets, original_image_shapes
    ):
        augmented_image, augmented_task_targets = \
            augment_single(preprocessed_image, task_targets, original_image_shape)
        augmented_images.append(augmented_image)
        augmented_targets.append(augmented_task_targets)

    return np.stack(augmented_images), augmented_targets


def augment_single(preprocessed_image: np.ndarray, targets: List[int],
                   original_image_shape: Tuple[int]) -> Tuple[np.ndarray, List[int]]:
    """Augments a single image by shifting it around in the padded area."""
    padding_height = preprocessed_image.shape[0] - min(original_image_shape[0],
                                                       conf.image_max_height)
    padding_width = preprocessed_image.shape[1] - min(original_image_shape[1],
                                                      conf.image_max_width)

    shift_height = random.randint(0, padding_height)
    shift_width = random.randint(0, padding_width)

    cut_image = preprocessed_image[0: preprocessed_image.shape[0] - shift_height,
                                   0: preprocessed_image.shape[1] - shift_width]

    augmented_image = np.pad(cut_image, [(shift_height, 0), (shift_width, 0), (0, 0)],
                             constant_values=0)
    augmented_targets = targets[:]
    augmented_targets[0] += shift_height
    return augmented_image, augmented_targets


def preprocess_targets(targets: List[List[int]]):
    return np.vstack([np.asarray([target[0] / conf.image_downscale_factor, target[1]])
                      for target in targets])


def preprocess_images(images: List[np.ndarray]) -> np.ndarray:
    return np.stack([preprocess_image(img) for img in images])


def preprocess_image(img: np.ndarray) -> np.ndarray:
    """Runs the preprocessing pipeline."""
    img = make_grayscale(img)
    img = truncate(img, conf.image_max_height, conf.image_max_width)
    img = blur(img)
    # normalize so that max value is 0.0 post inversion
    img = np.max(img) - img
    img = pad(img, conf.image_max_height, conf.image_max_width)
    return np.expand_dims(img, 2)


def make_grayscale(img: np.ndarray) -> np.ndarray:
    """Turns BGR image into grayscale."""
    if len(img.shape) == 3 and img.shape[2] == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        return img


def truncate(img: np.ndarray, max_height: int, max_width: int) -> np.ndarray:
    """Truncates images too the max size of the network input."""
    return img[:max_height, :max_width]


def invert(img: np.ndarray) -> np.ndarray:
    """Inverts an image to make black font have value instead of no-value."""
    return cv2.bitwise_not(img)


def normalize(img: np.ndarray) -> np.ndarray:
    """Normalizes the image to have values between 0 and 1."""
    return cv2.normalize(img, None, alpha=-1, beta=1,
                         norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)


def pad(img: np.ndarray, target_height: int, target_width: int) -> np.ndarray:
    """Pads the image to have the target size for the network input."""
    return np.pad(img, [(0, max(0, target_height-img.shape[0])),
                        (0, max(0, target_width-img.shape[1]))],
                  constant_values=0)


def blur(img: np.ndarray) -> np.ndarray:
    """Blurs the image."""
    return cv2.blur(img, (5, 5))
