import numpy as np
import cv2
from typing import Dict, Any

def apply_augmentation(image: np.ndarray, config: Dict[str, Any]) -> np.ndarray:
    """
    Apply a sequence of augmentations to the image based on config.
    Args:
        image: Input image (numpy array)
        config: Dict with keys: flip_mode, rotation, brightness, contrast
    Returns:
        Augmented image (numpy array)
    """
    aug_img = image.copy()
    # Flip
    flip_mode = config.get('flip_mode', 'none')
    if flip_mode == 'horizontal':
        aug_img = cv2.flip(aug_img, 1)
    elif flip_mode == 'vertical':
        aug_img = cv2.flip(aug_img, 0)
    # Rotation
    angle = config.get('rotation', 0)
    if angle != 0:
        h, w = aug_img.shape[:2]
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
        aug_img = cv2.warpAffine(aug_img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    # Brightness
    brightness = config.get('brightness', 1.0)
    if brightness != 1.0:
        aug_img = cv2.convertScaleAbs(aug_img, alpha=brightness, beta=0)
    # Contrast
    contrast = config.get('contrast', 1.0)
    if contrast != 1.0:
        mean = np.mean(aug_img)
        aug_img = cv2.addWeighted(aug_img, contrast, np.full_like(aug_img, mean, dtype=aug_img.dtype), 1 - contrast, 0)
    return aug_img

def flip_image(image: np.ndarray) -> np.ndarray:
    """
    Flip the image horizontally.
    """
    return np.fliplr(image) if image is not None else None

def rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
    """
    Rotate the image by specified angle.
    """
    if image is None:
        return None
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
    return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

def adjust_brightness(image: np.ndarray, factor: float) -> np.ndarray:
    """
    Adjust image brightness.
    """
    if image is None:
        return None
    return cv2.convertScaleAbs(image, alpha=factor, beta=0)

def adjust_contrast(image: np.ndarray, factor: float) -> np.ndarray:
    """
    Adjust image contrast.
    """
    if image is None:
        return None
    mean = np.mean(image)
    return cv2.addWeighted(image, factor, np.full_like(image, mean, dtype=image.dtype), 1 - factor, 0)

def apply_blur(image: np.ndarray, kernel_size: int) -> np.ndarray:
    """
    Apply Gaussian blur to the image.
    """
    if image is None or kernel_size <= 0:
        return image
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0) 