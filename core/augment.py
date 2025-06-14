import numpy as np
import cv2
from typing import Dict, Any, Tuple
import random

def apply_augmentation(image: np.ndarray, config: Dict[str, Any]) -> np.ndarray:
    """
    Apply a sequence of augmentations to the image based on config.
    Args:
        image: Input image (numpy array)
        config: Dict with augmentation parameters
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
    
    # Blur
    blur_kernel = config.get('blur_kernel', 0)
    if blur_kernel > 0:
        # Ensure kernel size is odd
        kernel_size = 2 * int(blur_kernel) + 1
        aug_img = cv2.GaussianBlur(aug_img, (kernel_size, kernel_size), 0)
    
    # Color jitter
    if len(aug_img.shape) == 3:  # Only apply to color images
        # Convert to HSV for color adjustments
        hsv = cv2.cvtColor(aug_img, cv2.COLOR_RGB2HSV).astype(np.float32)
        
        # Hue adjustment
        hue_shift = config.get('hue_shift', 0)
        if hue_shift != 0:
            hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180
        
        # Saturation adjustment
        saturation = config.get('saturation', 1.0)
        if saturation != 1.0:
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation, 0, 255)
        
        # Convert back to RGB
        aug_img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    
    # Occlusion
    occlusion_size = config.get('occlusion_size', 0)
    if occlusion_size > 0:
        aug_img = apply_occlusion(aug_img, occlusion_size)
    
    return aug_img

def apply_occlusion(image: np.ndarray, size_ratio: float) -> np.ndarray:
    """
    Apply random occlusion to the image.
    
    Args:
        image: Input image
        size_ratio: Ratio of occlusion size to image size (0-1)
        
    Returns:
        Image with random occlusion
    """
    if size_ratio <= 0:
        return image
        
    h, w = image.shape[:2]
    # Calculate occlusion size based on ratio
    occl_h = int(h * size_ratio)
    occl_w = int(w * size_ratio)
    
    # Ensure minimum size
    occl_h = max(10, occl_h)
    occl_w = max(10, occl_w)
    
    # Random position
    x = random.randint(0, w - occl_w)
    y = random.randint(0, h - occl_h)
    
    # Create occlusion (black rectangle)
    image[y:y+occl_h, x:x+occl_w] = 0
    
    return image

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
    # Ensure kernel size is odd
    kernel_size = 2 * int(kernel_size) + 1
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0) 