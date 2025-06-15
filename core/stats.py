import numpy as np
import cv2
from typing import Dict, List, Union

def extract_stats(image: np.ndarray) -> Dict[str, float]:
    """
    Extract various statistics from an image.
    
    Args:
        image: Input image (can be grayscale or RGB)
        
    Returns:
        Dictionary containing:
        - brightness: average pixel value
        - contrast: standard deviation of pixel values
        - saturation: average saturation (for color images)
        - sharpness: Laplacian variance
        - entropy: Shannon entropy
    """
    stats = {}
    
    # Convert to grayscale if RGB
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        # Calculate saturation for color images
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        stats['saturation'] = np.mean(hsv[:, :, 1])
    else:
        gray = image
        stats['saturation'] = 0.0  # No saturation for grayscale images
    
    # Calculate basic statistics
    stats['brightness'] = calculate_brightness(gray)
    stats['contrast'] = calculate_contrast(gray)
    stats['sharpness'] = calculate_sharpness(gray)
    stats['entropy'] = calculate_entropy(gray)
    
    return stats

def calculate_brightness(image: np.ndarray) -> float:
    """
    Calculate image brightness as the mean pixel value.
    
    Args:
        image: Grayscale image
        
    Returns:
        Average brightness value
    """
    return np.mean(image)

def calculate_contrast(image: np.ndarray) -> float:
    """
    Calculate image contrast as the standard deviation of pixel values.
    
    Args:
        image: Grayscale image
        
    Returns:
        Contrast value
    """
    return np.std(image)

def calculate_sharpness(image: np.ndarray) -> float:
    """
    Calculate image sharpness using Laplacian variance.
    
    Args:
        image: Grayscale image
        
    Returns:
        Sharpness value
    """
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    return np.var(laplacian)

def calculate_entropy(image: np.ndarray) -> float:
    """
    Calculate Shannon entropy of the image.
    
    Args:
        image: Grayscale image
        
    Returns:
        Entropy value
    """
    # Calculate histogram
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    hist = hist.ravel() / hist.sum()
    
    # Calculate entropy
    entropy = -np.sum(hist * np.log2(hist + np.finfo(float).eps))
    return entropy

def analyze_sample(image: np.ndarray) -> Dict[str, float]:
    """
    Analyze image characteristics and return statistics.
    
    Args:
        image: Input image
        
    Returns:
        Dictionary containing image statistics
    """
    return extract_stats(image)

def get_recommendations(stats: Dict[str, float]) -> List[str]:
    """
    Generate augmentation recommendations based on image statistics.
    Only suggest augmentations that are implemented.
    Args:
        stats: Dictionary containing image statistics
    Returns:
        List of recommended augmentations
    """
    recommendations = []

    # Brightness recommendations
    if stats['brightness'] < 100:
        recommendations.append("Increase brightness using the Brightness slider.")
    elif stats['brightness'] > 200:
        recommendations.append("Decrease brightness using the Brightness slider.")

    # Contrast recommendations
    if stats['contrast'] < 30:
        recommendations.append("Increase contrast using the Contrast slider.")
    elif stats['contrast'] > 100:
        recommendations.append("Decrease contrast using the Contrast slider.")

    # Sharpness recommendations
    if stats['sharpness'] < 100:
        recommendations.append("Increase sharpness by increasing Contrast.")
    elif stats['sharpness'] > 300:
        recommendations.append("Apply Blur to reduce excessive sharpness.")

    # Saturation recommendations (for color images)
    if stats['saturation'] > 0:  # Color image
        if stats['saturation'] < 50:
            recommendations.append("Increase color vibrancy using the Saturation or Hue Shift sliders.")
        elif stats['saturation'] > 200:
            recommendations.append("Reduce color intensity using the Saturation slider.")

    # Occlusion recommendation (if image is too uniform/clean)
    if stats['entropy'] < 5.0:
        recommendations.append("Add occlusion (random patch) to increase image diversity.")

    return recommendations 