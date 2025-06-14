import json
from typing import Union, Dict, List, Tuple
import cv2
import numpy as np
import base64
from pathlib import Path
import imageio

def load_data(file_path: str) -> Tuple[Union[np.ndarray, List[np.ndarray], Dict], str]:
    """
    Load data from various file formats (image, video, or JSONL).
    
    Args:
        file_path: Path to the input file
        
    Returns:
        Tuple of (loaded data, file type)
        - For images: (image array, 'image')
        - For videos: (list of frames, 'video')
        - For JSONL: (dictionary of frames, 'jsonl')
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # Determine file type
    if file_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
        return load_image(str(file_path)), 'image'
    elif file_path.suffix.lower() in ['.mp4', '.avi', '.mov']:
        return load_video(str(file_path)), 'video'
    elif file_path.suffix.lower() == '.jsonl':
        return load_jsonl(str(file_path)), 'jsonl'
    else:
        raise ValueError(f"Unsupported file type: {file_path.suffix}")

def load_image(file_path: str) -> np.ndarray:
    """
    Load a single image file.
    
    Args:
        file_path: Path to the image file
        
    Returns:
        Image as numpy array
    """
    try:
        image = cv2.imread(file_path)
        if image is None:
            raise ValueError(f"Failed to load image: {file_path}")
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    except Exception as e:
        raise RuntimeError(f"Error loading image {file_path}: {str(e)}")

def load_video(file_path: str) -> List[np.ndarray]:
    """
    Load a video file and return frames.
    
    Args:
        file_path: Path to the video file
        
    Returns:
        List of frames as numpy arrays
    """
    try:
        frames = []
        cap = cv2.VideoCapture(file_path)
        
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {file_path}")
            
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
            
        cap.release()
        
        if not frames:
            raise ValueError(f"No frames found in video: {file_path}")
            
        return frames
    except Exception as e:
        raise RuntimeError(f"Error loading video {file_path}: {str(e)}")

def load_jsonl(file_path: str) -> Dict:
    """
    Load a JSONL file containing episode frames.
    Each line should be a JSON object with either:
    - 'frame_path': path to image file
    - 'frame_base64': base64 encoded image
    
    Args:
        file_path: Path to the JSONL file
        
    Returns:
        Dictionary containing frames and metadata
    """
    try:
        frames = []
        metadata = []
        
        with open(file_path, 'r') as f:
            for line in f:
                data = json.loads(line.strip())
                metadata.append(data)
                
                # Handle frame data
                if 'frame_path' in data:
                    frame = load_image(data['frame_path'])
                elif 'frame_base64' in data:
                    # Decode base64 image
                    frame_data = base64.b64decode(data['frame_base64'])
                    frame = imageio.imread(frame_data)
                else:
                    raise ValueError("JSONL line must contain either 'frame_path' or 'frame_base64'")
                
                frames.append(frame)
        
        if not frames:
            raise ValueError(f"No frames found in JSONL: {file_path}")
            
        return {
            'frames': frames,
            'metadata': metadata
        }
    except Exception as e:
        raise RuntimeError(f"Error loading JSONL {file_path}: {str(e)}") 