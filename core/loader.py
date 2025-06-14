import json
from typing import Union, Dict, List, Tuple
import cv2
import numpy as np
import base64
from pathlib import Path
import imageio
import subprocess
import tempfile
import os

def get_video_codec(file_path: str) -> str:
    """
    Returns the codec name of the first video stream in the file using ffprobe.
    """
    try:
        result = subprocess.run([
            'ffprobe', '-v', 'error', '-select_streams', 'v:0',
            '-show_entries', 'stream=codec_name',
            '-of', 'default=nw=1', file_path
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        for line in result.stdout.splitlines():
            if line.startswith('codec_name='):
                return line.split('=')[1].strip()
        return None
    except Exception:
        return None

def convert_video_to_h264(input_path: str) -> str:
    """
    Converts a video to H.264 codec using ffmpeg and returns the path to the converted file.
    """
    temp_dir = tempfile.gettempdir()
    output_path = os.path.join(temp_dir, f"converted_{os.path.basename(input_path)}")
    command = [
        'ffmpeg', '-y', '-i', input_path,
        '-c:v', 'libx264', '-crf', '23', '-preset', 'fast',
        '-c:a', 'copy', output_path
    ]
    subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return output_path

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
        # Check codec and convert if needed
        codec = get_video_codec(str(file_path))
        if codec and codec != 'h264':
            converted_path = convert_video_to_h264(str(file_path))
            frames = load_video(converted_path)
            # Optionally, clean up the temp file after loading
            os.remove(converted_path)
            return frames, 'video'
        else:
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