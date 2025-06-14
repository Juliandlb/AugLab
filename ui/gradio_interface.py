import gradio as gr
import numpy as np
from core.loader import load_data
from core.augment import apply_augmentation
from core.stats import analyze_sample, get_recommendations
from typing import Tuple, Dict, Any

class AugLabInterface:
    def __init__(self):
        self.current_data = None
        self.current_type = None
        self.current_frame_idx = 0
        self.total_frames = 0

    def load_file(self, file_obj: gr.File) -> Tuple[np.ndarray, int, str]:
        """
        Handle file upload and return the first frame.
        
        Args:
            file_obj: Gradio File object
            
        Returns:
            Tuple of (first frame, total frames, file type)
        """
        if file_obj is None:
            return None, 0, "No file uploaded"
            
        try:
            # Load the file
            self.current_data, self.current_type = load_data(file_obj.name)
            
            # Get first frame based on file type
            if self.current_type == 'image':
                first_frame = self.current_data
                self.total_frames = 1
            elif self.current_type == 'video':
                first_frame = self.current_data[0]
                self.total_frames = len(self.current_data)
            elif self.current_type == 'jsonl':
                first_frame = self.current_data['frames'][0]
                self.total_frames = len(self.current_data['frames'])
            
            self.current_frame_idx = 0
            return first_frame, self.total_frames, self.current_type
            
        except Exception as e:
            return None, 0, f"Error loading file: {str(e)}"

    def get_frame(self, frame_idx: int) -> np.ndarray:
        """
        Get frame at specified index.
        
        Args:
            frame_idx: Index of the frame to retrieve
            
        Returns:
            Frame as numpy array
        """
        if self.current_data is None:
            return None
            
        try:
            if self.current_type == 'image':
                return self.current_data
            elif self.current_type == 'video':
                return self.current_data[frame_idx]
            elif self.current_type == 'jsonl':
                return self.current_data['frames'][frame_idx]
        except IndexError:
            return None

def create_interface():
    """
    Creates and returns the Gradio interface for AugLab.
    """
    interface = AugLabInterface()
    
    with gr.Blocks(title="AugLab - Interactive Augmentation Playground") as app:
        gr.Markdown("# AugLab - Interactive Augmentation Playground")
        
        with gr.Row():
            # Input section
            with gr.Column():
                input_file = gr.File(label="Upload Image/Video/JSONL")
                frame_slider = gr.Slider(
                    minimum=0,
                    maximum=100,
                    step=1,
                    label="Frame",
                    interactive=True
                )
                file_type = gr.Textbox(label="File Type", interactive=False)
            
            # Preview section
            with gr.Column():
                original_preview = gr.Image(label="Original")
                augmented_preview = gr.Image(label="Augmented")
        
        # Augmentation controls
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Augmentation Controls")
                flip_btn = gr.Button("Flip")
                rotation_slider = gr.Slider(minimum=-180, maximum=180, step=1, label="Rotation")
                brightness_slider = gr.Slider(minimum=0.0, maximum=2.0, step=0.1, label="Brightness")
                contrast_slider = gr.Slider(minimum=0.0, maximum=2.0, step=0.1, label="Contrast")
                blur_slider = gr.Slider(minimum=0, maximum=10, step=1, label="Blur")
        
        # Recommendations section
        with gr.Row():
            recommendations = gr.Textbox(label="Augmentation Recommendations", interactive=False)
        
        # Export section
        with gr.Row():
            export_btn = gr.Button("Export Augmented Data")
            save_config_btn = gr.Button("Save Augmentation Config")
        
        # Event handlers
        def on_file_upload(file_obj):
            frame, total_frames, file_type = interface.load_file(file_obj)
            return frame, total_frames, file_type
        
        def on_frame_change(frame_idx):
            frame = interface.get_frame(frame_idx)
            return frame
        
        # Connect events
        input_file.upload(
            fn=on_file_upload,
            inputs=[input_file],
            outputs=[original_preview, frame_slider, file_type]
        )
        
        frame_slider.change(
            fn=on_frame_change,
            inputs=[frame_slider],
            outputs=[original_preview]
        )
    
    return app 