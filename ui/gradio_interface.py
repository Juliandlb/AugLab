import gradio as gr
import numpy as np
from core.loader import load_data
from core.augment import apply_augmentation, flip_image
from core.stats import analyze_sample, get_recommendations
from typing import Tuple, Dict, Any

class AugLabInterface:
    def __init__(self):
        self.current_data = None
        self.current_type = None
        self.current_frame_idx = 0
        self.total_frames = 0
        self.current_filename = ""

    def load_file(self, file_obj: gr.File) -> Tuple[np.ndarray, int, str, np.ndarray, str]:
        """
        Handle file upload and return the first frame.
        
        Args:
            file_obj: Gradio File object
            
        Returns:
            Tuple of (first frame, total frames, file type, augmented frame, file name)
        """
        if file_obj is None:
            return None, 0, "No file uploaded", None, ""
            
        try:
            # Load the file
            self.current_data, self.current_type = load_data(file_obj.name)
            self.current_filename = file_obj.name.split("/")[-1]
            
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
            # Augmented is just a flip for now
            augmented = flip_image(first_frame)
            return first_frame, self.total_frames, self.current_type, augmented, self.current_filename
            
        except Exception as e:
            return None, 0, f"Error loading file: {str(e)}", None, ""

    def get_frame(self, frame_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get frame at specified index.
        
        Args:
            frame_idx: Index of the frame to retrieve
            
        Returns:
            Tuple of (original frame, augmented frame)
        """
        if self.current_data is None:
            return None, None
            
        try:
            if self.current_type == 'image':
                frame = self.current_data
            elif self.current_type == 'video':
                frame = self.current_data[frame_idx]
            elif self.current_type == 'jsonl':
                frame = self.current_data['frames'][frame_idx]
            else:
                return None, None
            # Augmented is just a flip for now
            augmented = flip_image(frame)
            return frame, augmented
        except IndexError:
            return None, None

def create_interface():
    """
    Creates and returns the Gradio interface for AugLab.
    """
    interface = AugLabInterface()
    
    with gr.Blocks(title="AugLab - Interactive Augmentation Playground", css="""
    .arrow-btn {
        min-width: 36px !important;
        max-width: 36px !important;
        min-height: 36px !important;
        max-height: 36px !important;
        padding: 0 !important;
        font-size: 1.5rem !important;
        border-radius: 6px !important;
    }
    .arrow-btn-right {
        margin-right: 12px !important;
    }
    .slider-row {
        display: flex;
        align-items: center;
        gap: 8px;
    }
    """) as app:
        gr.Markdown("# AugLab - Interactive Augmentation Playground")
        
        # Top: Upload, file name, file type grouped in a single box
        with gr.Row():
            with gr.Column():
                with gr.Group():
                    input_file = gr.File(label="Upload image/video/JSONL")
                    file_name = gr.Textbox(label="File Name", interactive=False)
                    file_type = gr.Textbox(label="File Type", interactive=False)
        
        # Middle: Frame navigation (slider and buttons in the same row and group)
        with gr.Row():
            with gr.Group():
                with gr.Row(elem_classes=["slider-row"]):
                    frame_slider = gr.Slider(
                        minimum=0,
                        maximum=100,
                        step=1,
                        label="Frame",
                        interactive=True
                    )
                    left_btn = gr.Button("←", size="sm", variant="secondary", elem_classes=["arrow-btn"])
                    right_btn = gr.Button("→", size="sm", variant="secondary", elem_classes=["arrow-btn", "arrow-btn-right"])
        
        # Bottom: Previews
        with gr.Row():
            original_preview = gr.Image(label="Original", elem_id="original_preview")
            augmented_preview = gr.Image(label="Augmented", interactive=False, elem_id="augmented_preview")
        
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
            frame, total_frames, file_type_val, augmented, filename = interface.load_file(file_obj)
            # Always show first frame and set slider to 0
            return frame, total_frames, file_type_val, augmented, filename, 0
        
        def on_frame_change(frame_idx):
            frame, augmented = interface.get_frame(frame_idx)
            return frame, augmented
        
        def on_left(current_idx):
            idx = max(0, current_idx - 1)
            frame, augmented = interface.get_frame(idx)
            return idx, frame, augmented
        
        def on_right(current_idx):
            idx = min(interface.total_frames - 1, current_idx + 1)
            frame, augmented = interface.get_frame(idx)
            return idx, frame, augmented
        
        # Connect events
        input_file.upload(
            fn=on_file_upload,
            inputs=[input_file],
            outputs=[original_preview, frame_slider, file_type, augmented_preview, file_name, frame_slider]
        )
        
        frame_slider.change(
            fn=on_frame_change,
            inputs=[frame_slider],
            outputs=[original_preview, augmented_preview]
        )
        
        left_btn.click(
            fn=on_left,
            inputs=[frame_slider],
            outputs=[frame_slider, original_preview, augmented_preview]
        )
        
        right_btn.click(
            fn=on_right,
            inputs=[frame_slider],
            outputs=[frame_slider, original_preview, augmented_preview]
        )
    
    return app 