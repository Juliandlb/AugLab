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
        self.aug_config = {
            'flip_mode': 'none',
            'rotation': 0,
            'brightness': 1.0,
            'contrast': 1.0
        }
        self.last_frame_idx = 0

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
            self.last_frame_idx = 0
            aug_img = apply_augmentation(first_frame, self.aug_config)
            return first_frame, self.total_frames, self.current_type, aug_img, self.current_filename
            
        except Exception as e:
            return None, 0, f"Error loading file: {str(e)}", None, ""

    def get_frame(self, frame_idx: int) -> np.ndarray:
        """
        Get frame at specified index.
        
        Args:
            frame_idx: Index of the frame to retrieve
            
        Returns:
            Original frame
        """
        if self.current_data is None:
            return None
            
        try:
            if self.current_type == 'image':
                frame = self.current_data
            elif self.current_type == 'video':
                frame = self.current_data[frame_idx]
            elif self.current_type == 'jsonl':
                frame = self.current_data['frames'][frame_idx]
            else:
                return None
            return frame
        except IndexError:
            return None

    def get_augmented(self, frame_idx: int, flip_mode: str, rotation: float, brightness: float, contrast: float) -> np.ndarray:
        frame = self.get_frame(frame_idx)
        if frame is None:
            return None
        config = {
            'flip_mode': flip_mode,
            'rotation': rotation,
            'brightness': brightness,
            'contrast': contrast
        }
        return apply_augmentation(frame, config)

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
        
        # Previews above augmentation controls
        with gr.Row():
            original_preview = gr.Image(label="Original", elem_id="original_preview")
            augmented_preview = gr.Image(label="Augmented", interactive=False, elem_id="augmented_preview")
        
        # Augmentation controls below previews
        with gr.Row():
            with gr.Group():
                flip_mode = gr.Dropdown(["none", "horizontal", "vertical"], value="none", label="Flip")
                rotation = gr.Slider(minimum=-45, maximum=45, value=0, step=1, label="Rotation (degrees)")
                brightness = gr.Slider(minimum=0.5, maximum=2.0, value=1.0, step=0.01, label="Brightness")
                contrast = gr.Slider(minimum=0.5, maximum=2.0, value=1.0, step=0.01, label="Contrast")
                apply_btn = gr.Button("Apply Augmentation", variant="primary")
        
        # Recommendations section
        with gr.Row():
            recommendations = gr.Textbox(label="Augmentation Recommendations", interactive=False)
        
        # Export section
        with gr.Row():
            export_btn = gr.Button("Export Augmented Data")
            save_config_btn = gr.Button("Save Augmentation Config")
        
        # Event handlers
        def on_file_upload(file_obj, flip_mode_val, rotation_val, brightness_val, contrast_val):
            interface.aug_config = {
                'flip_mode': flip_mode_val,
                'rotation': rotation_val,
                'brightness': brightness_val,
                'contrast': contrast_val
            }
            frame, total_frames, file_type_val, aug_img, filename = interface.load_file(file_obj)
            # Get recommendations for the first frame
            if frame is not None:
                stats = analyze_sample(frame)
                recommendations = get_recommendations(stats)
                recommendations_text = "\n".join(recommendations) if recommendations else "No recommendations at this time."
            else:
                recommendations_text = "No recommendations available."
            return frame, total_frames, file_type_val, aug_img, filename, 0, recommendations_text
        
        def on_frame_change(frame_idx, flip_mode_val, rotation_val, brightness_val, contrast_val):
            frame = interface.get_frame(frame_idx)
            aug_img = interface.get_augmented(frame_idx, flip_mode_val, rotation_val, brightness_val, contrast_val)
            # Get recommendations based on current frame
            stats = analyze_sample(frame)
            recommendations = get_recommendations(stats)
            return frame, aug_img, "\n".join(recommendations) if recommendations else "No recommendations at this time."
        
        def on_left(current_idx, flip_mode_val, rotation_val, brightness_val, contrast_val):
            idx = max(0, current_idx - 1)
            frame = interface.get_frame(idx)
            aug_img = interface.get_augmented(idx, flip_mode_val, rotation_val, brightness_val, contrast_val)
            # Get recommendations based on current frame
            stats = analyze_sample(frame)
            recommendations = get_recommendations(stats)
            return idx, frame, aug_img, "\n".join(recommendations) if recommendations else "No recommendations at this time."
        
        def on_right(current_idx, flip_mode_val, rotation_val, brightness_val, contrast_val):
            idx = min(interface.total_frames - 1, current_idx + 1)
            frame = interface.get_frame(idx)
            aug_img = interface.get_augmented(idx, flip_mode_val, rotation_val, brightness_val, contrast_val)
            # Get recommendations based on current frame
            stats = analyze_sample(frame)
            recommendations = get_recommendations(stats)
            return idx, frame, aug_img, "\n".join(recommendations) if recommendations else "No recommendations at this time."
        
        def on_apply(frame_idx, flip_mode_val, rotation_val, brightness_val, contrast_val):
            aug_img = interface.get_augmented(frame_idx, flip_mode_val, rotation_val, brightness_val, contrast_val)
            # Get recommendations based on current frame
            frame = interface.get_frame(frame_idx)
            stats = analyze_sample(frame)
            recommendations = get_recommendations(stats)
            return aug_img, "\n".join(recommendations) if recommendations else "No recommendations at this time."
        
        # Connect events
        input_file.upload(
            fn=on_file_upload,
            inputs=[input_file, flip_mode, rotation, brightness, contrast],
            outputs=[original_preview, frame_slider, file_type, augmented_preview, file_name, frame_slider, recommendations]
        )
        
        frame_slider.change(
            fn=on_frame_change,
            inputs=[frame_slider, flip_mode, rotation, brightness, contrast],
            outputs=[original_preview, augmented_preview, recommendations]
        )
        
        left_btn.click(
            fn=on_left,
            inputs=[frame_slider, flip_mode, rotation, brightness, contrast],
            outputs=[frame_slider, original_preview, augmented_preview, recommendations]
        )
        
        right_btn.click(
            fn=on_right,
            inputs=[frame_slider, flip_mode, rotation, brightness, contrast],
            outputs=[frame_slider, original_preview, augmented_preview, recommendations]
        )
        
        # Add real-time updates for augmentation controls
        flip_mode.change(
            fn=on_apply,
            inputs=[frame_slider, flip_mode, rotation, brightness, contrast],
            outputs=[augmented_preview, recommendations]
        )
        
        rotation.change(
            fn=on_apply,
            inputs=[frame_slider, flip_mode, rotation, brightness, contrast],
            outputs=[augmented_preview, recommendations]
        )
        
        brightness.change(
            fn=on_apply,
            inputs=[frame_slider, flip_mode, rotation, brightness, contrast],
            outputs=[augmented_preview, recommendations]
        )
        
        contrast.change(
            fn=on_apply,
            inputs=[frame_slider, flip_mode, rotation, brightness, contrast],
            outputs=[augmented_preview, recommendations]
        )
        
        # Keep the apply button for explicit updates if needed
        apply_btn.click(
            fn=on_apply,
            inputs=[frame_slider, flip_mode, rotation, brightness, contrast],
            outputs=[augmented_preview, recommendations]
        )
    
    return app 