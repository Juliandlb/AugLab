import gradio as gr
import numpy as np
import json
import os
from datetime import datetime
from core.loader import load_data
from core.augment import apply_augmentation, flip_image
from core.stats import analyze_sample, get_recommendations
from typing import Tuple, Dict, Any, Optional
import cv2
import imageio
import random

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
            'contrast': 1.0,
            'blur_kernel': 0,
            'hue_shift': 0,
            'saturation': 1.0,
            'occlusion_size': 0
        }
        self.last_frame_idx = 0
        self.export_dir = "exports"
        os.makedirs(self.export_dir, exist_ok=True)

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

    def get_augmented(self, frame_idx: int, flip_mode: str, rotation: float, brightness: float, contrast: float, blur_kernel: float, hue_shift: float, saturation: float, occlusion_size: float) -> np.ndarray:
        frame = self.get_frame(frame_idx)
        if frame is None:
            return None
        config = {
            'flip_mode': flip_mode,
            'rotation': rotation,
            'brightness': brightness,
            'contrast': contrast,
            'blur_kernel': blur_kernel,
            'hue_shift': hue_shift,
            'saturation': saturation,
            'occlusion_size': occlusion_size
        }
        return apply_augmentation(frame, config)

    def export_augmented_data(self, frame_idx: int, flip_mode: str, rotation: float, brightness: float, contrast: float, blur_kernel: float, hue_shift: float, saturation: float, occlusion_size: float) -> Tuple[str, str]:
        """
        Export the current augmented frame or video using the provided augmentation parameters.
        """
        if self.current_data is None:
            return None, "No data loaded"
        
        aug_config = {
            'flip_mode': flip_mode,
            'rotation': rotation,
            'brightness': brightness,
            'contrast': contrast,
            'blur_kernel': blur_kernel,
            'hue_shift': hue_shift,
            'saturation': saturation,
            'occlusion_size': occlusion_size
        }
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = os.path.splitext(self.current_filename)[0]
        try:
            if self.current_type == 'image':
                aug_img = self.get_augmented(frame_idx, **aug_config)
                if aug_img is None:
                    return None, "Failed to generate augmented image"
                output_path = os.path.join(self.export_dir, f"{base_name}_aug_{timestamp}.png")
                if aug_img.dtype != np.uint8:
                    aug_img = np.clip(aug_img, 0, 255).astype(np.uint8)
                cv2.imwrite(output_path, cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR))
                return output_path, f"Exported augmented image to {output_path}"
            elif self.current_type == 'video':
                output_path = os.path.join(self.export_dir, f"{base_name}_aug_{timestamp}.mp4")
                try:
                    with imageio.get_writer(output_path, fps=30, quality=8) as writer:
                        for i in range(len(self.current_data)):
                            aug_frame = self.get_augmented(i, **aug_config)
                            if aug_frame is not None:
                                if aug_frame.dtype != np.uint8:
                                    aug_frame = np.clip(aug_frame, 0, 255).astype(np.uint8)
                                if aug_frame.shape[2] == 3:
                                    writer.append_data(aug_frame)
                                else:
                                    print(f"[DEBUG] Skipping frame {i}: not 3 channels")
                            else:
                                print(f"[DEBUG] Skipping frame {i}: aug_frame is None")
                    print(f"[DEBUG] Video export complete: {output_path}")
                except Exception as e:
                    return None, f"imageio export failed: {str(e)}"
                return output_path, f"Exported augmented video to {output_path}"
            elif self.current_type == 'jsonl':
                output_path = os.path.join(self.export_dir, f"{base_name}_aug_{timestamp}.jsonl")
                aug_data = self.current_data.copy()
                aug_data['augmentation_config'] = aug_config
                if 'frames' in aug_data:
                    for i, frame in enumerate(aug_data['frames']):
                        aug_frame = self.get_augmented(i, **aug_config)
                        if aug_frame is not None:
                            frame['augmented'] = True
                            frame['augmentation_params'] = aug_config
                with open(output_path, 'w') as f:
                    json.dump(aug_data, f, indent=2)
                return output_path, f"Exported augmented JSONL to {output_path}"
        except Exception as e:
            return None, f"Export failed: {str(e)}"

    def export_config(self, flip_mode: str, rotation: float, brightness: float, contrast: float, blur_kernel: float, hue_shift: float, saturation: float, occlusion_size: float) -> Tuple[str, str]:
        """
        Export the current augmentation configuration using the provided parameters.
        """
        if self.current_data is None:
            return None, "No data loaded"
        aug_config = {
            'flip_mode': flip_mode,
            'rotation': rotation,
            'brightness': brightness,
            'contrast': contrast,
            'blur_kernel': blur_kernel,
            'hue_shift': hue_shift,
            'saturation': saturation,
            'occlusion_size': occlusion_size
        }
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = os.path.splitext(self.current_filename)[0]
        try:
            config_data = {
                'filename': self.current_filename,
                'file_type': self.current_type,
                'augmentation_config': aug_config,
                'timestamp': timestamp
            }
            output_path = os.path.join(self.export_dir, f"{base_name}_config_{timestamp}.json")
            with open(output_path, 'w') as f:
                json.dump(config_data, f, indent=2)
            return output_path, f"Exported configuration to {output_path}"
        except Exception as e:
            return None, f"Export failed: {str(e)}"

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
                    demo_btn = gr.Button("Load Demo Sample", variant="secondary")
                    file_name = gr.Textbox(label="File Name", interactive=False)
                    file_type = gr.Textbox(label="File Type", interactive=False)
        
        # Middle: Frame navigation (slider and buttons in the same row and group)
        with gr.Row():
            with gr.Group():
                with gr.Row(elem_classes=["slider-row"]):
                    frame_slider = gr.Slider(
                        minimum=0,
                        maximum=0,
                        step=1,
                        value=0,
                        label="Frame",
                        interactive=True
                    )
                    left_btn = gr.Button("←", size="sm", variant="secondary", elem_classes=["arrow-btn"])
                    right_btn = gr.Button("→", size="sm", variant="secondary", elem_classes=["arrow-btn", "arrow-btn-right"])
        
        # Previews above augmentation controls
        with gr.Row():
            original_preview = gr.Image(label="Original", elem_id="original_preview")
            augmented_preview = gr.Image(label="Augmented", interactive=False, elem_id="augmented_preview")
        
        # Recommendations section (move above controls)
        with gr.Row():
            with gr.Column():
                recommendations = gr.Textbox(label="Augmentation Recommendations", interactive=False)
                agent_btn = gr.Button("Let the Agent Adjust", elem_id="agent_btn", variant="primary")
        
        # Augmentation controls below previews
        with gr.Row():
            with gr.Group():
                flip_mode = gr.Dropdown(["none", "horizontal", "vertical"], value="none", label="Flip")
                rotation = gr.Slider(minimum=-45, maximum=45, value=0, step=1, label="Rotation (degrees)")
                brightness = gr.Slider(minimum=0.5, maximum=2.0, value=1.0, step=0.01, label="Brightness")
                contrast = gr.Slider(minimum=0.5, maximum=2.0, value=1.0, step=0.01, label="Contrast")
                blur_kernel = gr.Slider(minimum=0, maximum=10, value=0, step=1, label="Blur (kernel size)")
                hue_shift = gr.Slider(minimum=-90, maximum=90, value=0, step=1, label="Hue Shift (degrees)")
                saturation = gr.Slider(minimum=0.0, maximum=2.0, value=1.0, step=0.01, label="Saturation")
                occlusion_size = gr.Slider(minimum=0, maximum=0.5, value=0, step=0.01, label="Occlusion Size (ratio)")
        
        # Export section
        with gr.Row():
            with gr.Column():
                export_btn = gr.Button("Export Augmented Data", variant="primary")
                export_output = gr.File(label="Download Exported Data")
                export_message = gr.Textbox(label="Export Status", interactive=False)
            
            with gr.Column():
                save_config_btn = gr.Button("Save Augmentation Config", variant="secondary")
                config_output = gr.File(label="Download Configuration")
                config_message = gr.Textbox(label="Config Status", interactive=False)
        
        # Event handlers
        def on_file_upload(file_obj, flip_mode_val, rotation_val, brightness_val, contrast_val, blur_kernel_val, hue_shift_val, saturation_val, occlusion_size_val):
            interface.aug_config = {
                'flip_mode': flip_mode_val,
                'rotation': rotation_val,
                'brightness': brightness_val,
                'contrast': contrast_val,
                'blur_kernel': blur_kernel_val,
                'hue_shift': hue_shift_val,
                'saturation': saturation_val,
                'occlusion_size': occlusion_size_val
            }
            frame, total_frames, file_type_val, aug_img, filename = interface.load_file(file_obj)
            # Compute stats for all frames and average them
            stats_list = []
            if interface.current_data is not None:
                if interface.current_type == 'image':
                    stats_list.append(analyze_sample(interface.current_data))
                elif interface.current_type == 'video':
                    for f in interface.current_data:
                        stats_list.append(analyze_sample(f))
                elif interface.current_type == 'jsonl':
                    for f in interface.current_data['frames']:
                        stats_list.append(analyze_sample(f))
            if stats_list:
                avg_stats = {k: float(np.mean([s[k] for s in stats_list])) for k in stats_list[0]}
                recommendations = get_recommendations(avg_stats)
                recommendations_text = "\n".join(recommendations) if recommendations else "No recommendations at this time."
            else:
                recommendations_text = "No recommendations available."
            # Set slider max to total_frames-1 (since index is zero-based)
            return frame, gr.update(maximum=max(0, total_frames-1), value=0, minimum=0), file_type_val, aug_img, filename, 0, recommendations_text
        
        def on_demo_click(flip_mode_val, rotation_val, brightness_val, contrast_val, blur_kernel_val, hue_shift_val, saturation_val, occlusion_size_val):
            import types
            demo_path = os.path.join("examples", "episode_000000.mp4")
            # Create a fake Gradio File object
            class DummyFile:
                def __init__(self, name):
                    self.name = name
            demo_file = DummyFile(demo_path)
            return on_file_upload(demo_file, flip_mode_val, rotation_val, brightness_val, contrast_val, blur_kernel_val, hue_shift_val, saturation_val, occlusion_size_val)
        
        def on_frame_change(frame_idx, flip_mode_val, rotation_val, brightness_val, contrast_val, blur_kernel_val, hue_shift_val, saturation_val, occlusion_size_val):
            frame = interface.get_frame(frame_idx)
            aug_img = interface.get_augmented(frame_idx, flip_mode_val, rotation_val, brightness_val, contrast_val, blur_kernel_val, hue_shift_val, saturation_val, occlusion_size_val)
            # Get recommendations based on current frame
            stats = analyze_sample(frame)
            recommendations = get_recommendations(stats)
            return frame, aug_img, "\n".join(recommendations) if recommendations else "No recommendations at this time."
        
        def on_left(current_idx, flip_mode_val, rotation_val, brightness_val, contrast_val, blur_kernel_val, hue_shift_val, saturation_val, occlusion_size_val):
            idx = max(0, current_idx - 1)
            frame = interface.get_frame(idx)
            aug_img = interface.get_augmented(idx, flip_mode_val, rotation_val, brightness_val, contrast_val, blur_kernel_val, hue_shift_val, saturation_val, occlusion_size_val)
            # Get recommendations based on current frame
            stats = analyze_sample(frame)
            recommendations = get_recommendations(stats)
            return idx, frame, aug_img, "\n".join(recommendations) if recommendations else "No recommendations at this time."
        
        def on_right(current_idx, flip_mode_val, rotation_val, brightness_val, contrast_val, blur_kernel_val, hue_shift_val, saturation_val, occlusion_size_val):
            idx = min(interface.total_frames - 1, current_idx + 1)
            frame = interface.get_frame(idx)
            aug_img = interface.get_augmented(idx, flip_mode_val, rotation_val, brightness_val, contrast_val, blur_kernel_val, hue_shift_val, saturation_val, occlusion_size_val)
            # Get recommendations based on current frame
            stats = analyze_sample(frame)
            recommendations = get_recommendations(stats)
            return idx, frame, aug_img, "\n".join(recommendations) if recommendations else "No recommendations at this time."
        
        def on_export(frame_idx, flip_mode_val, rotation_val, brightness_val, contrast_val, blur_kernel_val, hue_shift_val, saturation_val, occlusion_size_val):
            file_path, message = interface.export_augmented_data(
                frame_idx, flip_mode_val, rotation_val, brightness_val, contrast_val, blur_kernel_val, hue_shift_val, saturation_val, occlusion_size_val
            )
            return file_path, message
            
        def on_save_config(flip_mode_val, rotation_val, brightness_val, contrast_val, blur_kernel_val, hue_shift_val, saturation_val, occlusion_size_val):
            file_path, message = interface.export_config(
                flip_mode_val, rotation_val, brightness_val, contrast_val, blur_kernel_val, hue_shift_val, saturation_val, occlusion_size_val
            )
            return file_path, message
        
        # --- AGENT LOGIC ---
        def agent_adjust(recommendations_text, brightness_val, contrast_val, blur_kernel_val, hue_shift_val, saturation_val, occlusion_size_val, current_frame):
            # Reset to default values before applying new adjustments
            new_brightness = 1.0
            new_contrast = 1.0
            new_blur = 0
            new_hue = 0
            new_saturation = 1.0
            new_occlusion = 0
            # Parse recommendations and adjust accordingly
            if "Increase brightness" in recommendations_text:
                new_brightness = min(new_brightness + 0.2, 2.0)
            if "Decrease brightness" in recommendations_text:
                new_brightness = max(new_brightness - 0.2, 0.5)
            if "Increase contrast" in recommendations_text:
                new_contrast = min(new_contrast + 0.2, 2.0)
            if "Decrease contrast" in recommendations_text:
                new_contrast = max(new_contrast - 0.2, 0.5)
            if "Apply Blur" in recommendations_text:
                new_blur = min(new_blur + 2, 10)
            if "Increase sharpness" in recommendations_text:
                new_contrast = min(new_contrast + 0.2, 2.0)
            if "Increase color vibrancy" in recommendations_text:
                new_saturation = min(new_saturation + 0.5, 2.0)
                new_hue = random.randint(-90, 90)
            if "Reduce color intensity" in recommendations_text:
                new_saturation = max(new_saturation - 0.2, 0.0)
            if "Add occlusion" in recommendations_text:
                new_occlusion = min(new_occlusion + 0.05, 0.5)
            # Apply the new adjustments to the current frame
            params = {
                'flip_mode': flip_mode.value,
                'rotation': rotation.value,
                'brightness': new_brightness,
                'contrast': new_contrast,
                'blur_kernel': new_blur,
                'hue_shift': new_hue,
                'saturation': new_saturation,
                'occlusion_size': new_occlusion
            }
            augmented_frame = apply_augmentation(current_frame, params)
            # Compute stats for the augmented frame
            augmented_stats = analyze_sample(augmented_frame)
            # Compare with original stats
            original_stats = analyze_sample(current_frame)
            # Check if changes are too soft
            if abs(augmented_stats['brightness'] - original_stats['brightness']) < 0.2:
                new_brightness = min(new_brightness + 0.5, 2.0)
            if abs(augmented_stats['contrast'] - original_stats['contrast']) < 0.2:
                new_contrast = min(new_contrast + 0.5, 2.0)
            if abs(augmented_stats['saturation'] - original_stats['saturation']) < 0.2:
                new_saturation = min(new_saturation + 0.5, 2.0)
            return (
                new_brightness, new_contrast, new_blur, new_hue, new_saturation, new_occlusion
            )
        # Connect agent button
        agent_btn.click(
            fn=agent_adjust,
            inputs=[recommendations, brightness, contrast, blur_kernel, hue_shift, saturation, occlusion_size, frame_slider],
            outputs=[brightness, contrast, blur_kernel, hue_shift, saturation, occlusion_size]
        )
        
        # Connect events
        input_file.upload(
            fn=on_file_upload,
            inputs=[input_file, flip_mode, rotation, brightness, contrast, blur_kernel, hue_shift, saturation, occlusion_size],
            outputs=[original_preview, frame_slider, file_type, augmented_preview, file_name, frame_slider, recommendations]
        )
        
        demo_btn.click(
            fn=on_demo_click,
            inputs=[flip_mode, rotation, brightness, contrast, blur_kernel, hue_shift, saturation, occlusion_size],
            outputs=[original_preview, frame_slider, file_type, augmented_preview, file_name, frame_slider, recommendations]
        )
        
        frame_slider.change(
            fn=on_frame_change,
            inputs=[frame_slider, flip_mode, rotation, brightness, contrast, blur_kernel, hue_shift, saturation, occlusion_size],
            outputs=[original_preview, augmented_preview, recommendations]
        )
        
        left_btn.click(
            fn=on_left,
            inputs=[frame_slider, flip_mode, rotation, brightness, contrast, blur_kernel, hue_shift, saturation, occlusion_size],
            outputs=[frame_slider, original_preview, augmented_preview, recommendations]
        )
        
        right_btn.click(
            fn=on_right,
            inputs=[frame_slider, flip_mode, rotation, brightness, contrast, blur_kernel, hue_shift, saturation, occlusion_size],
            outputs=[frame_slider, original_preview, augmented_preview, recommendations]
        )
        
        # Add real-time updates for augmentation controls
        flip_mode.change(
            fn=on_frame_change,
            inputs=[frame_slider, flip_mode, rotation, brightness, contrast, blur_kernel, hue_shift, saturation, occlusion_size],
            outputs=[original_preview, augmented_preview, recommendations]
        )
        
        rotation.change(
            fn=on_frame_change,
            inputs=[frame_slider, flip_mode, rotation, brightness, contrast, blur_kernel, hue_shift, saturation, occlusion_size],
            outputs=[original_preview, augmented_preview, recommendations]
        )
        
        brightness.change(
            fn=on_frame_change,
            inputs=[frame_slider, flip_mode, rotation, brightness, contrast, blur_kernel, hue_shift, saturation, occlusion_size],
            outputs=[original_preview, augmented_preview, recommendations]
        )
        
        contrast.change(
            fn=on_frame_change,
            inputs=[frame_slider, flip_mode, rotation, brightness, contrast, blur_kernel, hue_shift, saturation, occlusion_size],
            outputs=[original_preview, augmented_preview, recommendations]
        )
        
        blur_kernel.change(
            fn=on_frame_change,
            inputs=[frame_slider, flip_mode, rotation, brightness, contrast, blur_kernel, hue_shift, saturation, occlusion_size],
            outputs=[original_preview, augmented_preview, recommendations]
        )
        
        hue_shift.change(
            fn=on_frame_change,
            inputs=[frame_slider, flip_mode, rotation, brightness, contrast, blur_kernel, hue_shift, saturation, occlusion_size],
            outputs=[original_preview, augmented_preview, recommendations]
        )
        
        saturation.change(
            fn=on_frame_change,
            inputs=[frame_slider, flip_mode, rotation, brightness, contrast, blur_kernel, hue_shift, saturation, occlusion_size],
            outputs=[original_preview, augmented_preview, recommendations]
        )
        
        occlusion_size.change(
            fn=on_frame_change,
            inputs=[frame_slider, flip_mode, rotation, brightness, contrast, blur_kernel, hue_shift, saturation, occlusion_size],
            outputs=[original_preview, augmented_preview, recommendations]
        )
        
        export_btn.click(
            fn=on_export,
            inputs=[frame_slider, flip_mode, rotation, brightness, contrast, blur_kernel, hue_shift, saturation, occlusion_size],
            outputs=[export_output, export_message]
        )
        
        save_config_btn.click(
            fn=on_save_config,
            inputs=[flip_mode, rotation, brightness, contrast, blur_kernel, hue_shift, saturation, occlusion_size],
            outputs=[config_output, config_message]
        )
        
        # --- NEW SECTION: DATABASE INPUT ---
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Coming Soon: Database Input\nYou will soon be able to input a complete database for advanced analysis and augmentation.")
    
    return app 