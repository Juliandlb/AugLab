# AugLab - Interactive Augmentation Playground

An interactive tool for applying and experimenting with image augmentations on robot perception data.

## Features

- Upload and view images, videos, or JSONL files
- Interactive real-time augmentation controls (all in one panel):
  - Image flipping (horizontal/vertical)
  - Rotation (-45° to 45°)
  - Brightness adjustment (0.5x to 2.0x)
  - Contrast adjustment (0.5x to 2.0x)
  - **Blur** (Gaussian blur, kernel size 0–10)
  - **Color jitter**:
    - Hue shift (-90° to 90°)
    - Saturation adjustment (0.0x to 2.0x)
  - **Occlusion** (random rectangle patch, size 0–0.5 ratio)
- Live preview of augmented data with synchronized frame navigation
- Smart, actionable augmentation recommendations based on image analysis:
  - Only suggests available controls: blur, color jitter (hue/saturation), occlusion, brightness, contrast, sharpness
  - Context-aware advice (e.g., "Apply Blur to reduce sharpness", "Increase color vibrancy using Saturation slider")
  - Brightness, contrast, sharpness, and saturation suggestions
- Export augmented data and configurations (coming soon)

## Setup

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python main.py
```

## Project Structure

```
auglab/
├── main.py              # Main entry point
├── ui/
│   └── gradio_interface.py  # Gradio UI implementation with all controls in one panel
├── core/
│   ├── loader.py       # Data loading utilities for images, videos, and JSONL
│   ├── augment.py      # Augmentation functions (flip, rotate, brightness, contrast, blur, color jitter, occlusion)
│   ├── stats.py        # Image analysis and recommendations
│   └── utils.py        # Utility functions for config and data export (in development)
└── assets/             # Static assets
```

## Current Status

- ✅ Basic UI implementation with Gradio
- ✅ Real-time augmentation controls (full set)
- ✅ Frame navigation with synchronized previews
- ✅ Support for multiple file formats (images, videos, JSONL)
- ✅ Smart recommendations based on image analysis
- 🔄 Export functionality (in development)

## Image Analysis Features

The application analyzes images in real-time to provide actionable, context-aware augmentation recommendations. Only implemented augmentations are suggested:

- **Brightness Analysis**: Detects if images are too dark or too bright
- **Contrast Analysis**: Measures image contrast and suggests adjustments
- **Sharpness Detection**: Identifies blurry images and suggests improvements
- **Saturation Analysis**: For color images, suggests saturation adjustments
- **Entropy Calculation**: Measures image information content

## License

MIT License 