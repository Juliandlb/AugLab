# AugLab - Interactive Augmentation Playground

An interactive tool for applying and experimenting with image augmentations on robot perception data.

## Features

- Upload and view images, videos, or JSONL files
- Interactive real-time augmentation controls:
  - Image flipping (horizontal/vertical)
  - Rotation (-45° to 45°)
  - Brightness adjustment (0.5x to 2.0x)
  - Contrast adjustment (0.5x to 2.0x)
- Live preview of augmented data with synchronized frame navigation
- Smart augmentation recommendations (coming soon)
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
│   └── gradio_interface.py  # Gradio UI implementation with real-time controls
├── core/
│   ├── loader.py       # Data loading utilities for images, videos, and JSONL
│   ├── augment.py      # Augmentation functions (flip, rotate, brightness, contrast)
│   ├── stats.py        # Image analysis and recommendations (in development)
│   └── utils.py        # Utility functions for config and data export (in development)
└── assets/             # Static assets
```

## Current Status

- ✅ Basic UI implementation with Gradio
- ✅ Real-time augmentation controls
- ✅ Frame navigation with synchronized previews
- ✅ Support for multiple file formats (images, videos, JSONL)
- 🔄 Smart recommendations (in development)
- 🔄 Export functionality (in development)

## License

MIT License 