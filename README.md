# AugLab - Interactive Augmentation Playground

An interactive tool for applying and experimenting with image augmentations on robot perception data.

## Features

- Upload and view images, videos, or JSONL files
- Interactive augmentation controls
- Live preview of augmented data
- Smart augmentation recommendations
- Export augmented data and configurations

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
│   └── gradio_interface.py  # Gradio UI implementation
├── core/
│   ├── loader.py       # Data loading utilities
│   ├── augment.py      # Augmentation functions
│   ├── stats.py        # Image analysis and recommendations
│   └── utils.py        # Utility functions
└── assets/             # Static assets
```

## License

MIT License 