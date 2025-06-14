#!/usr/bin/env python3

from ui.gradio_interface import create_interface
from core.loader import load_data
from core.augment import apply_augmentation
from core.stats import analyze_sample, get_recommendations

def main():
    """
    Main entry point for AugLab application.
    Initializes the Gradio interface and starts the server.
    """
    # Create and launch the Gradio interface
    interface = create_interface()
    interface.launch()

if __name__ == "__main__":
    main() 