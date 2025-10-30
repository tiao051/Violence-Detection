"""
exp002_yolov8s_baseline.py

Description:
    This script initiates a baseline training session for a YOLOv8s model
    on the custom violence detection dataset. It uses a pretrained YOLOv8s
    model and fine-tunes it according to the parameters specified in the
    'violence_config.yaml' file.

Usage:
    python training/experiments/exp002_yolov8s_baseline.py
"""
from ultralytics import YOLO
import torch
import yaml
from pathlib import Path

# Path to the configuration file
CONFIG_PATH = 'training/configs/violence_config.yaml'

def load_training_params(path: str) -> dict:
    """Loads training parameters from the YAML configuration file."""
    try:
        with open(path, 'r') as f:
            config = yaml.safe_load(f)
        # Return the training parameters (epochs, batch, imgsz, project)
        return config.get('training_parameters', {})
    except FileNotFoundError:
        print(f"FATAL: Configuration file not found at {path}")
        return {}

def run_training():
    """
    Orchestrates the model training process.
    """
    # --- Load Training Parameters ---
    train_params = load_training_params(CONFIG_PATH)
    if not train_params:
        return  
    
    # --- Environment Verification ---
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Current GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}")

    # --- Model Initialization ---
    # Load the official YOLOv8s weights as a starting point for transfer learning.
    print("Initializing model with 'yolov8s.pt' pretrained weights...")
    model = YOLO('weights/pretrained/yolov8s.pt')

    # --- Training Execution ---
    # The `train` method handles the entire training loop, including data loading,
    # augmentation, validation, and logging.
    print("Starting model training...")
    results = model.train(
        data=CONFIG_PATH,                                                    # Path to the dataset configuration file.
        epochs=train_params.get('epochs'),                                   # Total number of training epochs.
        imgsz=train_params.get('imgsz'),                                     # Input image size (height, width).
        batch=train_params.get('batch'),                                     # Number of images per batch.
        project=train_params.get('project'),                                 # Directory to save all training runs.
        name='exp002_yolov8s_baseline'                                       # Specific name for this experiment run.
    )
    print("Training session completed successfully!")
    print(f"All results, logs, and weights are saved in: {results.save_dir}")

if __name__ == '__main__':
    # This block ensures the training function is called only when the script is executed directly.
    run_training()
