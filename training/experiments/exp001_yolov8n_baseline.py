"""
exp001_yolov8n_baseline.py

Description:
    This script initiates a baseline training session for a YOLOv8n model
    on the custom violence detection dataset. It uses a pretrained YOLOv8n
    model and fine-tunes it according to the parameters specified in the
    'violence_config.yaml' file.

Usage:
    python training/experiments/exp001_yolov8n_baseline.py
"""
from ultralytics import YOLO
import torch

def run_training():
    """
    Orchestrates the model training process.
    """
    # --- Environment Verification ---
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Current GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}")

    # --- Model Initialization ---
    # Load the official YOLOv8n weights as a starting point for transfer learning.
    print("Initializing model with 'yolov8n.pt' pretrained weights...")
    model = YOLO('weights/pretrained/yolov8n.pt')

    # --- Training Execution ---
    # The `train` method handles the entire training loop, including data loading,
    # augmentation, validation, and logging.
    print("Starting model training...")
    results = model.train(
        data='training/configs/violence_config.yaml',  # Path to the dataset configuration file.
        epochs=50,                                     # Total number of training epochs.
        imgsz=640,                                     # Input image size (height, width).
        batch=16,                                      # Number of images per batch.
        project='training/training_runs',              # Directory to save all training runs.
        name='exp001_yolov8n_baseline'                 # Specific name for this experiment run.
    )
    print("Training session completed successfully!")
    print(f"All results, logs, and weights are saved in: {results.save_dir}")

if __name__ == '__main__':
    # This block ensures the training function is called only when the script is executed directly.
    run_training()
