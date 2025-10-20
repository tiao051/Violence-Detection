"""
evaluate_fps.py

Description:
    This script provides a standardized way to evaluate the inference performance (FPS)
    of a trained YOLOv8 model on a given video file. It calculates key metrics such as
    average latency and FPS, and can optionally generate an output video with
    the model's predictions rendered.

Usage:
    python training/evaluation/evaluate_fps.py
"""
import cv2
import time
import torch
import sys
from pathlib import Path

# Add the project root to the Python path to enable seamless module imports.
# This allows the script to find modules like `ultralytics` from the project's context.
# Assumes the script resides in `training/evaluation`, thus navigating two levels up.
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from ultralytics import YOLO

def evaluate_model_fps(model_path: str, video_path: str, output_video_path: str = None):
    """
    Measures and reports the inference speed (FPS) of a YOLO model on a video.

    This function streams a video file, runs inference on each frame, and calculates
    performance metrics. It's designed to provide a consistent benchmark for model speed.

    Args:
        model_path (str): The file path to the trained YOLO model weights (.pt).
        video_path (str): The file path to the input video for evaluation.
        output_video_path (str, optional): If specified, the function will save an
                                           annotated video to this path. Defaults to None.
    """
    print("--- Initializing FPS Evaluation ---")

    # --- Step 1: Environment Setup & Verification ---
    # Determine the computation device (prioritize CUDA if available).
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Computation device selected: {device.upper()}")
    if device == 'cuda':
        print(f"GPU Details: {torch.cuda.get_device_name(0)}")

    # --- Step 2: Load the YOLOv8 Model ---
    try:
        print(f"Loading model from '{model_path}'...")
        model = YOLO(model_path)
        # Transfer the model to the selected compute device.
        model.to(device)
        print("Model loaded and configured successfully.")
    except Exception as e:
        print(f"FATAL: Failed to load the model. Aborting. Error: {e}")
        return

    # --- Step 3: Initialize Video Capture and Writer ---
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video source at '{video_path}'. Please check the path.")
        return

    # Retrieve video properties to configure the output writer.
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    
    out_writer = None
    if output_video_path:
        # Define the codec and create VideoWriter object. 'mp4v' is a good default for .mp4 files.
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_writer = cv2.VideoWriter(output_video_path, fourcc, video_fps, (frame_width, frame_height))
        print(f"Output video will be saved to: {output_video_path}")

    # --- Step 4: Frame-by-Frame Inference and Timing ---
    frame_count = 0
    total_inference_time = 0.0
    
    print("\nProcessing video stream for inference...")
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            # End of the video stream.
            break

        frame_count += 1

        # Isolate the inference call for precise timing.
        start_time = time.perf_counter()
        
        # Perform inference on the current frame.
        # `verbose=False` suppresses the detailed per-frame output from the YOLO model.
        results = model(frame, verbose=False)
        
        end_time = time.perf_counter()
        
        # Aggregate the time taken for model inference.
        total_inference_time += (end_time - start_time)

        # If an output path is provided, render the results and write the frame.
        if output_video_path:
            # `results[0].plot()` is a convenience method that draws detections on the frame.
            annotated_frame = results[0].plot()
            out_writer.write(annotated_frame)

    print("Finished processing all frames.")

    # --- Step 5: Calculate and Display Performance Metrics ---
    if frame_count > 0:
        avg_latency_ms = (total_inference_time / frame_count) * 1000
        avg_fps = frame_count / total_inference_time

        print("\n--- Performance Evaluation Summary ---")
        print(f"Total Frames Processed: {frame_count}")
        print(f"Total Inference Time: {total_inference_time:.2f} seconds")
        print(f"Average Latency per Frame: {avg_latency_ms:.2f} ms")
        print(f"Average Inference Speed: {avg_fps:.2f} FPS")
        print("------------------------------------")
    else:
        print("Warning: No frames were processed from the video source.")

    # --- Step 6: Cleanup ---
    # Release all video-related resources.
    cap.release()
    if out_writer:
        out_writer.release()
    cv2.destroyAllWindows()
    print("Resources have been released.")


if __name__ == '__main__':
    """
    This block allows the script to be run directly from the command line.
    Users should configure the model and video paths below.
    """
    # --- CONFIGURATION SECTION ---
    # Define the path to the model weights to be evaluated.
    MODEL_TO_EVALUATE = 'training/training_runs/exp001_yolov8n_baseline/weights/best.pt'
    
    # Define the path to the video file for the performance test.
    VIDEO_FOR_TEST = 'test-data/val_video_01.avi' # Ensure this file exists.

    # (Optional) Specify a path to save the output video with detections.
    # Set to `None` to disable video saving.
    OUTPUT_VIDEO_NAME = 'evaluation_results/yolov8n_output.mp4'

    # --- EXECUTION ---
    # Ensure the output directory exists before starting.
    if OUTPUT_VIDEO_NAME:
        output_dir = Path(OUTPUT_VIDEO_NAME).parent
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Execute the evaluation function with the configured paths.
    evaluate_model_fps(
        model_path=MODEL_TO_EVALUATE,
        video_path=VIDEO_FOR_TEST,
        output_video_path=OUTPUT_VIDEO_NAME
    )