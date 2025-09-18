# Detection Phase Preprocessing

This document describes the **preprocessing pipeline** for the Detection phase.  
The goal is to transform raw input frames into a format suitable for object detection models (e.g., YOLO).

---

## 1. Frame Preprocessing

The core preprocessing steps ensure that raw frames are aligned with the model’s input requirements.

**Pipeline:**

**Steps:**

- **Resize**  
  - Scale input image to the target model size.  
  - Example: `640x640` (YOLO) or `416x416` (MobileNet).

- **Normalize**  
  - Convert pixel values from `0–255` range to `0–1`.  
  - Optionally standardize using dataset-specific statistics (e.g., ImageNet mean & std).

- **Format Conversion**  
  - Convert from **BGR → RGB** (OpenCV uses BGR, most models expect RGB).  
  - Rearrange dimensions from **HWC → CHW** (Height, Width, Channel → Channel, Height, Width).

- **Batch Preparation**  
  - Add an additional batch dimension (e.g., `1x3x640x640`) for inference.

---

## 2. Quality Enhancement (Optional)

These steps are optional and used to improve detection accuracy under challenging video conditions.

- **Denoising**  
  - Remove video or sensor noise for cleaner input.

- **Brightness/Contrast Adjustment**  
  - Automatically adjust lighting for frames captured in poor conditions.

- **Sharpening**  
  - Enhance edges to make objects more distinct, especially in blurry frames.

---

## Example Flow

1. Capture frame from video stream.  
2. Resize frame → `640x640`.  
3. Normalize pixel values → `[0,1]`.  
4. Convert BGR → RGB, format to CHW.  
5. Add batch dimension.  
6. (Optional) Apply denoising or sharpening.  
7. Send preprocessed frame to detection model.

---

## Notes

- The exact preprocessing configuration depends on the detection model being used.  
- Over-processing (e.g., excessive sharpening or denoising) may reduce detection accuracy.  
- Always validate enhancements against real-world test data.
