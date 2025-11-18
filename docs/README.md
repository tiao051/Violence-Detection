# Violence Detection Paper - Architecture Understanding

## Overview
The paper proposes a hybrid architecture for real-time violence detection in videos that balances **efficiency** and **effectiveness**. The key innovation is achieving high accuracy while maintaining low computational cost for deployment in real-world scenarios.

## Architecture Components

The model consists of three main modules working sequentially:

### 1. Spatial Motion Extractor (SME)
**Purpose**: Extract regions of interest containing movement from video frames

**Input**: 30 consecutive RGB frames, resized to 224×224 pixels

**Process**:
1. **Motion Detection**: Calculate Euclidean distance between two consecutive frames F_t and F_{t+1}:
   - Formula: d_t = sqrt(sum over 3 channels of (F^i_{t+1} - F^i_t)^2)
   - Result: Grayscale motion boundaries with background removed

2. **Region Expansion**: Apply morphological dilation
   - Kernel: 3×3
   - Iterations: 12
   - Purpose: Convert motion boundaries into motion regions (b_t)

3. **Motion Extraction**: Dot product between dilated mask and original frame
   - Result: M_t ∈ R^{3×W×H} - frame with violent scene highlighted on black background

**Key Advantage**: 
- Acts as an attention mechanism using only image processing operations
- **Zero learnable parameters** - purely computational
- Focuses model on relevant motion areas

**Output**: 30 frames with motion regions highlighted (M_t for t=0 to 29)

---

### 2. Short Temporal Extractor (STE)
**Purpose**: Extract short-term spatiotemporal features efficiently using 2D CNN

**Input**: 30 motion-extracted frames (M_t) from SME

**Process**:
1. **Frame Averaging**: For every 3 consecutive frames M_t, M_{t+1}, M_{t+2}:
   - Calculate average across RGB channels for each frame: p_t = sum over 3 channels of (M^c_t)
   - Results in 3 grayscale-like values: p_t, p_{t+1}, p_{t+2}

2. **Temporal Composite Assembly**: Stack the 3 averaged values as channels
   - Create single frame P_{t,t+1,t+2} with 3 channels
   - Each channel represents temporal information from one frame
   - Color information is intentionally lost (not relevant for violence detection)

3. **Feature Extraction**: Pass composite through 2D CNN backbone (MobileNetV2)
   - Treats temporal composite as a regular RGB image
   - Extracts spatiotemporal features efficiently

**Key Advantages**:
- **Reduces frame count from 30 to 10** (T/3) - massive efficiency gain
- Uses efficient 2D CNN instead of computationally expensive 3D CNN, LSTM, or two-stream networks
- Captures rapid movements characteristic of violent actions (punches, kicks, throws)

**Output**: Feature map B ∈ R^{T/3 × C × W × H}
- T/3 = 10 temporal frames
- C = 1280 channels (MobileNetV2 output)
- W × H = 7×7 spatial dimensions

---

### 3. Global Temporal Extractor (GTE)
**Purpose**: Learn long-term temporal relationships across all 10 frames to improve effectiveness

**Input**: Feature map B from STE (10 × 1280 × 7 × 7)

**Process**:

1. **Spatial Compression** (Global Average Pooling):
   - Formula: S^c = (1/(H×W)) × sum over H,W of B^c(i,j)
   - Apply GAP to each of the 1280 channels across 7×7 spatial dimensions
   - Result: S = [b^c_1, b^c_2, ..., b^c_{T/3}] where each b^c is a 1280-dimensional vector
   - Shape: (10, 1280) - one feature vector per temporal frame

2. **Temporal Compression** (Global Average Pooling across time):
   - Formula: q = (1/C) × sum over C of S^c
   - Average across all 1280 channels for each temporal position
   - Result: q = [q_1, q_2, ..., q_{T/3}] - a 10-dimensional vector
   - Captures average temporal activation pattern

3. **Temporal Excitation** (Learning temporal importance):
   - Pass q through two fully connected layers with sigmoid activation
   - FC1: Reduces dimension (e.g., 10 → 5)
   - ReLU activation
   - FC2: Restores dimension (e.g., 5 → 10)
   - Sigmoid activation
   - Result: E = [e_1, e_2, ..., e_{T/3}] - temporal importance weights in range [0, 1]
   - **Purpose**: Learn which temporal frames are most important for violence detection

4. **Channel Recalibration** (Feature refinement):
   - Element-wise multiplication between E and S
   - Reweights each temporal frame's features by its importance
   - Result: S' - recalibrated features emphasizing important temporal moments

5. **Temporal Aggregation**:
   - Sum S' across temporal dimension (10 frames)
   - Result: F - single 1280-dimensional feature vector representing entire video

6. **Classification**:
   - Final fully connected layer: 1280 → 2
   - Outputs: [violence_score, non_violence_score]
   - Softmax for probabilities

**Key Advantages**:
- Captures long-term temporal relationships across all frames
- Learns to focus on temporally important moments
- Minimal computational overhead (only FC layers and pooling)
- Improves effectiveness without compromising efficiency

**Output**: Classification logits/probabilities for Violence vs Non-Violence

---

## Complete Pipeline Summary

```
Input: 30 frames (224×224 RGB)
    ↓
[SME] Spatial Motion Extraction (0 parameters)
    → 30 motion-highlighted frames
    ↓
[STE] Short Temporal Extractor (MobileNetV2 backbone)
    → Group 3 frames → average channels → create temporal composite
    → 10 composites → extract features via 2D CNN
    → Output: (10, 1280, 7, 7) feature map
    ↓
[GTE] Global Temporal Extractor
    → Spatial compression: (10, 1280, 7, 7) → (10, 1280)
    → Temporal compression: (10, 1280) → (10,)
    → Temporal excitation: (10,) → FC layers → (10,) weights
    → Channel recalibration: apply weights to features
    → Temporal aggregation: (10, 1280) → (1280,)
    → Classification: (1280,) → 2 classes
    ↓
Output: Violence / Non-Violence prediction
```

---

## Key Design Principles

1. **Efficiency First**:
   - SME uses only image processing (no parameters)
   - STE reduces frames from 30 to 10 (3x reduction)
   - Uses efficient 2D CNN (MobileNetV2) instead of 3D CNN
   - Minimal overhead in GTE (only pooling + small FC layers)

2. **Effectiveness Through Smart Design**:
   - SME acts as attention mechanism on motion
   - STE captures short-term rapid movements (characteristic of violence)
   - GTE learns long-term temporal patterns and frame importance
   - End-to-end trainable (except SME preprocessing)

3. **Inspired by Human Perception**:
   - SME mimics human visual attention focusing on movement
   - Hierarchical temporal understanding (short-term → long-term)

4. **Real-World Deployment Ready**:
   - Low FLOPs for real-time processing
   - Suitable for video surveillance systems
   - Balance of accuracy and speed

---

## Mathematical Operations Summary

### SME Module (Spatial Motion Extractor)
```
1. Euclidean Distance between frames:
   - Compare RGB values of consecutive frames pixel-by-pixel
   - Calculate: distance = sqrt(sum of squared differences across 3 channels)
   
2. Morphological Dilation:
   - Expand motion boundaries using 3×3 kernel, 12 iterations
   - Purpose: Convert thin motion lines into motion regions
   
3. Motion Extraction:
   - Multiply dilated mask with original frame (pixel-wise)
   - Result: Highlights motion areas on black background
```

### STE Module (Short Temporal Extractor)
```
1. Channel Averaging (Reduce color info to temporal info):
   - For each frame, average its RGB channels → single value
   - Example: If R=100, G=110, B=120 → average = 110
   
2. Temporal Composite Assembly:
   - Take 3 consecutive averaged frames
   - Stack them as 3 channels of a single image
   - Now each "channel" represents one frame's temporal data
   
3. Feature Extraction:
   - Pass temporal composite through MobileNetV2 CNN
   - Extract spatial features at 7×7 resolution with 1280 channels
```

### GTE Module (Global Temporal Extractor)
```
1. Spatial Compression (Summarize spatial info):
   - Average each 1280 channels across 7×7 spatial dimensions
   - Result: 10 vectors, each 1280-dimensional (one per frame)
   
2. Temporal Compression (Find avg temporal pattern):
   - Average all 1280 channels across 10 frames
   - Result: Single 10-dimensional vector (one value per frame)
   
3. Temporal Excitation (Learn frame importance):
   - Pass through 2 fully connected layers + ReLU + Sigmoid
   - Output: Importance weight for each of 10 frames (0 to 1)
   
4. Channel Recalibration:
   - Multiply each frame's features by its importance weight
   - Emphasizes important frames, suppresses unimportant ones
   
5. Temporal Aggregation:
   - Sum all 10 frames' weighted features
   - Result: Single 1280-dimensional vector for entire video
   
6. Classification:
   - Fully connected layer: 1280 → 2 classes
   - Output: Violence probability vs Non-Violence probability
```

---

## Implementation Notes

### Current Codebase Alignment

The codebase structure matches the paper:

1. **SME Implementation** (`ai_service/remonet/sme/extractor.py`):
   - Euclidean distance calculation
   - Morphological dilation (3×3 kernel, 8 iterations - paper uses 12)
   - Binary thresholding
   - Motion extraction via masking

2. **STE Implementation** (`ai_service/remonet/ste/extractor.py`):
   - Channel averaging for 3 frames
   - Temporal composite creation
   - MobileNetV2 backbone
   - Outputs feature map (10, 1280, 7, 7)

3. **GTE Implementation** (`ai_service/remonet/gte/extractor.py`):
   - Spatial compression (Global Average Pooling)
   - Temporal compression
   - Temporal Excitation module (2 FC layers)
   - Channel recalibration
   - Temporal aggregation
   - Final classification layer

### Training Pipeline
- Dataset: RWF-2000 (primary), Hockey Fight (secondary)
- Input: 30 frames extracted at 30 FPS, 224×224 resolution
- Frame extraction → SME → STE → GTE → Classification
- Loss: Cross-entropy for binary classification
- Optimizer: Adam with OneCycleLR scheduler
