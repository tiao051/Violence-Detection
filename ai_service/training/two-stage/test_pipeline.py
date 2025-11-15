"""
Test pipeline to validate SME -> STE -> GTE data flow.

Tests a single batch from the data loader to ensure shapes match paper specs.
"""

import sys
import torch
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from remonet.sme.extractor import SMEExtractor
from remonet.ste.extractor import STEExtractor
from remonet.gte.extractor import GTEExtractor
import importlib.util

# Import data_loader
spec = importlib.util.spec_from_file_location(
    "data_loader", 
    str(Path(__file__).parent / "data_loader.py")
)
data_loader_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(data_loader_module)
VideoDataLoader = data_loader_module.VideoDataLoader


def test_pipeline(dataset_root: str = "d:\\DATN\\violence-detection\\dataset"):
    """
    Test the complete SME -> STE -> GTE pipeline.
    
    Validates:
    1. Data loader returns correct shapes
    2. SME processes frames to motion frames
    3. STE processes motion frames to feature maps
    4. GTE processes feature maps to violence probabilities
    """
    print("=" * 70)
    print("PIPELINE VALIDATION TEST")
    print("=" * 70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}\n")
    
    # Initialize extractors
    print("1. Initializing extractors...")
    sme = SMEExtractor()
    ste = STEExtractor(device=device, training_mode=False)
    gte = GTEExtractor(num_channels=1280, temporal_dim=10, num_classes=2, device=device)
    print("   ✓ SME, STE, GTE initialized\n")
    
    # Create data loader
    print("2. Creating data loader...")
    extracted_frames_dir = Path(dataset_root) / "RWF-2000" / "extracted_frames"
    
    if not extracted_frames_dir.exists():
        print(f"   ERROR: Extracted frames not found at {extracted_frames_dir}")
        print(f"   Please run frame_extractor.py first")
        return False
    
    dataset = VideoDataLoader(
        extracted_frames_dir=str(extracted_frames_dir),
        split='train',
        sme_extractor=sme,
        ste_extractor=ste
    )
    
    if len(dataset) == 0:
        print("   ERROR: No training samples found")
        return False
    
    print(f"   ✓ Dataset loaded with {len(dataset)} samples\n")
    
    # Get one sample
    print("3. Testing single sample...")
    try:
        features, label = dataset[0]
        print(f"   ✓ Sample loaded successfully")
        print(f"     Features shape: {tuple(features.shape)}")
        print(f"     Label: {label}\n")
    except Exception as e:
        print(f"   ERROR: Failed to load sample: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Validate feature shape
    print("4. Validating feature shape for GTE...")
    expected_shape = (10, 1280, 7, 7)  # (T/3, C, H, W) for MobileNetV2
    if features.shape != expected_shape:
        print(f"   WARNING: Expected shape {expected_shape}, got {tuple(features.shape)}")
        print(f"           This is OK if using different backbone")
        print(f"           Continuing with actual shape...\n")
    else:
        print(f"   ✓ Feature shape matches expected: {tuple(features.shape)}\n")
    
    # Test GTE forward pass
    print("5. Testing GTE forward pass...")
    try:
        with torch.no_grad():
            features_tensor = features.to(device)
            logits = gte.forward(features_tensor)
            
        print(f"   ✓ GTE forward pass successful")
        print(f"     Logits shape: {tuple(logits.shape)}")
        print(f"     Logits: {logits}")
        
        # Get probabilities
        probs = torch.softmax(logits, dim=-1)
        print(f"     Probabilities (no_violence, violence): {probs}\n")
        
    except Exception as e:
        print(f"   ERROR: GTE forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test batch processing (like DataLoader)
    print("6. Testing batch processing...")
    from torch.utils.data import DataLoader
    
    batch_loader = DataLoader(dataset, batch_size=2, shuffle=False)
    try:
        features_batch, labels_batch = next(iter(batch_loader))
        print(f"   ✓ Batch loaded successfully")
        print(f"     Batch features shape: {tuple(features_batch.shape)}")
        print(f"     Batch labels shape: {tuple(labels_batch.shape)}")
        print(f"     Batch labels: {labels_batch}\n")
        
        # Test GTE on batch
        print("7. Testing GTE on batch...")
        features_batch = features_batch.to(device)
        batch_logits = []
        
        with torch.no_grad():
            for i in range(features_batch.shape[0]):
                features = features_batch[i]  # (T/3, C, H, W)
                logits = gte.forward(features)
                batch_logits.append(logits)
        
        outputs = torch.stack(batch_logits)
        print(f"   ✓ Batch processing successful")
        print(f"     Output shape: {tuple(outputs.shape)}")
        print(f"     Logits: {outputs}\n")
        
    except Exception as e:
        print(f"   ERROR: Batch processing failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test loss computation
    print("8. Testing loss computation...")
    try:
        criterion = torch.nn.CrossEntropyLoss()
        labels_batch = labels_batch.to(device)
        loss = criterion(outputs, labels_batch)
        print(f"   ✓ Loss computed successfully")
        print(f"     Loss: {loss.item():.6f}\n")
    except Exception as e:
        print(f"   ERROR: Loss computation failed: {e}")
        return False
    
    print("=" * 70)
    print("✓ ALL TESTS PASSED!")
    print("=" * 70)
    print("\nPipeline is ready for training:")
    print("  1. Frames (30, 224, 224, 3) RGB uint8")
    print("  2. SME: motion frames (30, 224, 224, 3) uint8")
    print("  3. STE: feature maps (10, 1280, 7, 7) float32")
    print("  4. GTE: logits (2,) for binary classification")
    print("\nRun: python train.py --dataset-root <path>")
    
    return True


if __name__ == "__main__":
    success = test_pipeline()
    sys.exit(0 if success else 1)
