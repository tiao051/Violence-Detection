import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import sys
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import json
from tqdm import tqdm

# Add ai_service to path (3 levels up from here: training/two-stage/ -> ai_service root is ../../)
# Actually, we need project root to import ai_service if we run as script
# But typically we want to import from ai_service root.
# Let's align with train.py: sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent)) # Violence-Detection root
sys.path.insert(0, str(Path(__file__).parent.parent.parent)) # ai_service root

from ai_service.remonet.gte.extractor import GTEExtractor
from ai_service.remonet.ste.extractor import BackboneType, BACKBONE_CONFIG
from data_loader import VideoDataLoader, AugmentationConfig

def evaluate(model_path, dataset_root, backbone='mobilenet_v3_small', batch_size=8, device='cuda'):
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load Model
    backbone_config = BACKBONE_CONFIG[BackboneType(backbone)]
    num_channels = backbone_config['out_channels']
    temporal_dim = 10
    
    model = GTEExtractor(
        num_channels=num_channels,
        temporal_dim=temporal_dim,
        num_classes=2,
        device=device
    )
    
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint['model_state_dict']
    
    # Filter out thop metadata (total_ops, total_params)
    new_state_dict = {}
    for k, v in state_dict.items():
        if 'total_ops' not in k and 'total_params' not in k:
            new_state_dict[k] = v
            
    model.load_state_dict(new_state_dict, strict=False)
    model.to(device)
    model.eval()
    
    # Load Data
    extracted_frames_dir = Path(dataset_root) / 'UVD' / 'extracted_frames'
    dataset = VideoDataLoader(
        extracted_frames_dir=str(extracted_frames_dir),
        split='val', # Use validation set for evaluation
        dataset='uvd',
        backbone=backbone,
        augmentation_config=AugmentationConfig(enable_augmentation=False)
    )
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    all_preds = []
    all_labels = []
    
    print("Running evaluation...")
    with torch.no_grad():
        for features, labels in tqdm(dataloader):
            features = features.to(device).float()
            outputs = model(features)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            
    # Calculate Metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='binary', pos_label=0) # 0 is Violence
    recall = recall_score(all_labels, all_preds, average='binary', pos_label=0)
    f1 = f1_score(all_labels, all_preds, average='binary', pos_label=0)
    cm = confusion_matrix(all_labels, all_preds)
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm.tolist()
    }
    
    print("\nEvaluation Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("Confusion Matrix:")
    print(cm)
    
    # Save Metrics
    with open('uvd_evaluation_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
        
    # Plot Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Violence', 'NonViolence'], yticklabels=['Violence', 'NonViolence'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix - UVD')
    plt.savefig('uvd_confusion_matrix.png')
    print("Metrics saved to uvd_evaluation_metrics.json and uvd_confusion_matrix.png")

if __name__ == "__main__":
    # Check if model exists
    # Model is typically in checkpoints folder relative to this script
    model_path = Path(__file__).parent / 'checkpoints' / 'best_model.pt'
    if not model_path.exists():
        print(f"Model not found at {model_path}. Please train first.")
        sys.exit(1)
        
    evaluate(
        model_path=str(model_path),
        dataset_root='dataset', # Assumes running from project root
        backbone='mobilenet_v3_small' # Ensure this matches training
    )
