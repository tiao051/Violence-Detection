"""
Quick model checkpoint inspector.

Display model information and training metrics from checkpoint.

Usage:
    python check_model.py --checkpoint checkpoints/best_model_hf.pt
    python check_model.py --checkpoint checkpoints/best_model_rwf.pt
    python check_model.py --checkpoint checkpoints/best_model_hf.pt --show-layers
"""

import torch
import argparse
from pathlib import Path
from datetime import datetime


def format_size(size_bytes: float) -> str:
    """Convert bytes to human readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.2f} TB"


def inspect_checkpoint(checkpoint_path: str, show_layers: bool = False):
    """Inspect checkpoint file."""
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        print(f"Checkpoint not found: {checkpoint_path}")
        return

    # File info
    print(f"\nFILE INFORMATION")
    print(f"{'─'*70}")
    print(f"Path:       {checkpoint_path}")
    print(f"Size:       {format_size(checkpoint_path.stat().st_size)}")
    
    timestamp = datetime.fromtimestamp(checkpoint_path.stat().st_mtime)
    print(f"Modified:   {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load checkpoint
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
    except Exception as e:
        print(f"\nError loading checkpoint: {e}")
        return
    
    # Checkpoint keys
    print(f"\nCHECKPOINT KEYS")
    print(f"{'─'*70}")
    for key in checkpoint.keys():
        print(f"  • {key}")
    
    # Training metrics
    print(f"\nTRAINING METRICS")
    print(f"{'─'*70}")
    if 'best_val_acc' in checkpoint:
        best_val_acc = checkpoint['best_val_acc']
        print(f"Best Val Accuracy:  {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
    
    # Model state dict info
    if 'model_state_dict' in checkpoint:
        model_dict = checkpoint['model_state_dict']
        
        print(f"\nMODEL INFORMATION")
        print(f"{'─'*70}")
        print(f"Total Parameters: {sum(p.numel() for p in model_dict.values()):,}")
        
        # Model size estimation
        total_size = 0
        for param_tensor in model_dict.values():
            total_size += param_tensor.numel() * param_tensor.element_size()
        print(f"Model Size:       {format_size(total_size)}")
        
        if show_layers:
            print(f"\nLAYER BREAKDOWN")
            print(f"{'─'*70}")
            total_params = 0
            for name, param in model_dict.items():
                num_params = param.numel()
                param_size = format_size(num_params * param.element_size())
                total_params += num_params
                print(f"  {name:<50} {num_params:>12,} params ({param_size:>10})")
            print(f"{'─'*70}")
            print(f"  {'TOTAL':<50} {total_params:>12,} params")
    
    # Optimizer state dict info
    if 'optimizer_state_dict' in checkpoint:
        optimizer_dict = checkpoint['optimizer_state_dict']
        
        print(f"\nOPTIMIZER INFORMATION")
        print(f"{'─'*70}")
        if 'param_groups' in optimizer_dict:
            for i, group in enumerate(optimizer_dict['param_groups']):
                print(f"Param Group {i}:")
                for key, value in group.items():
                    if key != 'params':
                        print(f"  {key:<20} {value}")
    

def main():
    parser = argparse.ArgumentParser(
        description='Quick model checkpoint inspector',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    
    parser.add_argument('--checkpoint', type=str, help='Path to checkpoint file')
    parser.add_argument('--show-layers', action='store_true', help='Show layer breakdown')
    
    args = parser.parse_args()
    
    if args.checkpoint:
        inspect_checkpoint(args.checkpoint, show_layers=args.show_layers)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
