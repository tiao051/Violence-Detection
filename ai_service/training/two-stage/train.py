"""
Training script for RWF-2000 violence detection using SME+STE features.

Usage:
    python train.py --dataset {rwf-2000|hockey-fight} [OPTIONS]

Arguments:
    --dataset {rwf-2000,hockey-fight}  Dataset to use for training (required)

Options:
    --dataset-root PATH                Path to dataset root (default: auto-detect from workspace)
    --backbone BACKBONE                STE backbone for feature extraction (default: mobilenet_v2)
    --epochs EPOCHS                    Number of epochs (default: 100)
    --batch-size BATCH_SIZE            Batch size (default: 2)
    --lr LEARNING_RATE                 Learning rate (default: 0.001)
    --device DEVICE                    Device to use: cpu/cuda (default: cuda if available)
    --num-workers NUM_WORKERS          Number of DataLoader workers (default: 2)

Examples:
    python train.py --dataset rwf-2000 --backbone mobilenet_v2
    python train.py --dataset rwf-2000 --backbone mobilenet_v3_small
    python train.py --dataset rwf-2000 --epochs 20
    python train.py --dataset hockey-fight --epochs 50 --batch-size 4
    python train.py --dataset rwf-2000 --dataset-root d:/custom/path --epochs 50
"""

import argparse
import sys
import logging
from thop import profile
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime

import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from data_loader import VideoDataLoader, AugmentationConfig
from remonet.gte.extractor import GTEExtractor
from remonet.ste.extractor import BackboneType, BACKBONE_CONFIG

@dataclass
class TrainConfig:
    """Training configuration"""
    epochs: int = 100
    batch_size: int = 8
    learning_rate: float = 1e-3
    weight_decay: float = 1e-2
    adam_epsilon: float = 1e-9
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    extracted_frames_dir: str = None
    num_workers: int = 2
    dataset: str = None  # Dataset name: 'rwf-2000' or 'hockey-fight' (required)
    backbone: str = 'mobilenet_v2'  # STE backbone to use for feature extraction
    
    # One-Cycle LR Scheduler config (per paper)
    scheduler_max_lr: float = 1e-3
    scheduler_min_lr: float = 1e-8
    
    # Augmentation config
    augmentation_config: AugmentationConfig = None
    
    def __post_init__(self):
        """Initialize default augmentation config if not provided."""
        if self.augmentation_config is None:
            self.augmentation_config = AugmentationConfig(enable_augmentation=True)


class Trainer:
    
    """Trainer for violence detection model."""
    
    def __init__(self, config: TrainConfig):
        """Initialize trainer."""
        self.config = config
        self.device = torch.device(config.device)
        
        # Setup logging
        self.logger = self._setup_logger()
        self.logger.info("Training Configuration")
        self.logger.info(f"Epochs: {config.epochs}")
        self.logger.info(f"Batch size: {config.batch_size}")
        self.logger.info(f"Learning rate: {config.learning_rate}")
        self.logger.info(f"Weight decay: {config.weight_decay}")
        self.logger.info(f"Adam epsilon: {config.adam_epsilon}")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Extracted frames: {config.extracted_frames_dir}")
        self.logger.info(f"Num workers: {config.num_workers}")
        self.logger.info(f"Backbone: {config.backbone}")
        
        # Create data loaders
        self.train_loader = self._create_dataloader('train')
        self.val_loader = self._create_dataloader('val')
        
        self.logger.info(f"Train samples: {len(self.train_loader.dataset)}")
        self.logger.info(f"Val samples: {len(self.val_loader.dataset)}\n")
        
        # Get backbone configuration from STE
        backbone_config = BACKBONE_CONFIG[BackboneType(config.backbone)]
        num_channels = backbone_config['out_channels']
        temporal_dim = 10  # 30 frames / 3 = 10
        spatial_size = backbone_config['spatial_size']
        self.logger.info(f"Expected STE feature shape: ({temporal_dim}, {num_channels}, {spatial_size}, {spatial_size})\n")
        
        
        # Initialize GTE model
        self.model = GTEExtractor(
            num_channels=num_channels,
            temporal_dim=temporal_dim,
            num_classes=2,
            device=config.device
        )
        
        # Meansure total system peroformance (FLOPS, params)
        self._measure_total_system(
            gte_model=self.model,
            temporal_dim=temporal_dim,
            channels=num_channels,
            device=config.device
        )
        
        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = Adam(
            self.model.parameters(),
            lr=config.learning_rate,
            eps=config.adam_epsilon,
            weight_decay=config.weight_decay
        )
        
        # Learning rate scheduler (One-Cycle per paper)
        # total_steps = num_batches * num_epochs
        num_batches = len(self.train_loader)
        total_steps = num_batches * config.epochs
        
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=config.scheduler_max_lr,
            total_steps=total_steps,
            pct_start=0.3,
            anneal_strategy='cos',
            cycle_momentum=False
        )
        
        # Metrics
        self.best_val_acc = 0.0
        self.best_epoch = 0
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
    
    def _measure_total_system(self, gte_model, temporal_dim, channels, device):
        """
        Measure parameters and FLOPs for the entire system (STE + GTE).
        Since STE is pre-extracted, we instantiate a dummy MobileNetV2 to measure it.
        """
        self.logger.info("SYSTEM EFFICIENCY BENCHMARK (STE + GTE)")

        # 1. Measure GTE (Head)
        gte_params = sum(p.numel() for p in gte_model.parameters())
        gte_flops = 0
        
        try:
            # Dummy input for GTE: (Batch=1, Time=10, Channels, 7, 7)
            dummy_gte = torch.randn(1, temporal_dim, channels, 7, 7).to(device)
            gte_model.eval()
            macs_gte, _ = profile(gte_model, inputs=(dummy_gte,), verbose=False)
            gte_flops = macs_gte * 2  # FLOPs approx 2 * MACs
        except Exception as e:
            self.logger.warning(f"Error measuring GTE FLOPs: {e}")

        # 2. Measure STE Backbone (MobileNetV2 - Assuming this is what used for feature extraction)
        # We create a temporary model just for measurement
        backbone = models.mobilenet_v2()
        backbone_params = sum(p.numel() for p in backbone.parameters())
        backbone_flops = 0
        
        try:
            # Dummy input for Backbone: 1 Frame (1, 3, 224, 224)
            dummy_ste = torch.randn(1, 3, 224, 224)
            macs_ste, _ = profile(backbone, inputs=(dummy_ste,), verbose=False)
                
            # IMPORTANT: The paper processes 30 frames total.
            # So backbone FLOPs = Single Frame FLOPs * 30
            backbone_flops = (macs_ste * 2) * 30 
        except Exception as e:
            self.logger.warning(f"Error measuring STE FLOPs: {e}")

        # 3. Calculate Totals
        total_params = gte_params + backbone_params
        total_flops_g = (gte_flops + backbone_flops) / 1e9  # Convert to Giga
        
        # 4. Log Results
        self.logger.info(f"1. STE Backbone (MobileNetV2 - Simulated):")
        self.logger.info(f"   - Params: {backbone_params/1e6:.2f} M")
        self.logger.info(f"   - FLOPs (30 frames): {backbone_flops/1e9:.2f} G")
        
        self.logger.info(f"2. GTE Head (Actual Model):")
        self.logger.info(f"   - Params: {gte_params/1e6:.2f} M")
        self.logger.info(f"   - FLOPs: {gte_flops/1e9:.4f} G")
        
        self.logger.info(f"   - Total Params: {total_params/1e6:.2f} M")
        self.logger.info(f"   - Total FLOPs:  {total_flops_g:.2f} G")

    def _setup_logger(self) -> logging.Logger:
        """Setup logging to file and console."""
        log_dir = Path(__file__).parent / 'logs'
        log_dir.mkdir(exist_ok=True)
        
        log_file = log_dir / f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        # File handler
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        logger.addHandler(fh)
        logger.addHandler(ch)
        
        return logger
    
    def _create_dataloader(self, split: str) -> DataLoader:
        """Create data loader for train/val/test split."""
        dataset = VideoDataLoader(
            extracted_frames_dir=self.config.extracted_frames_dir,
            split=split,
            augmentation_config=self.config.augmentation_config,
            dataset=self.config.dataset
        )
        
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=(split == 'train'),
            num_workers=self.config.num_workers
        )
    
    def train_epoch(self) -> dict:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for features_batch, labels in tqdm(self.train_loader, desc='Train', leave=False):
            # features_batch: (batch_size, T/3, C, H, W)
            features_batch = features_batch.to(self.device).float()
            labels = labels.to(self.device).long()
            
            # Zero gradients once per batch
            self.optimizer.zero_grad()
            
            batch_size = features_batch.shape[0]
            batch_logits = []
            
            # Process each sample through GTE
            for i in range(batch_size):
                features = features_batch[i]  # (T/3, C, H, W)
                logits = self.model.forward(features)  # (num_classes,)
                batch_logits.append(logits)
            
            # Stack logits: (batch_size, num_classes)
            outputs = torch.stack(batch_logits)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()  # OneCycleLR steps after each batch
            
            # Calculate metrics
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        
        return {
            'loss': total_loss / len(self.train_loader),
            'accuracy': correct / total
        }
    
    def validate(self) -> dict:
        """Validate model."""
        self.model.eval()
        correct = 0
        total = 0
        total_loss = 0.0
        
        with torch.no_grad():
            for features_batch, labels in tqdm(self.val_loader, desc='Val', leave=False):
                # features_batch: (batch_size, T/3, C, H, W)
                features_batch = features_batch.to(self.device).float()
                labels = labels.to(self.device).long()
                
                batch_size = features_batch.shape[0]
                batch_logits = []
                
                # Process each sample
                for i in range(batch_size):
                    features = features_batch[i]  # (T/3, C, H, W)
                    logits = self.model.forward(features)  # (num_classes,)
                    batch_logits.append(logits)
                
                # Stack logits: (batch_size, num_classes)
                outputs = torch.stack(batch_logits)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
        
        return {
            'loss': total_loss / len(self.val_loader),
            'accuracy': correct / total
        }
    
    def test(self) -> dict:
        """Test model on test split (if available)."""
        try:
            test_loader = self._create_dataloader('test')
        except ValueError:
            self.logger.warning("Test split not available for this dataset")
            return None
        
        self.model.eval()
        correct = 0
        total = 0
        total_loss = 0.0
        
        with torch.no_grad():
            for features_batch, labels in tqdm(test_loader, desc='Test', leave=False):
                features_batch = features_batch.to(self.device).float()
                labels = labels.to(self.device).long()
                
                batch_size = features_batch.shape[0]
                batch_logits = []
                
                for i in range(batch_size):
                    features = features_batch[i]
                    logits = self.model.forward(features)
                    batch_logits.append(logits)
                
                outputs = torch.stack(batch_logits)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
        
        return {
            'loss': total_loss / len(test_loader),
            'accuracy': correct / total
        }
    
    def train(self):
        """Train for multiple epochs with early stopping."""
        for epoch in range(self.config.epochs):
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Store metrics
            self.train_losses.append(train_metrics['loss'])
            self.val_losses.append(val_metrics['loss'])
            self.train_accs.append(train_metrics['accuracy'])
            self.val_accs.append(val_metrics['accuracy'])
            
            # Get metrics
            train_loss = train_metrics['loss']
            val_loss = val_metrics['loss']
            train_acc = train_metrics['accuracy']
            val_acc = val_metrics['accuracy']
            
            # Detect overfitting
            overfitting_gap = train_acc - val_acc
            is_overfitting = overfitting_gap > 0.15
            
            # Log
            current_lr = self.optimizer.param_groups[0]['lr']
            self.logger.info(f"Epoch {epoch+1}/{self.config.epochs} (LR: {current_lr:.2e})")
            self.logger.info(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
            self.logger.info(f"  Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
            self.logger.info(f"  Gap: {overfitting_gap:+.4f}")
            
            # Early stopping: check if validation accuracy improved
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_epoch = epoch + 1
                self._save_model('best_model.pt')
                self.logger.info(f"Saved best model (val acc: {self.best_val_acc:.4f})")
            
            self.logger.info("")
        
        self._log_summary()
        
        # Test on test split if available
        test_metrics = self.test()
        if test_metrics:
            self.logger.info(f"\nTest Results:")
            self.logger.info(f"  Test Loss: {test_metrics['loss']:.4f}")
            self.logger.info(f"  Test Accuracy: {test_metrics['accuracy']:.4f}")
    
    def _log_summary(self):
        """Log training summary."""
        self.logger.info(f"\n" + "="*60)
        self.logger.info("Training Complete")
        self.logger.info(f"Best: Epoch {self.best_epoch} | Val Acc: {self.best_val_acc:.4f}")
        self.logger.info(f"Final: Epoch {len(self.train_accs)} | Train: {self.train_accs[-1]:.4f} | Val: {self.val_accs[-1]:.4f}")
        self.logger.info("="*60)
    
    def _save_model(self, filename: str):
        """Save model checkpoint."""
        save_dir = Path(__file__).parent / 'checkpoints'
        save_dir.mkdir(exist_ok=True)
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_acc': self.best_val_acc,
        }, save_dir / filename)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Train violence detection model',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--dataset', type=str, required=True, choices=['rwf-2000', 'hockey-fight'],
                        help='Dataset to use for training (required: rwf-2000 or hockey-fight)')
    parser.add_argument('--dataset-root', type=str, default=None, 
                        help='Path to dataset root (default: auto-detect from workspace)')
    parser.add_argument('--backbone', type=str, default='mobilenet_v2',
                        choices=['mobilenet_v2', 'mobilenet_v3_small', 'mobilenet_v3_large', 'efficientnet_b0', 'mnasnet'],
                        help='STE backbone for feature extraction (default: mobilenet_v2)')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs (default: 100)')
    parser.add_argument('--batch-size', type=int, default=2, help='Batch size (default: 2)')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate (default: 1e-3)')
    parser.add_argument('--device', type=str, default=None, help='Device: cpu/cuda (default: auto)')
    parser.add_argument('--num-workers', type=int, default=2, help='Number of DataLoader workers (default: 2)')
    
    args = parser.parse_args()
    
    # Auto-detect dataset root if not provided
    if args.dataset_root:
        dataset_root = Path(args.dataset_root).resolve()
    else:
        # Try to find dataset root: look for 'dataset' directory in parent directories
        current_dir = Path(__file__).parent
        dataset_root = None
        
        # Check common locations relative to this script
        for candidate in [
            current_dir.parent.parent.parent / 'dataset',  # violence-detection/dataset
            Path.cwd() / 'dataset',  # Current working dir / dataset
            Path.cwd().parent / 'dataset',  # Parent dir / dataset
        ]:
            if candidate.exists():
                dataset_root = candidate.resolve()
                break
        
        if not dataset_root:
            print(f"ERROR: Could not auto-detect dataset root. Please specify --dataset-root")
            sys.exit(1)
    
    if not dataset_root.exists():
        print(f"ERROR: Dataset root not found: {dataset_root}")
        sys.exit(1)
    
    # Determine extracted frames directory based on dataset
    if args.dataset == 'rwf-2000':
        extracted_frames_dir = dataset_root / 'RWF-2000' / 'extracted_frames'
    elif args.dataset == 'hockey-fight':
        extracted_frames_dir = dataset_root / 'HockeyFight' / 'extracted_frames'
    else:
        print(f"ERROR: Unknown dataset: {args.dataset}")
        sys.exit(1)
    
    if not extracted_frames_dir.exists():
        print(f"ERROR: Extracted frames not found. Run: python frame_extractor.py --dataset {args.dataset}")
        sys.exit(1)
    
    # Create config
    config = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=args.device or ('cuda' if torch.cuda.is_available() else 'cpu'),
        extracted_frames_dir=str(extracted_frames_dir),
        num_workers=args.num_workers,
        dataset=args.dataset,
        backbone=args.backbone
    )
    
    # Start training
    trainer = Trainer(config)
    trainer.train()

if __name__ == '__main__':
    main()