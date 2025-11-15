"""
Training script for RWF-2000 violence detection using SME+STE features.

Usage:
    python train.py --dataset-root <path> [OPTIONS]

Arguments:
    --dataset-root PATH         Path to dataset root (required)
    --epochs EPOCHS             Number of epochs (default: 20)
    --batch-size BATCH_SIZE     Batch size (default: 32)
    --lr LEARNING_RATE          Learning rate (default: 0.001)
    --device DEVICE             Device to use: cpu/cuda (default: cuda if available)

Example:
    python train.py --dataset-root d:/DATN/violence-detection/dataset --epochs 20 --batch-size 32
"""

import argparse
import sys
import logging
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# Import VideoDataLoader trực tiếp để hỗ trợ multiprocessing trên Windows
from data_loader import VideoDataLoader
from remonet.gte.extractor import GTEExtractor


@dataclass
class TrainConfig:
    """Training configuration."""
    epochs: int = 100
    batch_size: int = 16
    learning_rate: float = 1e-3
    weight_decay: float = 1e-2
    adam_epsilon: float = 1e-9
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    extracted_frames_dir: str = None
    num_workers: int = 2
    
    # OneCycle LR Scheduler config
    scheduler_min_lr: float = 1e-8
    scheduler_patience: int = 2
    scheduler_factor: float = 0.5


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
        
        # Create data loaders
        self.train_loader = self._create_dataloader('train')
        self.val_loader = self._create_dataloader('val')
        
        self.logger.info(f"Train samples: {len(self.train_loader.dataset)}")
        self.logger.info(f"Val samples: {len(self.val_loader.dataset)}\n")
        
        # Use default feature shape from STE (MobileNetV2)
        num_channels = 1280
        temporal_dim = 10  # 30 frames / 3 = 10
        self.logger.info(f"Expected STE feature shape: ({temporal_dim}, {num_channels}, H, W)\n")
        
        
        # Initialize GTE model
        self.model = GTEExtractor(
            num_channels=num_channels,
            temporal_dim=temporal_dim,
            num_classes=2,
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
        
        # Learning rate scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=config.scheduler_factor,
            patience=config.scheduler_patience,
            min_lr=config.scheduler_min_lr
        )
        
        # Metrics
        self.best_val_acc = 0.0
        self.best_epoch = 0
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
    
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
        """Create data loader for train/val split."""
        dataset = VideoDataLoader(
            extracted_frames_dir=self.config.extracted_frames_dir,
            split=split
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
    
    def train(self):
        """Train for multiple epochs."""
        self.logger.info("="*60)
        self.logger.info("Starting Training")
        self.logger.info("="*60 + "\n")
        
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
            
            # Update learning rate
            old_lr = self.optimizer.param_groups[0]['lr']
            self.scheduler.step(val_metrics['accuracy'])
            new_lr = self.optimizer.param_groups[0]['lr']
            
            # Get metrics
            train_loss = train_metrics['loss']
            val_loss = val_metrics['loss']
            train_acc = train_metrics['accuracy']
            val_acc = val_metrics['accuracy']
            
            # Detect overfitting
            overfitting_gap = train_acc - val_acc
            is_overfitting = overfitting_gap > 0.15
            
            # Log
            self.logger.info(f"Epoch {epoch+1}/{self.config.epochs}")
            self.logger.info(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
            self.logger.info(f"  Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
            
            if old_lr != new_lr:
                self.logger.info(f"  LR adjusted: {old_lr:.2e} → {new_lr:.2e}")
            
            # Check overfitting
            if is_overfitting:
                self.logger.warning(f"OVERFITTING DETECTED! Gap: {overfitting_gap:.4f}")
            
            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_epoch = epoch + 1
                self._save_model('best_model.pt')
                self.logger.info(f"Saved best model (val acc: {self.best_val_acc:.4f})")
            
            # Early stopping
            if epoch > 0:
                prev_best = max(self.val_accs[:-1]) if len(self.val_accs) > 1 else 0
                if val_acc <= prev_best and epoch - self.best_epoch > 10:
                    self.logger.warning(f"No improvement for 10 epochs")
            
            self.logger.info("")
        
        self._log_summary()
    
    def _log_summary(self):
        """Log training summary."""
        self.logger.info("Training Summary")
        self.logger.info(f"Best epoch: {self.best_epoch}")
        self.logger.info(f"Best validation accuracy: {self.best_val_acc:.4f}")
        
        # Best val accuracy epoch
        best_val_idx = self.val_accs.index(max(self.val_accs))
        self.logger.info(f"  Train acc at best epoch: {self.train_accs[best_val_idx]:.4f}")
        self.logger.info(f"  Train loss at best epoch: {self.train_losses[best_val_idx]:.4f}")
        self.logger.info(f"  Val loss at best epoch: {self.val_losses[best_val_idx]:.4f}")
        
        # Final metrics
        self.logger.info(f"\nFinal epoch ({len(self.train_accs)}):")
        self.logger.info(f"  Train acc: {self.train_accs[-1]:.4f}")
        self.logger.info(f"  Val acc: {self.val_accs[-1]:.4f}")
        self.logger.info(f"  Train loss: {self.train_losses[-1]:.4f}")
        self.logger.info(f"  Val loss: {self.val_losses[-1]:.4f}")
        
        # Check if model improved significantly
        max_val_acc = max(self.val_accs)
        if max_val_acc < 0.6:
            self.logger.warning("Model accuracy is low (<60%)")
    
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
    
    parser.add_argument('--dataset-root', type=str, required=True, help='Path to dataset root')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs (default: 100)')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size (default: 16)')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate (default: 1e-3)')
    parser.add_argument('--device', type=str, default=None, help='Device: cpu/cuda (default: auto)')
    parser.add_argument('--num-workers', type=int, default=2, help='Number of DataLoader workers (default: 2)')
    
    args = parser.parse_args()
    
    # Validate dataset root
    dataset_root = Path(args.dataset_root).resolve()
    if not dataset_root.exists():
        print(f"ERROR: Dataset root not found: {dataset_root}")
        sys.exit(1)
    
    extracted_frames_dir = dataset_root / 'RWF-2000' / 'extracted_frames'
    if not extracted_frames_dir.exists():
        print(f"ERROR: Extracted frames directory not found: {extracted_frames_dir}")
        print(f"Run frame_extractor.py first")
        sys.exit(1)
    
    # Create config
    config = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=args.device or ('cuda' if torch.cuda.is_available() else 'cpu'),
        extracted_frames_dir=str(extracted_frames_dir),
        num_workers=args.num_workers
    )
    
    # Start training
    trainer = Trainer(config)
    trainer.train()
    
    trainer.logger.info(f"Best validation accuracy: {trainer.best_val_acc:.4f}")
    trainer.logger.info(f"Best model saved to: checkpoints/best_model.pt")

if __name__ == '__main__':
    main()