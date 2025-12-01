"""
Comprehensive model evaluation script.

Evaluates trained models on multiple metrics:
- Classification metrics (accuracy, precision, recall, F1)
- Per-class performance
- Cross-dataset evaluation
- Performance benchmarks (latency, throughput)

Usage:
    python evaluate.py --model-path checkpoints/best_model.pt --dataset rwf-2000 --backbone mobilenet_v3_small
    python evaluate.py --model-path checkpoints/best_model.pt --dataset rwf-2000 --backbone mobilenet_v3_small --benchmark
"""

import torch
import argparse
import sys
import logging
import time
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from data_loader import VideoDataLoader, AugmentationConfig
from remonet.gte.extractor import GTEExtractor
from remonet.ste.extractor import BackboneType, BACKBONE_CONFIG


class ModelEvaluator:
    """Comprehensive model evaluation."""
    
    def __init__(self, model_path: str, dataset: str, backbone: str, device: str = 'cuda'):
        """
        Initialize evaluator.
        
        Args:
            model_path: Path to saved model checkpoint
            dataset: Dataset name ('rwf-2000' or 'hockey-fight')
            backbone: Backbone name
            device: Device to use ('cuda' or 'cpu')
        """
        self.model_path = model_path
        self.dataset = dataset
        self.backbone = backbone
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Setup logging
        self.logger = self._setup_logger()
        
        # Load model
        self.model = self._load_model()
        self.logger.info(f"Model loaded from: {model_path}")
        
        # Create data loaders
        self.val_loader = self._create_dataloader('val')
        self.test_loader = self._create_dataloader('test')
        
        self.logger.info(f"Val samples: {len(self.val_loader.dataset)}")
        if self.test_loader:
            self.logger.info(f"Test samples: {len(self.test_loader.dataset)}")
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logging."""
        log_dir = Path(__file__).parent / 'logs'
        log_dir.mkdir(exist_ok=True)
        
        log_file = log_dir / f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        logger.addHandler(fh)
        logger.addHandler(ch)
        
        return logger
    
    def _load_model(self) -> GTEExtractor:
        """Load trained model."""
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        # Get backbone config
        backbone_config = BACKBONE_CONFIG[BackboneType(self.backbone)]
        num_channels = backbone_config['out_channels']
        temporal_dim = 10
        
        # Initialize model
        model = GTEExtractor(
            num_channels=num_channels,
            temporal_dim=temporal_dim,
            num_classes=2,
            device=str(self.device)
        )
        
        # Load checkpoint
        checkpoint = torch.load(self.model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        return model
    
    def _create_dataloader(self, split: str) -> DataLoader:
        """Create data loader."""
        # Auto-detect dataset root
        current_dir = Path(__file__).parent
        dataset_root = None
        
        for candidate in [
            current_dir.parent.parent.parent / 'dataset',
            Path.cwd() / 'dataset',
            Path.cwd().parent / 'dataset',
        ]:
            if candidate.exists():
                dataset_root = candidate.resolve()
                break
        
        if not dataset_root:
            self.logger.warning("Could not auto-detect dataset root")
            return None
        
        # Get extracted frames directory
        if self.dataset == 'rwf-2000':
            extracted_frames_dir = dataset_root / 'RWF-2000' / 'extracted_frames'
        elif self.dataset == 'hockey-fight':
            extracted_frames_dir = dataset_root / 'HockeyFight' / 'extracted_frames'
        else:
            return None
        
        if not extracted_frames_dir.exists():
            self.logger.warning(f"Extracted frames not found: {extracted_frames_dir}")
            return None
        
        try:
            dataset = VideoDataLoader(
                extracted_frames_dir=str(extracted_frames_dir),
                split=split,
                augmentation_config=AugmentationConfig(enable_augmentation=False),
                dataset=self.dataset,
                backbone=self.backbone
            )
            
            return DataLoader(
                dataset,
                batch_size=2,
                shuffle=False,
                num_workers=0
            )
        except ValueError:
            return None
    
    def evaluate(self, data_loader: DataLoader, split_name: str = 'val') -> Dict:
        """
        Evaluate model on dataset.
        
        Returns:
            Dict with all evaluation metrics
        """
        if not data_loader:
            self.logger.warning(f"No {split_name} data available")
            return None
        
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"Evaluating on {split_name.upper()} set")
        self.logger.info(f"{'='*60}\n")
        
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for features_batch, labels in tqdm(data_loader, desc=f'Evaluate {split_name}', leave=False):
                features_batch = features_batch.to(self.device).float()
                labels = labels.to(self.device).long()
                
                batch_size = features_batch.shape[0]
                batch_logits = []
                
                # Process each sample
                for i in range(batch_size):
                    features = features_batch[i]  # (T/3, C, H, W)
                    logits = self.model.forward(features)  # (num_classes,)
                    batch_logits.append(logits)
                
                # Stack logits
                outputs = torch.stack(batch_logits)
                
                # Get predictions and probabilities
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(outputs, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        
        # Calculate metrics
        metrics = self._calculate_metrics(all_preds, all_labels, all_probs)
        
        # Log results
        self._log_results(metrics, split_name)
        
        return metrics
    
    def _calculate_metrics(self, preds: np.ndarray, labels: np.ndarray, probs: np.ndarray) -> Dict:
        """Calculate all evaluation metrics."""
        metrics = {}
        
        # Overall metrics
        metrics['accuracy'] = accuracy_score(labels, preds)
        metrics['precision'] = precision_score(labels, preds, average='weighted', zero_division=0)
        metrics['recall'] = recall_score(labels, preds, average='weighted', zero_division=0)
        metrics['f1'] = f1_score(labels, preds, average='weighted', zero_division=0)
        
        # Per-class metrics
        metrics['accuracy_per_class'] = {}
        metrics['precision_per_class'] = {}
        metrics['recall_per_class'] = {}
        metrics['f1_per_class'] = {}
        
        class_names = ['Violence', 'NonViolence']
        for i, class_name in enumerate(class_names):
            class_mask = labels == i
            if class_mask.sum() > 0:
                class_preds = preds[class_mask]
                class_labels = labels[class_mask]
                
                metrics['accuracy_per_class'][class_name] = accuracy_score(class_labels, class_preds)
                metrics['precision_per_class'][class_name] = precision_score(
                    class_labels, class_preds, average='binary', zero_division=0
                ) if len(np.unique(class_labels)) > 0 else 0
                metrics['recall_per_class'][class_name] = recall_score(
                    class_labels, class_preds, average='binary', zero_division=0
                ) if len(np.unique(class_labels)) > 0 else 0
                metrics['f1_per_class'][class_name] = f1_score(
                    class_labels, class_preds, average='binary', zero_division=0
                ) if len(np.unique(class_labels)) > 0 else 0
        
        # Confusion matrix
        metrics['confusion_matrix'] = confusion_matrix(labels, preds)
        
        # ROC-AUC
        try:
            if len(np.unique(labels)) > 1:
                metrics['roc_auc'] = roc_auc_score(labels, probs[:, 1])
            else:
                metrics['roc_auc'] = None
        except:
            metrics['roc_auc'] = None
        
        return metrics
    
    def _log_results(self, metrics: Dict, split_name: str):
        """Log evaluation results."""
        self.logger.info(f"Overall Metrics ({split_name.upper()}):")
        self.logger.info(f"  Accuracy:  {metrics['accuracy']:.4f}")
        self.logger.info(f"  Precision: {metrics['precision']:.4f}")
        self.logger.info(f"  Recall:    {metrics['recall']:.4f}")
        self.logger.info(f"  F1-Score:  {metrics['f1']:.4f}")
        
        if metrics['roc_auc'] is not None:
            self.logger.info(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
        
        self.logger.info(f"\nPer-Class Metrics:")
        for class_name in ['Violence', 'NonViolence']:
            if class_name in metrics['accuracy_per_class']:
                self.logger.info(f"\n  {class_name}:")
                self.logger.info(f"    Accuracy:  {metrics['accuracy_per_class'][class_name]:.4f}")
                self.logger.info(f"    Precision: {metrics['precision_per_class'][class_name]:.4f}")
                self.logger.info(f"    Recall:    {metrics['recall_per_class'][class_name]:.4f}")
                self.logger.info(f"    F1-Score:  {metrics['f1_per_class'][class_name]:.4f}")
        
        # Confusion matrix
        cm = metrics['confusion_matrix']
        self.logger.info(f"\nConfusion Matrix:")
        self.logger.info(f"  [[TN={cm[0,0]}, FP={cm[0,1]}],")
        self.logger.info(f"   [FN={cm[1,0]}, TP={cm[1,1]}]]")
    
    def benchmark_performance(self) -> Dict:
        """Benchmark model performance (latency, throughput, memory)."""
        self.logger.info(f"\n{'='*60}")
        self.logger.info("PERFORMANCE BENCHMARK")
        self.logger.info(f"{'='*60}\n")
        
        if not self.val_loader:
            self.logger.warning("No validation data for benchmarking")
            return None
        
        benchmark_data = {
            'latencies': [],
            'throughputs': []
        }
        
        with torch.no_grad():
            for features_batch, _ in tqdm(self.val_loader, desc='Benchmark', leave=False):
                features_batch = features_batch.to(self.device).float()
                batch_size = features_batch.shape[0]
                
                # Warm-up
                for i in range(batch_size):
                    _ = self.model.forward(features_batch[i])
                
                # Benchmark
                start = time.perf_counter()
                for i in range(batch_size):
                    _ = self.model.forward(features_batch[i])
                elapsed = (time.perf_counter() - start) * 1000  # Convert to ms
                
                latency_per_sample = elapsed / batch_size
                throughput = 1000 / latency_per_sample  # samples per second
                
                benchmark_data['latencies'].append(latency_per_sample)
                benchmark_data['throughputs'].append(throughput)
        
        # Calculate statistics
        latencies = np.array(benchmark_data['latencies'])
        throughputs = np.array(benchmark_data['throughputs'])
        
        stats = {
            'avg_latency_ms': latencies.mean(),
            'std_latency_ms': latencies.std(),
            'min_latency_ms': latencies.min(),
            'max_latency_ms': latencies.max(),
            'avg_throughput': throughputs.mean(),
        }
        
        # Log results
        self.logger.info("Inference Latency:")
        self.logger.info(f"  Mean:   {stats['avg_latency_ms']:.2f} ms")
        self.logger.info(f"  Std:    {stats['std_latency_ms']:.2f} ms")
        self.logger.info(f"  Min:    {stats['min_latency_ms']:.2f} ms")
        self.logger.info(f"  Max:    {stats['max_latency_ms']:.2f} ms")
        self.logger.info(f"\nThroughput: {stats['avg_throughput']:.2f} samples/sec")
        
        return stats
    
    def run_full_evaluation(self, benchmark: bool = False):
        """Run full evaluation."""
        self.logger.info(f"\n{'#'*60}")
        self.logger.info(f"# COMPREHENSIVE MODEL EVALUATION")
        self.logger.info(f"# Dataset: {self.dataset}")
        self.logger.info(f"# Backbone: {self.backbone}")
        self.logger.info(f"# Device: {self.device}")
        self.logger.info(f"{'#'*60}\n")
        
        # Validation evaluation
        val_metrics = self.evaluate(self.val_loader, 'val')
        
        # Test evaluation
        test_metrics = self.evaluate(self.test_loader, 'test')
        
        # Benchmark
        if benchmark:
            perf_stats = self.benchmark_performance()
        
        self.logger.info(f"\n{'='*60}")
        self.logger.info("EVALUATION COMPLETE")
        self.logger.info(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate trained violence detection model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python evaluate.py --model-path checkpoints/best_model.pt --dataset rwf-2000 --backbone mobilenet_v3_small
  python evaluate.py --model-path checkpoints/best_model.pt --dataset hockey-fight --backbone mobilenet_v3_small --benchmark
        """
    )
    
    parser.add_argument('--model-path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--dataset', type=str, required=True, choices=['rwf-2000', 'hockey-fight'],
                        help='Dataset name')
    parser.add_argument('--backbone', type=str, default='mobilenet_v2',
                        choices=['mobilenet_v2', 'mobilenet_v3_small', 'mobilenet_v3_large', 'efficientnet_b0', 'mnasnet'],
                        help='Backbone architecture')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'],
                        help='Device to use')
    parser.add_argument('--benchmark', action='store_true', help='Run performance benchmark')
    
    args = parser.parse_args()
    
    # Create evaluator
    evaluator = ModelEvaluator(
        model_path=args.model_path,
        dataset=args.dataset,
        backbone=args.backbone,
        device=args.device
    )
    
    # Run evaluation
    evaluator.run_full_evaluation(benchmark=args.benchmark)


if __name__ == '__main__':
    main()
