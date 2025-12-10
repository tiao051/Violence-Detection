import re
import matplotlib.pyplot as plt
import sys
from pathlib import Path

def parse_log(log_path):
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    epochs = []

    with open(log_path, 'r') as f:
        content = f.read()
        
    # Regex to find metrics
    # Example line: 2025-12-04 03:16:36,660 - INFO -   Train - Loss: 0.4775, Acc: 0.7460
    # Example line: 2025-12-04 03:16:36,660 - INFO -   Val   - Loss: 0.3831, Acc: 0.8360, Prec: 0.8415, Rec: 0.8280, F1: 0.8347
    
    train_pattern = re.compile(r"Train - Loss: ([\d\.]+), Acc: ([\d\.]+)")
    val_pattern = re.compile(r"Val\s+- Loss: ([\d\.]+), Acc: ([\d\.]+)")
    
    lines = content.split('\n')
    current_epoch = 0
    
    for line in lines:
        if "Epoch" in line and "/" in line:
            # Try to extract epoch number if needed, but sequential is fine
            pass
            
        train_match = train_pattern.search(line)
        if train_match:
            train_losses.append(float(train_match.group(1)))
            train_accs.append(float(train_match.group(2)))
            current_epoch += 1
            epochs.append(current_epoch)
            
        val_match = val_pattern.search(line)
        if val_match:
            val_losses.append(float(val_match.group(1)))
            val_accs.append(float(val_match.group(2)))

    return epochs, train_losses, val_losses, train_accs, val_accs

def plot_metrics(log_path, output_path):
    epochs, train_losses, val_losses, train_accs, val_accs = parse_log(log_path)
    
    # Ensure lengths match (in case log was cut off)
    min_len = min(len(epochs), len(train_losses), len(val_losses), len(train_accs), len(val_accs))
    epochs = epochs[:min_len]
    train_losses = train_losses[:min_len]
    val_losses = val_losses[:min_len]
    train_accs = train_accs[:min_len]
    val_accs = val_accs[:min_len]

    plt.figure(figsize=(12, 5))

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, label='Train Acc')
    plt.plot(epochs, val_accs, label='Val Acc')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Chart saved to {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_training_log.py <log_file>")
        sys.exit(1)
        
    log_file = sys.argv[1]
    output_file = "training_chart.png"
    plot_metrics(log_file, output_file)
