import re

# Update path to point to logs directory relative to project root or absolute
# Assuming running from project root
log_path = r'd:\doantotnghiep\Violence-Detection\ai_service\training\two-stage\logs\train_20251129_172334.log'

try:
    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
        
    # Find all validation accuracies
    val_accs = re.findall(r'val acc: (0\.\d+)', content)
    if val_accs:
        best_val_acc = max(val_accs)
        print(f"Best Validation Accuracy: {best_val_acc}")
    else:
        print("No validation accuracy found.")

    # Find Test Accuracy
    test_acc = re.search(r'Test Accuracy: (0\.\d+)', content)
    if test_acc:
        print(f"Test Accuracy: {test_acc.group(1)}")
    else:
        print("No Test Accuracy found.")
        
except Exception as e:
    print(f"Error reading file: {e}")
