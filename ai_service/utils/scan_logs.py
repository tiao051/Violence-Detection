import os
import re

# Update path to point to logs directory relative to project root or absolute
log_dir = r'd:\doantotnghiep\Violence-Detection\ai_service\training\two-stage\logs'
best_acc = 0.0
best_log = ""

for filename in os.listdir(log_dir):
    if filename.endswith(".log"):
        filepath = os.path.join(log_dir, filename)
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                
            # Check if it's RWF-2000
            if "RWF-2000" in content or "Train samples: 1600" in content:
                # Find max validation accuracy
                val_accs = re.findall(r'val acc: (0\.\d+)', content)
                if val_accs:
                    max_acc = max([float(x) for x in val_accs])
                    if max_acc > best_acc:
                        best_acc = max_acc
                        best_log = filename
        except:
            continue

print(f"Best RWF-2000 Log: {best_log}")
print(f"Best Validation Accuracy: {best_acc}")
