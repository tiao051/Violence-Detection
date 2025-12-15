# Violence Insights Module

ML-powered analytics for violence detection patterns.

## Quick Start

```python
from insights import InsightsModel, ViolenceEvent

# Train model
model = InsightsModel()
model.fit(events)

# Get insights
patterns = model.get_patterns()           # K-means clusters
rules = model.get_rules()                 # Association rules
prediction = model.predict(hour=20, day="Saturday", camera="Ngã ba Âu Cơ")

# Save/Load
model.save("trained_model.pkl")
model = InsightsModel.load("trained_model.pkl")
```

## ML Models

| Model                   | Algorithm     | Purpose              |
| ----------------------- | ------------- | -------------------- |
| ClusterAnalyzer         | K-means       | Group similar events |
| AssociationRuleAnalyzer | FP-Growth     | Find IF-THEN rules   |
| RiskPredictor           | Random Forest | Predict risk levels  |

## Module Structure

```
insights/
├── __init__.py           # Public API
├── README.md
├── core/                 # Core classes
│   ├── model.py         # InsightsModel
│   └── schema.py        # ViolenceEvent
├── algorithms/           # ML implementations
│   ├── cluster_analyzer.py
│   ├── association_analyzer.py
│   └── risk_predictor.py
├── data/                 # Data handling
│   ├── generator.py     # Mock data generator
│   └── analytics_events.csv
└── scripts/              # CLI tools
    ├── demo.py
    ├── train.py
    └── generate_dataset.py
```

## Camera Locations (Quận Tân Phú)

| Camera ID | Name                | Description                       |
| --------- | ------------------- | --------------------------------- |
| cam1      | Ngã tư Lê Trọng Tấn | Lê Trọng Tấn giao Tân Kỳ Tân Quý  |
| cam2      | Ngã tư Cộng Hòa     | Tân Kỳ Tân Quý giao Cộng Hòa      |
| cam3      | Ngã ba Âu Cơ        | Lũy Bán Bích giao Âu Cơ (Hotspot) |
| cam4      | Ngã tư Hòa Bình     | Hòa Bình giao Lạc Long Quân       |
| cam5      | Ngã tư Tân Sơn Nhì  | Tân Sơn Nhì giao Tây Thạnh        |

## Scripts

```bash
# Demo all models
python ai_service/insights/scripts/demo.py

# Generate CSV
python ai_service/insights/scripts/generate_dataset.py

# Train model
python ai_service/insights/scripts/train.py
```
