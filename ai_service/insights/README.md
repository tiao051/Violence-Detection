# Violence Insights Module

Machine Learning module for analyzing and predicting patterns from violence detection data.

## Quick Start

```python
from ai_service.insights import InsightsModel

# Train model
model = InsightsModel()
model.fit(events)  # events: List[ViolenceEvent]

# Or train with mock data (for testing)
model.fit_from_mock(n_events=20000, days=90)

# Get insights
patterns = model.get_patterns()           # K-means clusters
rules = model.get_rules()                 # Association rules
prediction = model.predict(hour=20, day="Saturday", camera="Ngã ba Âu Cơ")

# Save model (no need to retrain)
model.save("trained_model.pkl")

# Load and use immediately
model = InsightsModel.load("trained_model.pkl")
```

## ML Models

### 1. K-means Clustering (ClusterAnalyzer)

- **Algorithm**: K-means (scikit-learn)
- **Purpose**: Automatically group similar events into patterns
- **Output**: "evening/night hours, mostly weekends, near Ngã ba Âu Cơ, HIGH severity"

### 2. FP-Growth Association Rules (AssociationRuleAnalyzer)

- **Algorithm**: FP-Growth (mlxtend)
- **Purpose**: Discover IF-THEN rules from event data
- **Output**: "IF Saturday AND Evening -> severity_High (confidence: 75%)"

### 3. Random Forest Prediction (RiskPredictor)

- **Algorithm**: Random Forest Classifier (scikit-learn)
- **Purpose**: Predict risk level for specific conditions
- **Output**: "Saturday 20:00 at Ngã ba Âu Cơ -> High risk (+27% vs average)"

## Module Structure

```
insights/
├── __init__.py           # Exports
├── insights_model.py     # Unified model (main entry point)
├── README.md             # This file
│
├── data/                 # Data schema & generation
│   ├── event_schema.py   # ViolenceEvent class
│   ├── mock_generator.py # Mock data generator
│   └── analytics_events.csv  # Generated dataset
│
├── models/               # ML algorithms
│   ├── cluster_analyzer.py      # K-means
│   ├── association_analyzer.py  # FP-Growth
│   └── risk_predictor.py        # Random Forest
│
├── demo.py                       # Demo script
├── train_model.py                # Training script
├── generate_analytics_dataset.py # CSV generator
├── hdfs_upload.py                # HDFS upload script
└── kafka_analytics_producer.py   # Kafka producer
```

## Camera Locations (Quận Tân Phú)

| Camera ID | Name                | Description                       |
| --------- | ------------------- | --------------------------------- |
| cam1      | Ngã tư Lê Trọng Tấn | Lê Trọng Tấn giao Tân Kỳ Tân Quý  |
| cam2      | Ngã tư Cộng Hòa     | Tân Kỳ Tân Quý giao Cộng Hòa      |
| cam3      | Ngã ba Âu Cơ        | Lũy Bán Bích giao Âu Cơ (Hotspot) |
| cam4      | Ngã tư Hòa Bình     | Hòa Bình giao Lạc Long Quân       |
| cam5      | Ngã tư Tân Sơn Nhì  | Tân Sơn Nhì giao Tây Thạnh        |

## Dependencies

```
numpy
pandas
scikit-learn
mlxtend
joblib
```

## Demo

```bash
python ai_service/insights/demo.py
```
