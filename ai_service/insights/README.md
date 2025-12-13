# Violence Insights Module

Machine Learning module for analyzing and predicting patterns from violence detection data.

## Quick Start

```python
from ai_service.insights import InsightsModel

# Train model
model = InsightsModel()
model.fit(events)  # events: List[ViolenceEvent]

# Or train with mock data (for testing)
model.fit_from_mock(n_events=500, days=60)

# Get insights
patterns = model.get_patterns()           # K-means clusters
rules = model.get_rules()                 # Association rules
prediction = model.predict(hour=20, day="Saturday", camera="Parking Lot")

# Save model (no need to retrain)
model.save("insights_model.pkl")

# Load and use immediately
model = InsightsModel.load("insights_model.pkl")
```

## ML Models

### 1. K-means Clustering (ClusterAnalyzer)

- **Algorithm**: K-means (scikit-learn)
- **Purpose**: Automatically group similar events into patterns
- **Output**: "evening/night hours, mostly weekends, near Parking Lot, HIGH severity"

### 2. FP-Growth Association Rules (AssociationRuleAnalyzer)

- **Algorithm**: FP-Growth (mlxtend)
- **Purpose**: Discover IF-THEN rules from event data
- **Output**: "IF Saturday AND Evening -> severity_High (confidence: 75%)"

### 3. Random Forest Prediction (RiskPredictor)

- **Algorithm**: Random Forest Classifier (scikit-learn)
- **Purpose**: Predict risk level for specific conditions
- **Output**: "Saturday 20:00 at Parking Lot -> High risk (+27% vs average)"

## Module Structure

```
insights/
├── __init__.py           # Exports
├── insights_model.py     # Unified model (main entry point)
├── README.md             # Documentation
├── demo.py               # Demo script
├── data/
│   ├── event_schema.py   # ViolenceEvent schema
│   └── mock_generator.py # Mock data generator
└── models/
    ├── cluster_analyzer.py      # K-means
    ├── association_analyzer.py  # FP-Growth
    └── risk_predictor.py        # Random Forest
```

## Dependencies

```
numpy
pandas
scikit-learn
mlxtend
joblib
```

## Usage with Real Data from Firestore

```python
from ai_service.insights import InsightsModel
from ai_service.insights.data import ViolenceEvent

# Convert Firestore documents to ViolenceEvent
events = []
for doc in firestore_docs:
    event = ViolenceEvent.from_firestore(doc.to_dict(), doc.id)
    events.append(event)

# Train model
model = InsightsModel()
model.fit(events)

# Get full report
report = model.get_full_report()
```

## Demo

```bash
python ai_service/insights/demo.py
```
