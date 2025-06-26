# RandomForest Model Export

This directory contains the exported RandomForest model and related files.

## Files

- `RandomForest_model.pkl`: Pickled model file
- `RandomForest_model.joblib`: Joblib model file (recommended)
- `model_metadata.json`: Model metadata in JSON format
- `model_info.txt`: Human-readable model information
- `feature_importance.csv`: Feature importance scores
- `README.md`: This file

## Quick Start

```python
import joblib
import pandas as pd

# Load the model
model = joblib.load('RandomForest_model.joblib')

# Make predictions
# predictions = model.predict(your_data)
# probabilities = model.predict_proba(your_data)
```

## Model Performance

- **training_f1_score**: 0.9930
- **training_log_loss**: 0.0643
- **test_precision**: 0.8964
- **training_roc_auc**: 1.0000
- **test_accuracy**: 0.9297
- **training_accuracy_score**: 0.9930
- **training_score**: 0.9930
- **training_recall_score**: 0.9930
- **test_recall**: 0.7746
- **best_cv_score**: 0.9272
- **test_roc_auc**: 0.9747
- **training_precision_score**: 0.9931
- **test_f1_score**: 0.8311
