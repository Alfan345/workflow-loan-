# RandomForest_Lightweight Model Export

This directory contains the exported RandomForest_Lightweight model and related files.

## Files

- `RandomForest_Lightweight_model.pkl`: Pickled model file
- `RandomForest_Lightweight_model.joblib`: Joblib model file (recommended)
- `model_metadata.json`: Model metadata in JSON format
- `model_info.txt`: Human-readable model information
- `feature_importance.csv`: Feature importance scores
- `README.md`: This file

## Quick Start

```python
import joblib
import pandas as pd

# Load the model
model = joblib.load('RandomForest_Lightweight_model.joblib')

# Make predictions
# predictions = model.predict(your_data)
# probabilities = model.predict_proba(your_data)
```

## Model Performance

- **test_accuracy**: 0.9288
- **test_precision**: 0.8941
- **test_recall**: 0.7726
- **test_f1_score**: 0.8289
- **test_roc_auc**: 0.9742
- **best_cv_score**: 0.9262
