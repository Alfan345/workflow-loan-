import os
import pickle
import json
import joblib
import pandas as pd
from datetime import datetime
import mlflow
import mlflow.sklearn

def export_best_model(model, run_id, model_name, metrics_dict, export_dir="../models"):
    
    # Create export directory if it doesn't exist
    if not os.path.exists(export_dir):
        os.makedirs(export_dir)
    
    # Create timestamp for unique folder naming
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_folder = f"{model_name}_{timestamp}"
    model_path = os.path.join(export_dir, model_folder)
    
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    
    # 1. Save model using pickle
    pickle_path = os.path.join(model_path, f"{model_name}_model.pkl")
    with open(pickle_path, 'wb') as f:
        pickle.dump(model, f)
    
    # 2. Save model using joblib (alternative serialization)
    joblib_path = os.path.join(model_path, f"{model_name}_model.joblib")
    joblib.dump(model, joblib_path)
    
    # 3. Save model metadata
    metadata = {
        "model_name": model_name,
        "run_id": run_id,
        "export_timestamp": timestamp,
        "export_date": datetime.now().isoformat(),
        "model_parameters": model.get_params(),
        "metrics": metrics_dict,
        "model_type": str(type(model).__name__),
        "sklearn_version": None,  # You can add sklearn.__version__ if needed
        "files": {
            "pickle_model": f"{model_name}_model.pkl",
            "joblib_model": f"{model_name}_model.joblib",
            "metadata": "model_metadata.json",
            "model_info": "model_info.txt"
        }
    }
    
    metadata_path = os.path.join(model_path, "model_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    
    # 4. Save human-readable model information
    info_path = os.path.join(model_path, "model_info.txt")
    with open(info_path, 'w') as f:
        f.write(f"Model Export Information\n")
        f.write(f"========================\n\n")
        f.write(f"Model Name: {model_name}\n")
        f.write(f"Model Type: {type(model).__name__}\n")
        f.write(f"MLflow Run ID: {run_id}\n")
        f.write(f"Export Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write(f"Model Performance Metrics:\n")
        f.write(f"-------------------------\n")
        for metric, value in metrics_dict.items():
            if isinstance(value, float):
                f.write(f"{metric}: {value:.4f}\n")
            else:
                f.write(f"{metric}: {value}\n")
        
        f.write(f"\nModel Parameters:\n")
        f.write(f"----------------\n")
        for param, value in model.get_params().items():
            f.write(f"{param}: {value}\n")
        
        f.write(f"\nFiles in this export:\n")
        f.write(f"--------------------\n")
        f.write(f"- {model_name}_model.pkl: Pickled model (use with pickle.load())\n")
        f.write(f"- {model_name}_model.joblib: Joblib model (use with joblib.load())\n")
        f.write(f"- model_metadata.json: Machine-readable metadata\n")
        f.write(f"- model_info.txt: Human-readable information\n")
        
        f.write(f"\nUsage Example:\n")
        f.write(f"-------------\n")
        f.write(f"import pickle\n")
        f.write(f"import joblib\n\n")
        f.write(f"# Load using pickle\n")
        f.write(f"with open('{model_name}_model.pkl', 'rb') as f:\n")
        f.write(f"    model = pickle.load(f)\n\n")
        f.write(f"# Load using joblib (recommended)\n")
        f.write(f"model = joblib.load('{model_name}_model.joblib')\n\n")
        f.write(f"# Make predictions\n")
        f.write(f"predictions = model.predict(X_test)\n")
        f.write(f"probabilities = model.predict_proba(X_test)\n")
    
    # 5. Save feature importance if available (for tree-based models)
    if hasattr(model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'feature_index': range(len(model.feature_importances_)),
            'feature_name': [f'feature_{i}' for i in range(len(model.feature_importances_))],
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        feature_importance_path = os.path.join(model_path, "feature_importance.csv")
        feature_importance.to_csv(feature_importance_path, index=False)
    
    # 6. Create a simple README file
    readme_path = os.path.join(model_path, "README.md")
    with open(readme_path, 'w') as f:
        f.write(f"# {model_name} Model Export\n\n")
        f.write(f"This directory contains the exported {model_name} model and related files.\n\n")
        f.write(f"## Files\n\n")
        f.write(f"- `{model_name}_model.pkl`: Pickled model file\n")
        f.write(f"- `{model_name}_model.joblib`: Joblib model file (recommended)\n")
        f.write(f"- `model_metadata.json`: Model metadata in JSON format\n")
        f.write(f"- `model_info.txt`: Human-readable model information\n")
        if hasattr(model, 'feature_importances_'):
            f.write(f"- `feature_importance.csv`: Feature importance scores\n")
        f.write(f"- `README.md`: This file\n\n")
        f.write(f"## Quick Start\n\n")
        f.write(f"```python\n")
        f.write(f"import joblib\n")
        f.write(f"import pandas as pd\n\n")
        f.write(f"# Load the model\n")
        f.write(f"model = joblib.load('{model_name}_model.joblib')\n\n")
        f.write(f"# Make predictions\n")
        f.write(f"# predictions = model.predict(your_data)\n")
        f.write(f"# probabilities = model.predict_proba(your_data)\n")
        f.write(f"```\n\n")
        f.write(f"## Model Performance\n\n")
        for metric, value in metrics_dict.items():
            if isinstance(value, float):
                f.write(f"- **{metric}**: {value:.4f}\n")
            else:
                f.write(f"- **{metric}**: {value}\n")
    
    print(f"Model successfully exported to: {model_path}")
    print(f"Files created:")
    for file in os.listdir(model_path):
        print(f"  - {file}")
    
    return model_path

def load_exported_model(model_path, use_joblib=True):
    """
    Load an exported model from the specified path
    
    Args:
        model_path: Path to the exported model directory
        use_joblib: Whether to use joblib (True) or pickle (False)
        
    Returns:
        Loaded model object and metadata
    """
    # Load metadata
    metadata_path = os.path.join(model_path, "model_metadata.json")
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Load model
    if use_joblib:
        model_file = os.path.join(model_path, metadata['files']['joblib_model'])
        model = joblib.load(model_file)
    else:
        model_file = os.path.join(model_path, metadata['files']['pickle_model'])
        with open(model_file, 'rb') as f:
            model = pickle.load(f)
    
    return model, metadata