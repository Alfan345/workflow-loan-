import mlflow
import mlflow.sklearn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from data_preparation import load_and_prepare_data
from hyperparameter_tuning import hyperparameter_tuning, get_run_metrics
from model_export import export_best_model

if __name__ == "__main__":
    # Enable MLflow autolog for sklearn with specific configuration
    mlflow.sklearn.autolog(
        log_input_examples=False,
        log_model_signatures=True,
        log_models=True,
        log_datasets=False,  # Disable dataset logging to avoid the warning
        disable=False,
        exclusive=False,
        disable_for_unsupported_versions=False,
        silent=False
    )
    
    # Set up the MLflow tracking URI (local)
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("Loan Approval Classification")
    
    # Load and prepare data
    X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled = load_and_prepare_data()
    
    # Convert pandas Series to numpy arrays if needed
    if hasattr(y_train, 'values'):
        y_train = y_train.values
    if hasattr(y_test, 'values'):
        y_test = y_test.values
    
    # Define lighter hyperparameter grid for Random Forest
    rf_param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'max_features': ['sqrt']
    }

    # Run hyperparameter tuning for Random Forest
    print("Training Random Forest model with hyperparameter tuning...")
    best_rf, rf_run_id = hyperparameter_tuning("RandomForest", rf_param_grid, 
                                               X_train_scaled, X_test_scaled, y_train, y_test)

    # Get metrics for the model
    rf_metrics = get_run_metrics(rf_run_id)

    # Display results
    print("\nRandom Forest Model Results:")
    print(f"Best CV Score: {rf_metrics.get('best_cv_score', 0):.4f}")
    print(f"Test Accuracy: {rf_metrics.get('test_accuracy', 0):.4f}")
    print(f"Run ID: {rf_run_id}")

    # Export the best model
    print("\nExporting the best model...")
    export_path = export_best_model(best_rf, rf_run_id, "RandomForest", rf_metrics)
    print(f"Model exported to: {export_path}")

    # Instructions for launching the MLflow UI
    print("\nTo view the experiments in the MLflow UI, run the following command in your terminal:")
    print("mlflow ui")