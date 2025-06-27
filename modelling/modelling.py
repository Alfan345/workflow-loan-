import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from data_preparation import load_and_prepare_data

if __name__ == "__main__":
    # Enable MLflow autolog for sklearn - NO OPTIONS
    mlflow.sklearn.autolog()
    
    # Set up the MLflow tracking URI (local)
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("Loan Approval Classification")
    
    # Load and prepare data
    print("Loading and preparing data...")
    X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled = load_and_prepare_data()
    
    # Convert pandas Series to numpy arrays if needed
    if hasattr(y_train, 'values'):
        y_train = y_train.values
    if hasattr(y_test, 'values'):
        y_test = y_test.values
    
    # Start MLflow run for basic model training
    with mlflow.start_run(run_name="RandomForest_Basic_Training"):
        print("Training basic Random Forest model...")
        
        # Create basic Random Forest model with default parameters
        rf_model = RandomForestClassifier(random_state=42)
        
        # Train the model - autolog will automatically log this
        rf_model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = rf_model.predict(X_test_scaled)
        y_pred_proba = rf_model.predict_proba(X_test_scaled)[:, 1]
        
        # Calculate metrics
        test_accuracy = accuracy_score(y_test, y_pred)
        test_precision = precision_score(y_test, y_pred)
        test_recall = recall_score(y_test, y_pred)
        test_f1 = f1_score(y_test, y_pred)
        test_roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        # Display results
        print("\nBasic Random Forest Model Results:")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Test Precision: {test_precision:.4f}")
        print(f"Test Recall: {test_recall:.4f}")
        print(f"Test F1 Score: {test_f1:.4f}")
        print(f"Test ROC AUC: {test_roc_auc:.4f}")
        
        # Get current run info
        current_run = mlflow.active_run()
        print(f"Run ID: {current_run.info.run_id}")
    
    print("\nBasic model training completed!")
    print("To view the experiments in the MLflow UI, run: mlflow ui")