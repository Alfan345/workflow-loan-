import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report
)
from mlflow.models.signature import infer_signature

def train_and_log_model(model_name, model, params, X_train, X_test, y_train, y_test):
    """
    Train and log a model with MLflow
    
    Args:
        model_name: Name of the model
        model: Model instance
        params: Model parameters
        X_train, X_test, y_train, y_test: Training and testing data
        
    Returns:
        model, run_id
    """
    with mlflow.start_run(run_name=model_name) as run:
        # Set model parameters
        model.set_params(**params)
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        # Calculate ROC AUC if probabilities are available
        roc_auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None
        
        # Log parameters
        mlflow.log_params(params)
        
        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        if roc_auc is not None:
            mlflow.log_metric("roc_auc_score", roc_auc)
        
        # Create and log confusion matrix figure
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.tight_layout()
        
        # Save and log the figure
        confusion_matrix_path = f"confusion_matrix_{model_name}.png"
        plt.savefig(confusion_matrix_path)
        mlflow.log_artifact(confusion_matrix_path)
        plt.close()
        
        # Generate and log classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        report_path = f"classification_report_{model_name}.csv"
        report_df.to_csv(report_path)
        mlflow.log_artifact(report_path)
        
        # Log the model
        signature = infer_signature(X_train, model.predict(X_train))
        mlflow.sklearn.log_model(model, model_name, signature=signature)
        
        print(f"Model: {model_name}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        if roc_auc is not None:
            print(f"ROC AUC: {roc_auc:.4f}")
        print("-" * 50)
        
        return model, run.info.run_id