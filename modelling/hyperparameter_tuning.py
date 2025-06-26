import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.ensemble import RandomForestClassifier
from mlflow.models.signature import infer_signature

def hyperparameter_tuning(model_type, param_grid, X_train, X_test, y_train, y_test):
    """
    Perform hyperparameter tuning with MLflow autolog
    """
    with mlflow.start_run(run_name=f"{model_type}_hyperparameter_tuning"):
        
        if model_type == "RandomForest":
            base_model = RandomForestClassifier(random_state=42)
        else:
            raise ValueError(f"Only RandomForest is supported, got: {model_type}")
        
        # Perform Grid Search - autolog will automatically log this
        print(f"Starting Grid Search with {len(param_grid)} parameter combinations...")
        grid_search = GridSearchCV(
            base_model, 
            param_grid, 
            cv=5, 
            scoring='accuracy', 
            n_jobs=-1, 
            verbose=1,
            return_train_score=True
        )
        
        # GridSearchCV fit will be automatically logged by autolog
        grid_search.fit(X_train, y_train)
        
        # Get the best model
        best_model = grid_search.best_estimator_
        
        # Make predictions for additional metrics
        y_pred = best_model.predict(X_test)
        y_pred_proba = best_model.predict_proba(X_test)[:, 1]
        
        # Calculate additional metrics
        test_accuracy = accuracy_score(y_test, y_pred)
        test_precision = precision_score(y_test, y_pred)
        test_recall = recall_score(y_test, y_pred)
        test_f1 = f1_score(y_test, y_pred)
        test_roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        # Log additional metrics manually
        mlflow.log_metric("test_accuracy", test_accuracy)
        mlflow.log_metric("test_precision", test_precision)
        mlflow.log_metric("test_recall", test_recall)
        mlflow.log_metric("test_f1_score", test_f1)
        mlflow.log_metric("test_roc_auc", test_roc_auc)
        mlflow.log_metric("best_cv_score", grid_search.best_score_)
        
        # Log additional parameters
        mlflow.log_param("model_type", model_type)
        mlflow.log_param("cv_folds", 5)
        mlflow.log_param("scoring_metric", "accuracy")
        
        # Create and log confusion matrix
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Rejected', 'Approved'], 
                   yticklabels=['Rejected', 'Approved'])
        plt.title(f'Confusion Matrix - {model_type}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        confusion_matrix_path = "confusion_matrix.png"
        plt.savefig(confusion_matrix_path, dpi=300, bbox_inches='tight')
        mlflow.log_artifact(confusion_matrix_path)
        plt.close()
        
        # Create and log ROC curve
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {test_roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.tight_layout()
        
        roc_curve_path = "roc_curve.png"
        plt.savefig(roc_curve_path, dpi=300, bbox_inches='tight')
        mlflow.log_artifact(roc_curve_path)
        plt.close()
        
        # Log feature importance if available
        if hasattr(best_model, 'feature_importances_'):
            feature_names = [f'feature_{i}' for i in range(len(best_model.feature_importances_))]
            feature_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': best_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            # Create feature importance plot
            plt.figure(figsize=(12, 8))
            top_features = feature_importance.head(20)
            plt.barh(range(len(top_features)), top_features['importance'])
            plt.yticks(range(len(top_features)), top_features['feature'])
            plt.xlabel('Feature Importance')
            plt.title('Top 20 Feature Importances - Random Forest')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            
            feature_importance_path = "feature_importance.png"
            plt.savefig(feature_importance_path, dpi=300, bbox_inches='tight')
            mlflow.log_artifact(feature_importance_path)
            plt.close()
            
            # Save feature importance as CSV
            feature_importance_csv = "feature_importance.csv"
            feature_importance.to_csv(feature_importance_csv, index=False)
            mlflow.log_artifact(feature_importance_csv)
        
        # Save classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        report_csv_path = "classification_report.csv"
        report_df.to_csv(report_csv_path)
        mlflow.log_artifact(report_csv_path)
        
        # The best model will be automatically logged by autolog
        # But we can also log it manually with a specific name if needed
        signature = infer_signature(X_train, best_model.predict(X_train))
        mlflow.sklearn.log_model(
            best_model, 
            "best_model",
            signature=signature,
            input_example=X_train[:5].toarray() if hasattr(X_train, 'toarray') else X_train[:5]
        )
        
        print(f"Best {model_type} Model Results:")
        print(f"Best Parameters: {grid_search.best_params_}")
        print(f"Best CV Score: {grid_search.best_score_:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Test Precision: {test_precision:.4f}")
        print(f"Test Recall: {test_recall:.4f}")
        print(f"Test F1 Score: {test_f1:.4f}")
        print(f"Test ROC AUC: {test_roc_auc:.4f}")
        print("-" * 50)
        
        # Get current run info
        current_run = mlflow.active_run()
        return best_model, current_run.info.run_id

def get_run_metrics(run_id):
    """Get metrics from a run"""
    client = mlflow.tracking.MlflowClient()
    run = client.get_run(run_id)
    return run.data.metrics