import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)
from mlflow.models.signature import infer_signature
from data_preparation import load_and_prepare_data
from model_export import export_best_model

def create_confusion_matrix_plot(y_test, y_pred, model_name):
    """Create and save confusion matrix plot"""
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
               xticklabels=['Rejected', 'Approved'], 
               yticklabels=['Rejected', 'Approved'])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    confusion_matrix_path = "confusion_matrix.png"
    plt.savefig(confusion_matrix_path, dpi=300, bbox_inches='tight')
    mlflow.log_artifact(confusion_matrix_path)
    plt.close()

def create_roc_curve_plot(y_test, y_pred_proba, test_roc_auc):
    """Create and save ROC curve plot"""
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

def create_feature_importance_plot(model, feature_names=None):
    """Create and save feature importance plot"""
    if hasattr(model, 'feature_importances_'):
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(len(model.feature_importances_))]
        
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Create feature importance plot
        plt.figure(figsize=(12, 8))
        top_features = feature_importance.head(15)  # Reduced from 20 to 15
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Feature Importance')
        plt.title('Top 15 Feature Importances - Random Forest')
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

def hyperparameter_tuning_with_manual_logging():
    """
    Perform lightweight hyperparameter tuning with manual MLflow logging
    """
    # DISABLE autolog for manual logging
    mlflow.sklearn.autolog(disable=True)
    
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
    
   
    rf_param_grid = {
        'n_estimators': [50, 100],             
        'max_depth': [10, 20],                 
        'min_samples_split': [2, 5],           
        'min_samples_leaf': [1, 2],             
        'max_features': ['sqrt']                
    }
    
    print(f"Total combinations: {np.prod([len(v) for v in rf_param_grid.values()])} (lightweight)")
    
    # Start MLflow run for hyperparameter tuning
    with mlflow.start_run(run_name="RandomForest_Lightweight_Tuning"):
        print("Starting lightweight hyperparameter tuning...")
        
        # MANUAL LOGGING - Log hyperparameter grid
        mlflow.log_param("model_type", "RandomForestClassifier")
        mlflow.log_param("cv_folds", 3)                    # Reduced from 5 to 3
        mlflow.log_param("scoring_metric", "accuracy")
        mlflow.log_param("random_state", 42)
        mlflow.log_param("tuning_type", "lightweight")
        
        # Log parameter grid
        for param_name, param_values in rf_param_grid.items():
            mlflow.log_param(f"param_grid_{param_name}", str(param_values))
        
        # Create base model
        base_model = RandomForestClassifier(random_state=42, n_jobs=-1)
        
        # Perform Grid Search with reduced CV folds
        print(f"Running GridSearchCV with {np.prod([len(v) for v in rf_param_grid.values()])} combinations...")
        grid_search = GridSearchCV(
            base_model, 
            rf_param_grid, 
            cv=3,           # Reduced from 5 to 3 folds
            scoring='accuracy', 
            n_jobs=-1, 
            verbose=1,
            return_train_score=True
        )
        
        # Fit grid search
        print("Tuning started... (estimated time: 2-3 minutes)")
        grid_search.fit(X_train_scaled, y_train)
        print("Tuning completed!")
        
        # Get the best model
        best_model = grid_search.best_estimator_
        
        # MANUAL LOGGING - Log best parameters
        for param_name, param_value in grid_search.best_params_.items():
            mlflow.log_param(f"best_{param_name}", param_value)
        
        # MANUAL LOGGING - Log cross-validation score
        mlflow.log_metric("best_cv_score", grid_search.best_score_)
        mlflow.log_metric("mean_train_score", grid_search.cv_results_['mean_train_score'][grid_search.best_index_])
        mlflow.log_metric("std_train_score", grid_search.cv_results_['std_train_score'][grid_search.best_index_])
        mlflow.log_metric("mean_test_score", grid_search.cv_results_['mean_test_score'][grid_search.best_index_])
        mlflow.log_metric("std_test_score", grid_search.cv_results_['std_test_score'][grid_search.best_index_])
        
        # Make predictions with best model
        y_pred = best_model.predict(X_test_scaled)
        y_pred_proba = best_model.predict_proba(X_test_scaled)[:, 1]
        
        # Calculate test metrics
        test_accuracy = accuracy_score(y_test, y_pred)
        test_precision = precision_score(y_test, y_pred)
        test_recall = recall_score(y_test, y_pred)
        test_f1 = f1_score(y_test, y_pred)
        test_roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        # MANUAL LOGGING - Log test metrics
        mlflow.log_metric("test_accuracy", test_accuracy)
        mlflow.log_metric("test_precision", test_precision)
        mlflow.log_metric("test_recall", test_recall)
        mlflow.log_metric("test_f1_score", test_f1)
        mlflow.log_metric("test_roc_auc", test_roc_auc)
        
        # Create and log plots
        print("Creating and logging artifacts...")
        
        # Confusion Matrix
        create_confusion_matrix_plot(y_test, y_pred, "Random Forest Lightweight")
        
        # ROC Curve
        create_roc_curve_plot(y_test, y_pred_proba, test_roc_auc)
        
        # Feature Importance
        create_feature_importance_plot(best_model)
        
        # Classification Report
        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        report_csv_path = "classification_report.csv"
        report_df.to_csv(report_csv_path)
        mlflow.log_artifact(report_csv_path)
        
        # MANUAL LOGGING - Log the trained model
        signature = infer_signature(X_train_scaled, best_model.predict(X_train_scaled))
        mlflow.sklearn.log_model(
            best_model, 
            "best_lightweight_model",
            signature=signature,
            input_example=X_train_scaled[:3]  # Reduced sample size
        )
        
        # Log additional model info
        mlflow.log_param("n_features", X_train_scaled.shape[1])
        mlflow.log_param("training_samples", X_train_scaled.shape[0])
        mlflow.log_param("test_samples", X_test_scaled.shape[0])
        mlflow.log_param("total_combinations_tested", len(grid_search.cv_results_['params']))
        
        # Display results
        print(f"\nLightweight Hyperparameter Tuning Results:")
        print(f"Best Parameters: {grid_search.best_params_}")
        print(f"Best CV Score: {grid_search.best_score_:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Test Precision: {test_precision:.4f}")
        print(f"Test Recall: {test_recall:.4f}")
        print(f"Test F1 Score: {test_f1:.4f}")
        print(f"Test ROC AUC: {test_roc_auc:.4f}")
        print(f"Total combinations tested: {len(grid_search.cv_results_['params'])}")
        
        # Get current run info
        current_run = mlflow.active_run()
        run_id = current_run.info.run_id
        print(f"Run ID: {run_id}")
        
        # Export the best model
        print("\nExporting the best lightweight model...")
        metrics = {
            'test_accuracy': test_accuracy,
            'test_precision': test_precision,
            'test_recall': test_recall,
            'test_f1_score': test_f1,
            'test_roc_auc': test_roc_auc,
            'best_cv_score': grid_search.best_score_
        }
        
        export_path = export_best_model(best_model, run_id, "RandomForest_Lightweight", metrics)
        print(f"Model exported to: {export_path}")
        
        return best_model, run_id

if __name__ == "__main__":
    import time
    start_time = time.time()
    
    hyperparameter_tuning_with_manual_logging()
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"\nhyperparameter tuning completed!")
    print(f"Total execution time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print("To view the experiments in the MLflow UI, run: mlflow ui")