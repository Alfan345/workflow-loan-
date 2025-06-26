name: Loan Approval Model Training and Deployment

on:
  push:
    branches: [ main, develop ]
    paths: 
      - 'modelling/**'
      - 'preprocessed_loan_data.csv'
  pull_request:
    branches: [ main ]
    paths:
      - 'modelling/**'
      - 'preprocessed_loan_data.csv'
  workflow_dispatch:
    inputs:
      run_hyperparameter_tuning:
        description: 'Run hyperparameter tuning'
        required: false
        default: 'true'
        type: boolean
      export_model:
        description: 'Export trained model'
        required: false
        default: 'true'
        type: boolean

env:
  MLFLOW_TRACKING_URI: file:./mlruns
  MLFLOW_EXPERIMENT_NAME: "Loan Approval Classification"

jobs:
  setup:
    runs-on: ubuntu-latest
    outputs:
      python-version: ${{ steps.setup-python.outputs.python-version }}
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      id: setup-python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'
    
    - name: Cache pip dependencies
      uses: actions/cache@v3
      with:
        path: ~/.cache/pip
        key: ${{ runner.os }}-pip-${{ hashFiles('**/requirements.txt') }}
        restore-keys: |
          ${{ runner.os }}-pip-

  data-validation:
    needs: setup
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ needs.setup.outputs.python-version }}
    
    - name: Install dependencies
      run: |
        cd modelling
        pip install -r requirements.txt
    
    - name: Validate data file exists
      run: |
        if [ ! -f "preprocessed_loan_data.csv" ]; then
          echo "Error: preprocessed_loan_data.csv not found"
          exit 1
        fi
        echo "Data file found: preprocessed_loan_data.csv"
    
    - name: Test data preparation
      run: |
        cd modelling
        python -c "
        from data_preparation import load_and_prepare_data
        try:
            X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled = load_and_prepare_data()
            print(f'Data loaded successfully:')
            print(f'Training set shape: {X_train.shape}')
            print(f'Test set shape: {X_test.shape}')
            print(f'Training labels shape: {y_train.shape}')
            print(f'Test labels shape: {y_test.shape}')
        except Exception as e:
            print(f'Error loading data: {e}')
            exit(1)
        "

  model-training:
    needs: [setup, data-validation]
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ needs.setup.outputs.python-version }}
    
    - name: Install dependencies
      run: |
        cd modelling
        pip install -r requirements.txt
    
    - name: Create MLflow directory
      run: |
        mkdir -p modelling/mlruns
        mkdir -p models
    
    - name: Run model training
      run: |
        cd modelling
        python modelling.py
      env:
        PYTHONPATH: ${{ github.workspace }}/modelling
    
    - name: Upload MLflow artifacts
      uses: actions/upload-artifact@v3
      with:
        name: mlflow-artifacts
        path: modelling/mlruns/
        retention-days: 30
    
    - name: Upload trained models
      uses: actions/upload-artifact@v3
      with:
        name: trained-models
        path: models/
        retention-days: 90

  hyperparameter-tuning:
    needs: [setup, data-validation]
    runs-on: ubuntu-latest
    if: ${{ github.event.inputs.run_hyperparameter_tuning != 'false' }}
    strategy:
      matrix:
        n_estimators: ["50,100", "100,200", "200,300"]
        max_depth: ["5,10", "10,20", "20,30"]
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ needs.setup.outputs.python-version }}
    
    - name: Install dependencies
      run: |
        cd modelling
        pip install -r requirements.txt
    
    - name: Create MLflow directory
      run: mkdir -p modelling/mlruns
    
    - name: Run hyperparameter tuning
      run: |
        cd modelling
        python -c "
        from data_preparation import load_and_prepare_data
        from hyperparameter_tuning import hyperparameter_tuning
        import mlflow
        
        # Setup MLflow
        mlflow.set_tracking_uri('file:./mlruns')
        mlflow.set_experiment('Loan Approval Classification - HP Tuning')
        mlflow.sklearn.autolog()
        
        # Load data
        X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled = load_and_prepare_data()
        
        # Define parameter grid
        param_grid = {
            'n_estimators': [int(x) for x in '${{ matrix.n_estimators }}'.split(',')],
            'max_depth': [int(x) for x in '${{ matrix.max_depth }}'.split(',')],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2],
            'max_features': ['sqrt']
        }
        
        # Run tuning
        best_model, run_id = hyperparameter_tuning('RandomForest', param_grid, 
                                                   X_train_scaled, X_test_scaled, y_train, y_test)
        print(f'Best model run ID: {run_id}')
        "
      env:
        PYTHONPATH: ${{ github.workspace }}/modelling
    
    - name: Upload hyperparameter tuning results
      uses: actions/upload-artifact@v3
      with:
        name: hyperparameter-tuning-${{ matrix.n_estimators }}-${{ matrix.max_depth }}
        path: modelling/mlruns/
        retention-days: 30

  model-evaluation:
    needs: [model-training]
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'
    
    - name: Install dependencies
      run: |
        cd modelling
        pip install -r requirements.txt
    
    - name: Download MLflow artifacts
      uses: actions/download-artifact@v3
      with:
        name: mlflow-artifacts
        path: modelling/mlruns/
    
    - name: Generate model evaluation report
      run: |
        cd modelling
        python -c "
        import mlflow
        import pandas as pd
        from mlflow.tracking import MlflowClient
        
        # Setup MLflow client
        mlflow.set_tracking_uri('file:./mlruns')
        client = MlflowClient()
        
        # Get experiment
        experiment = client.get_experiment_by_name('Loan Approval Classification')
        if experiment:
            runs = client.search_runs(experiment.experiment_id)
            
            # Create evaluation report
            report_data = []
            for run in runs:
                metrics = run.data.metrics
                params = run.data.params
                report_data.append({
                    'run_id': run.info.run_id,
                    'run_name': run.data.tags.get('mlflow.runName', 'Unknown'),
                    'accuracy': metrics.get('test_accuracy', 0),
                    'precision': metrics.get('test_precision', 0),
                    'recall': metrics.get('test_recall', 0),
                    'f1_score': metrics.get('test_f1_score', 0),
                    'roc_auc': metrics.get('test_roc_auc', 0),
                    'n_estimators': params.get('n_estimators', 'N/A'),
                    'max_depth': params.get('max_depth', 'N/A')
                })
            
            # Save report
            df = pd.DataFrame(report_data)
            df.to_csv('model_evaluation_report.csv', index=False)
            print('Model evaluation report generated')
            print(df.to_string(index=False))
        else:
            print('No experiment found')
        "
    
    - name: Upload evaluation report
      uses: actions/upload-artifact@v3
      with:
        name: model-evaluation-report
        path: modelling/model_evaluation_report.csv

  deploy-model:
    needs: [model-training, model-evaluation]
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main' && github.event_name == 'push'
    steps:
    - uses: actions/checkout@v4
    
    - name: Download trained models
      uses: actions/download-artifact@v3
      with:
        name: trained-models
        path: models/
    
    - name: List model files
      run: |
        echo "Available model files:"
        find models/ -type f -name "*.pkl" -o -name "*.joblib" | head -10
    
    - name: Create model deployment package
      run: |
        # Create deployment directory
        mkdir -p deployment
        
        # Find the latest model directory
        LATEST_MODEL_DIR=$(find models/ -type d -name "RandomForest_*" | sort -r | head -1)
        
        if [ -n "$LATEST_MODEL_DIR" ]; then
          echo "Latest model directory: $LATEST_MODEL_DIR"
          cp -r "$LATEST_MODEL_DIR" deployment/
          
          # Create deployment script
          cat > deployment/deploy.py << 'EOF'
import joblib
import json
import os
from datetime import datetime

def load_model():
    """Load the latest trained model"""
    model_dirs = [d for d in os.listdir('.') if d.startswith('RandomForest_')]
    if not model_dirs:
        raise FileNotFoundError("No model directories found")
    
    latest_dir = sorted(model_dirs, reverse=True)[0]
    model_path = os.path.join(latest_dir, f"RandomForest_model.joblib")
    
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        print(f"Model loaded from: {model_path}")
        return model, latest_dir
    else:
        raise FileNotFoundError(f"Model file not found: {model_path}")

def predict(model, data):
    """Make predictions using the loaded model"""
    predictions = model.predict(data)
    probabilities = model.predict_proba(data)
    return predictions, probabilities

if __name__ == "__main__":
    try:
        model, model_dir = load_model()
        print(f"Model deployment ready: {model_dir}")
        print(f"Model type: {type(model).__name__}")
        print(f"Model parameters: {model.get_params()}")
    except Exception as e:
        print(f"Deployment failed: {e}")
EOF
          
          echo "Deployment package created successfully"
        else
          echo "No model directory found for deployment"
          exit 1
        fi
    
    - name: Upload deployment package
      uses: actions/upload-artifact@v3
      with:
        name: model-deployment-package
        path: deployment/
        retention-days: 365

  cleanup:
    needs: [model-training, model-evaluation, deploy-model]
    runs-on: ubuntu-latest
    if: always()
    steps:
    - name: Cleanup temporary files
      run: |
        echo "Workflow completed"
        echo "Artifacts uploaded for:"
        echo "- MLflow tracking data (30 days)"
        echo "- Trained models (90 days)"
        echo "- Model evaluation report"
        echo "- Deployment package (365 days)"
