import os
import time
import logging
import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from prometheus_client import Counter, Histogram, Gauge, generate_latest

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

prediction_count = 0
error_count = 0
prediction_latencies = []
last_predictions = []

HTTP_REQUESTS_TOTAL = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'status'])
PREDICTION_COUNT_PROMETHEUS = Counter('ml_predictions_total', 'Total ML predictions')
PREDICTION_ERRORS = Counter('ml_prediction_errors_total', 'Total prediction errors', ['error_type'])
REQUEST_DURATION = Histogram('http_request_duration_seconds', 'Request duration', ['endpoint'])
MODEL_HEALTH = Gauge('ml_model_health', 'Model health (1=healthy, 0=unhealthy)')

model = None
scaler = None

def find_and_load_model():
    """Find and load model with robust path checking"""
    global model, scaler
    
    logger.info("Starting model loading process...")
    logger.info(f"Current working directory: {os.getcwd()}")
    logger.info(f"Directory contents: {os.listdir('.')}")
    
    # Define possible model paths
    possible_model_paths = [
        "/app/models",  # Docker container path
        "../models",    # Relative path
        "/workspaces/workflow-loan-/models",  # Absolute workspace path
        "models",       # Current directory
        "./models"      # Explicit current directory
    ]
    
    # Environment variable override
    env_model_path = os.getenv('MODEL_PATH')
    if env_model_path:
        possible_model_paths.insert(0, env_model_path)
    
    logger.info(f"🔍 Checking model paths: {possible_model_paths}")
    
    # Find the correct models directory
    models_dir = None
    for path in possible_model_paths:
        logger.info(f"Checking path: {path}")
        if os.path.exists(path):
            logger.info(f"Found models directory: {path}")
            if os.path.isdir(path):
                contents = os.listdir(path)
                logger.info(f"Contents: {contents}")
                models_dir = path
                break
            else:
                logger.info(f"Path exists but is not a directory: {path}")
        else:
            logger.info(f"Path not found: {path}")
    
    if not models_dir:
        raise FileNotFoundError(f"No models directory found. Checked: {possible_model_paths}")
    
    # Find the model subdirectory (RandomForest_YYYYMMDD_HHMMSS)
    model_subdirs = [d for d in os.listdir(models_dir) 
                     if os.path.isdir(os.path.join(models_dir, d)) and d.startswith('RandomForest_')]
    
    logger.info(f"Found model subdirectories: {model_subdirs}")
    
    if not model_subdirs:
        raise FileNotFoundError(f"No RandomForest model directories found in {models_dir}")
    
    # Use the latest model directory
    latest_model_dir = sorted(model_subdirs, reverse=True)[0]
    model_path = os.path.join(models_dir, latest_model_dir)
    
    logger.info(f"Selected model directory: {model_path}")
    logger.info(f"Model directory contents: {os.listdir(model_path)}")
    
    # Load the model files
    model_files = [f for f in os.listdir(model_path) 
                   if f.endswith('.joblib') or f.endswith('.pkl')]
    
    logger.info(f"Found model files: {model_files}")
    
    # Try to load the model
    model_loaded = False
    for model_file in model_files:
        if 'model' in model_file.lower():
            try:
                full_model_path = os.path.join(model_path, model_file)
                logger.info(f"Attempting to load model from: {full_model_path}")
                logger.info(f"File size: {os.path.getsize(full_model_path)} bytes")
                
                model = joblib.load(full_model_path)
                logger.info(f"Model loaded successfully: {type(model)}")
                model_loaded = True
                break
                
            except Exception as e:
                logger.warning(f"Failed to load {model_file}: {e}")
                continue
    
    if not model_loaded:
        raise Exception(f"Failed to load any model files from {model_files}")
    
    logger.info("Skipping scaler loading for now")
    scaler = None
    
    logger.info("Model loading completed successfully")

# Load model with error handling
try:
    find_and_load_model()
    MODEL_HEALTH.set(1)  # Set healthy
    logger.info("Model service initialized successfully")
except Exception as e:
    MODEL_HEALTH.set(0)  # Set unhealthy
    logger.error(f"Error loading model: {e}")
    # Don't raise - let the service start anyway for debugging
    logger.warning("Service starting without model for debugging")

def preprocess_input(data):
    """Simplified preprocessing without scaler for now"""
    try:
        # Simple preprocessing without scaler
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        else:
            df = data.copy()
        
        # Basic numeric conversion
        numeric_columns = ['person_age', 'person_income', 'person_emp_exp', 'loan_amnt', 
                          'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length', 
                          'credit_score']
        
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # For now, return simple numpy array
        return df[numeric_columns].values if all(col in df.columns for col in numeric_columns) else None
        
    except Exception as e:
        logger.error(f"Error preprocessing input: {e}")
        raise

@app.route('/health', methods=['GET'])
def health():
    """Health check with metrics tracking"""
    start_time = time.time()
    
    try:
        HTTP_REQUESTS_TOTAL.labels(method='GET', endpoint='health', status='200').inc()
        
        health_status = {
            'status': 'healthy' if model is not None else 'degraded',
            'model_loaded': model is not None,
            'scaler_loaded': scaler is not None,
            'timestamp': time.time(),
            'prediction_count': prediction_count,
            'error_count': error_count,
            'working_directory': os.getcwd(),
            'available_endpoints': ['/health', '/predict', '/info', '/metrics']
        }
        
        return jsonify(health_status), 200
        
    finally:
        duration = time.time() - start_time
        REQUEST_DURATION.labels(endpoint='health').observe(duration)

@app.route('/predict', methods=['POST'])
def predict():
    """Prediction endpoint with simplified logic"""
    global prediction_count, error_count, last_predictions
    
    start_time = time.time()
    
    try:
      
        HTTP_REQUESTS_TOTAL.labels(method='POST', endpoint='predict', status='200').inc()
        
        if model is None:
            HTTP_REQUESTS_TOTAL.labels(method='POST', endpoint='predict', status='503').inc()
            return jsonify({'error': 'Model not loaded'}), 503
        
        # Get and process data
        data = request.json
        logger.info(f"Received prediction request: {data}")
        
        # Simplified prediction for testing
        try:
            
            prediction = 1  # Mock prediction
            probability = 0.75  # Mock probability
            
          
            prediction_count += 1  # Global counter
            PREDICTION_COUNT_PROMETHEUS.inc()  # Prometheus counter
            
            # Track latency
            latency = time.time() - start_time
            prediction_latencies.append(latency)
            
            # Track recent predictions
            prediction_result = {
                'prediction': int(prediction),
                'probability': float(probability),
                'loan_status': 'Approved' if prediction == 1 else 'Rejected',
                'latency': latency,
                'timestamp': time.time(),
                'note': 'Mock prediction for testing - model integration in progress'
            }
            
            last_predictions.append(prediction_result)
            if len(last_predictions) > 100:
                last_predictions.pop(0)
            
            logger.info(f"Prediction: {prediction_result['loan_status']} (prob: {probability:.3f})")
            return jsonify(prediction_result)
            
        except Exception as prediction_error:
            logger.error(f"Prediction processing error: {prediction_error}")
            raise prediction_error
        
    except Exception as e:
       
        error_count += 1  # Global counter
        PREDICTION_ERRORS.labels(error_type='prediction_error').inc()  # Prometheus counter
        HTTP_REQUESTS_TOTAL.labels(method='POST', endpoint='predict', status='500').inc()
        
        logger.error(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500
        
    finally:
        duration = time.time() - start_time
        REQUEST_DURATION.labels(endpoint='predict').observe(duration)

@app.route('/info', methods=['GET'])
def model_info():
    """Model info with metrics tracking"""
    start_time = time.time()
    
    try:
       
        HTTP_REQUESTS_TOTAL.labels(method='GET', endpoint='info', status='200').inc()
        
        info = {
            'model_type': str(type(model).__name__) if model else 'Not loaded',
            'model_loaded': model is not None,
            'scaler_loaded': scaler is not None,
            'prediction_count': prediction_count,
            'error_count': error_count,
            'average_latency': sum(prediction_latencies) / len(prediction_latencies) if prediction_latencies else 0,
            'recent_predictions_count': len(last_predictions),
            'service_info': {
                'working_directory': os.getcwd(),
                'python_version': os.sys.version,
                'available_routes': ['/health', '/predict', '/info', '/metrics']
            }
        }
        
        return jsonify(info), 200
        
    finally:
        duration = time.time() - start_time
        REQUEST_DURATION.labels(endpoint='info').observe(duration)

@app.route('/metrics', methods=['GET'])
def metrics():
    """Prometheus metrics endpoint"""
   
    return generate_latest(), 200, {'Content-Type': 'text/plain; charset=utf-8'}

@app.route('/debug', methods=['GET'])
def debug_info():
    """Debug endpoint to check environment"""
    return jsonify({
        'working_directory': os.getcwd(),
        'directory_contents': os.listdir('.'),
        'model_loaded': model is not None,
        'scaler_loaded': scaler is not None,
        'environment_variables': {
            'MODEL_PATH': os.getenv('MODEL_PATH'),
            'PORT': os.getenv('PORT'),
            'DEBUG': os.getenv('DEBUG')
        }
    })

if __name__ == "__main__":

    MODEL_HEALTH.set(1 if model else 0)
    
    port = int(os.getenv('PORT', 5001))
    logger.info(f"Starting model serving on port {port}")
    logger.info(f"Model loaded: {model is not None}")
    logger.info(f"Scaler loaded: {scaler is not None}")
    
    try:
        app.run(host='0.0.0.0', port=port, debug=False)
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        MODEL_HEALTH.set(0)
        raise