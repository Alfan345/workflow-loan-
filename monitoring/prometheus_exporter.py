import time
import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import os
import json
import requests
import psutil  # For system metrics

from prometheus_client import start_http_server, Gauge, Counter, Histogram
import mlflow
from mlflow.tracking import MlflowClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prometheus metrics for MLflow tracking
MLFLOW_EXPERIMENTS_TOTAL = Gauge('mlflow_experiments_total', 'Total number of MLflow experiments')
MLFLOW_RUNS_TOTAL = Gauge('mlflow_runs_total', 'Total number of MLflow runs', ['experiment_name'])
MLFLOW_RUNS_ACTIVE = Gauge('mlflow_runs_active', 'Number of active MLflow runs', ['experiment_name'])
MLFLOW_RUNS_COMPLETED = Gauge('mlflow_runs_completed', 'Number of completed MLflow runs', ['experiment_name'])
MLFLOW_RUNS_FAILED = Gauge('mlflow_runs_failed', 'Number of failed MLflow runs', ['experiment_name'])

# Model performance metrics
MODEL_ACCURACY = Gauge('model_accuracy', 'Model accuracy', ['model_name', 'run_id'])
MODEL_PRECISION = Gauge('model_precision', 'Model precision', ['model_name', 'run_id'])
MODEL_RECALL = Gauge('model_recall', 'Model recall', ['model_name', 'run_id'])
MODEL_F1_SCORE = Gauge('model_f1_score', 'Model F1 score', ['model_name', 'run_id'])
MODEL_ROC_AUC = Gauge('model_roc_auc', 'Model ROC AUC', ['model_name', 'run_id'])

# Model serving metrics
MODEL_SERVING_STATUS = Gauge('model_serving_status', 'Model serving status (1=up, 0=down)')
MODEL_SERVING_RESPONSE_TIME = Histogram('model_serving_response_time_seconds', 'Model serving response time')

# Data drift metrics (placeholder)
DATA_DRIFT_SCORE = Gauge('data_drift_score', 'Data drift score', ['feature'])
PREDICTION_DISTRIBUTION = Histogram('prediction_distribution', 'Distribution of predictions')

# FIXED: Remove duplicate metrics definitions - define each metric only once
# System metrics
SYSTEM_CPU_USAGE = Gauge('system_cpu_usage_percent', 'System CPU usage percentage')
SYSTEM_MEMORY_USAGE = Gauge('system_memory_usage_percent', 'System memory usage percentage')
SYSTEM_MEMORY_TOTAL = Gauge('system_memory_total_bytes', 'Total system memory in bytes')
SYSTEM_MEMORY_USED = Gauge('system_memory_used_bytes', 'Used system memory in bytes')
SYSTEM_MEMORY_AVAILABLE = Gauge('system_memory_available_bytes', 'Available system memory in bytes')
SYSTEM_DISK_USAGE = Gauge('system_disk_usage_percent', 'System disk usage percentage', ['mount_point'])
SYSTEM_DISK_FREE = Gauge('system_disk_free_bytes', 'Free disk space in bytes', ['mount_point'])
SYSTEM_DISK_TOTAL = Gauge('system_disk_total_bytes', 'Total disk space in bytes', ['mount_point'])

# Process-specific metrics
PROCESS_CPU_USAGE = Gauge('process_cpu_usage_percent', 'Process CPU usage percentage', ['process_name'])
PROCESS_MEMORY_USAGE = Gauge('process_memory_usage_bytes', 'Process memory usage in bytes', ['process_name'])
PROCESS_THREADS = Gauge('process_threads_count', 'Number of threads per process', ['process_name'])

# Application-specific metrics
APP_CPU_USAGE = Gauge('application_cpu_usage_percent', 'Application CPU usage', ['app_name'])
APP_MEMORY_USAGE = Gauge('application_memory_usage_mb', 'Application memory usage in MB', ['app_name'])

# Request tracking metrics
HTTP_REQUESTS_TOTAL = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'status'])
HTTP_REQUEST_DURATION = Histogram('http_request_duration_seconds', 'HTTP request duration', ['method', 'endpoint'])
ACTIVE_CONNECTIONS = Gauge('active_connections_count', 'Number of active connections')
REQUEST_RATE = Gauge('request_rate_per_second', 'Request rate per second')
ERROR_RATE = Gauge('error_rate_percent', 'Error rate percentage')

# Model-specific request metrics
MODEL_PREDICTIONS_TOTAL = Counter('model_predictions_total', 'Total predictions made', ['model_name', 'prediction_type'])
MODEL_PREDICTIONS_SUCCESS = Counter('model_predictions_success_total', 'Successful predictions', ['model_name'])
MODEL_PREDICTIONS_ERROR = Counter('model_predictions_error_total', 'Failed predictions', ['model_name', 'error_type'])
MODEL_PREDICTION_LATENCY = Histogram('model_prediction_latency_seconds', 'Model prediction latency', ['model_name'])

# Queue and throughput metrics
PREDICTION_QUEUE_SIZE = Gauge('prediction_queue_size', 'Number of requests in prediction queue')
THROUGHPUT_REQUESTS_PER_MINUTE = Gauge('throughput_requests_per_minute', 'Requests processed per minute')

class SystemMetricsCollector:
    """Collect system-level metrics"""
    
    def __init__(self):
        self.last_request_count = 0
        self.last_request_time = time.time()
        self.request_history = []
        
    def collect_system_metrics(self):
        """Collect CPU, Memory, and Disk metrics"""
        try:
            # System-level CPU
            cpu_percent = psutil.cpu_percent(interval=1)
            SYSTEM_CPU_USAGE.set(cpu_percent)
            
            # Memory metrics
            memory = psutil.virtual_memory()
            SYSTEM_MEMORY_USAGE.set(memory.percent)
            SYSTEM_MEMORY_TOTAL.set(memory.total)
            SYSTEM_MEMORY_USED.set(memory.used)
            SYSTEM_MEMORY_AVAILABLE.set(memory.available)
            
            # Disk metrics for root partition
            disk = psutil.disk_usage('/')
            SYSTEM_DISK_USAGE.labels(mount_point='/').set((disk.used / disk.total) * 100)
            SYSTEM_DISK_FREE.labels(mount_point='/').set(disk.free)
            SYSTEM_DISK_TOTAL.labels(mount_point='/').set(disk.total)
            
            logger.debug(f"System metrics - CPU: {cpu_percent}%, Memory: {memory.percent}%, Disk: {(disk.used/disk.total)*100:.1f}%")
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {str(e)}")
    
    def collect_process_metrics(self):
        """Collect process-specific metrics"""
        try:
            current_process = psutil.Process()
            
            # Current process metrics (Prometheus Exporter)
            PROCESS_CPU_USAGE.labels(process_name='prometheus_exporter').set(current_process.cpu_percent())
            PROCESS_MEMORY_USAGE.labels(process_name='prometheus_exporter').set(current_process.memory_info().rss)
            PROCESS_THREADS.labels(process_name='prometheus_exporter').set(current_process.num_threads())
            
            # Application-level metrics for model serving
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    if proc.info['cmdline'] and any('inference.py' in arg for arg in proc.info['cmdline']):
                        process = psutil.Process(proc.info['pid'])
                        
                        # Process metrics
                        PROCESS_CPU_USAGE.labels(process_name='model_serving').set(process.cpu_percent())
                        PROCESS_MEMORY_USAGE.labels(process_name='model_serving').set(process.memory_info().rss)
                        PROCESS_THREADS.labels(process_name='model_serving').set(process.num_threads())
                        
                        # Application metrics
                        APP_CPU_USAGE.labels(app_name='loan_model_api').set(process.cpu_percent())
                        APP_MEMORY_USAGE.labels(app_name='loan_model_api').set(process.memory_info().rss / 1024 / 1024)  # MB
                        break
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
                    
        except Exception as e:
            logger.error(f"Error collecting process metrics: {str(e)}")

class RequestMetricsCollector:
    """Collect REAL request metrics from model serving"""
    
    def __init__(self, model_serving_url: str):
        self.model_serving_url = model_serving_url
        self.last_total_predictions = 0
        self.last_total_errors = 0
        self.last_collection_time = time.time()
        self.real_metrics_available = False
        
    def collect_request_metrics(self):
        """Collect REAL request metrics from model serving endpoint"""
        try:
            # Get REAL metrics from model serving /metrics endpoint
            response = requests.get(f"{self.model_serving_url}/metrics", timeout=5)
            
            if response.status_code == 200:
                metrics_text = response.text
                self._parse_real_metrics(metrics_text)
                self.real_metrics_available = True
                logger.debug("✅ Successfully collected REAL request metrics")
            else:
                logger.warning(f"⚠️  Model serving /metrics endpoint returned {response.status_code}")
                self._set_zero_metrics()
                self.real_metrics_available = False
                
        except requests.exceptions.RequestException as e:
            logger.warning(f"⚠️  Model serving endpoint unreachable: {e}")
            self._set_zero_metrics()
            self.real_metrics_available = False
        except Exception as e:
            logger.error(f"❌ Error collecting REAL request metrics: {str(e)}")
            self._set_zero_metrics()
    
    def _parse_real_metrics(self, metrics_text: str):
        """Parse REAL Prometheus metrics from model serving"""
        try:
            lines = metrics_text.split('\n')
            current_requests = 0
            current_errors = 0
            current_time = time.time()
            current_connections = 0
            
            # Parse REAL metrics from model serving
            for line in lines:
                # Look for http_requests_total (not ml_predictions_total)
                if line.startswith('http_requests_total') and not line.startswith('#'):
                    try:
                        parts = line.split(' ')
                        if len(parts) >= 2:
                            value = float(parts[-1])
                            current_requests += value
                            
                            # Extract labels and copy to our metrics
                            if 'method="POST"' in line and 'endpoint="predict"' in line:
                                if 'status="200"' in line:
                                    HTTP_REQUESTS_TOTAL.labels(method='POST', endpoint='predict', status='200')._value._value = value
                                elif 'status="500"' in line:
                                    HTTP_REQUESTS_TOTAL.labels(method='POST', endpoint='predict', status='500')._value._value = value
                            elif 'method="GET"' in line and 'endpoint="health"' in line:
                                HTTP_REQUESTS_TOTAL.labels(method='GET', endpoint='health', status='200')._value._value = value
                            elif 'method="GET"' in line and 'endpoint="info"' in line:
                                HTTP_REQUESTS_TOTAL.labels(method='GET', endpoint='info', status='200')._value._value = value
                                
                    except (ValueError, IndexError):
                        continue
                        
                elif line.startswith('prediction_errors_total') and not line.startswith('#'):
                    try:
                        parts = line.split(' ')
                        if len(parts) >= 2:
                            value = float(parts[-1])
                            current_errors += value
                    except (ValueError, IndexError):
                        continue
                        
                elif line.startswith('active_connections_count') and not line.startswith('#'):
                    try:
                        parts = line.split(' ')
                        if len(parts) >= 2:
                            current_connections = float(parts[-1])
                    except (ValueError, IndexError):
                        continue
            
            # Update derived metrics with REAL data
            self._update_real_rates(current_requests, current_errors, current_time)
            
            # Update active connections with real data
            ACTIVE_CONNECTIONS.set(current_connections)
            
            logger.debug(f"📊 REAL metrics - Requests: {current_requests}, Errors: {current_errors}, Connections: {current_connections}")
            
        except Exception as e:
            logger.error(f"❌ Error parsing REAL metrics: {str(e)}")
    
    def _update_real_rates(self, current_requests, current_errors, current_time):
        """Update rate metrics based on REAL data"""
        try:
            # Calculate request rate from real data
            time_diff = current_time - self.last_collection_time
            if time_diff > 0 and hasattr(self, 'last_total_predictions'):
                prediction_diff = current_requests - self.last_total_predictions
                request_rate = prediction_diff / time_diff
                REQUEST_RATE.set(max(0, request_rate))
                
                # Requests per minute
                rpm = request_rate * 60
                THROUGHPUT_REQUESTS_PER_MINUTE.set(max(0, rpm))
                
                # Error rate from real data
                if prediction_diff > 0:
                    error_diff = current_errors - self.last_total_errors
                    error_rate = (error_diff / prediction_diff) * 100
                    ERROR_RATE.set(max(0, error_rate))
                else:
                    ERROR_RATE.set(0)
                    
                logger.debug(f"📈 REAL rates - Request rate: {request_rate:.2f}/s, RPM: {rpm:.1f}, Error rate: {ERROR_RATE._value.get():.1f}%")
            
            # Update tracking variables
            self.last_total_predictions = current_requests
            self.last_total_errors = current_errors
            self.last_collection_time = current_time
            
        except Exception as e:
            logger.error(f"❌ Error updating REAL rates: {str(e)}")
    
    def _set_zero_metrics(self):
        """Set metrics to zero when model serving is unavailable"""
        REQUEST_RATE.set(0)
        THROUGHPUT_REQUESTS_PER_MINUTE.set(0)
        ERROR_RATE.set(0)
        ACTIVE_CONNECTIONS.set(0)
        
    def simulate_queue_metrics(self):
        """Simulate queue metrics (keeping this for compatibility)"""
        import random
        # Only simulate if real metrics are not available
        if not self.real_metrics_available:
            PREDICTION_QUEUE_SIZE.set(0)
        else:
            # Keep queue size low when real metrics are working
            PREDICTION_QUEUE_SIZE.set(random.randint(0, 2))

class MLflowMetricsExporter:
    def __init__(self, tracking_uri: str = "file:../modelling/mlruns", 
                 model_serving_url: str = "http://localhost:5001"):  # Fixed port
        """
        Initialize MLflow metrics exporter
        
        Args:
            tracking_uri: MLflow tracking URI
            model_serving_url: URL of the model serving endpoint
        """
        self.tracking_uri = tracking_uri
        self.model_serving_url = model_serving_url
        self.client = None
        self.last_update = datetime.now()
        
        # Initialize metric collectors
        self.system_collector = SystemMetricsCollector()
        self.request_collector = RequestMetricsCollector(model_serving_url)
        
        # Setup MLflow client
        try:
            mlflow.set_tracking_uri(tracking_uri)
            self.client = MlflowClient(tracking_uri)
            logger.info(f"Connected to MLflow tracking server: {tracking_uri}")
        except Exception as e:
            logger.error(f"Failed to connect to MLflow: {str(e)}")
    
    def collect_mlflow_metrics(self):
        """Collect metrics from MLflow tracking server"""
        if not self.client:
            return
            
        try:
            # Get all experiments - FIXED: use search_experiments instead of list_experiments
            experiments = self.client.search_experiments()
            MLFLOW_EXPERIMENTS_TOTAL.set(len(experiments))
            
            for experiment in experiments:
                if experiment.name == 'Default':
                    continue
                    
                # Get runs for this experiment
                runs = self.client.search_runs(experiment.experiment_id)
                
                # Count runs by status
                active_count = sum(1 for run in runs if run.info.status == 'RUNNING')
                completed_count = sum(1 for run in runs if run.info.status == 'FINISHED')
                failed_count = sum(1 for run in runs if run.info.status == 'FAILED')
                
                # Update metrics
                MLFLOW_RUNS_TOTAL.labels(experiment_name=experiment.name).set(len(runs))
                MLFLOW_RUNS_ACTIVE.labels(experiment_name=experiment.name).set(active_count)
                MLFLOW_RUNS_COMPLETED.labels(experiment_name=experiment.name).set(completed_count)
                MLFLOW_RUNS_FAILED.labels(experiment_name=experiment.name).set(failed_count)
                
                # Get model performance metrics from recent runs
                for run in runs[:10]:  # Latest 10 runs
                    metrics = run.data.metrics
                    params = run.data.params
                    
                    model_name = params.get('model_type', 'RandomForest')
                    run_id = run.info.run_id[:8]  # Short run ID
                    
                    # Update model performance metrics
                    if 'test_accuracy' in metrics:
                        MODEL_ACCURACY.labels(model_name=model_name, run_id=run_id).set(metrics['test_accuracy'])
                    if 'test_precision' in metrics:
                        MODEL_PRECISION.labels(model_name=model_name, run_id=run_id).set(metrics['test_precision'])
                    if 'test_recall' in metrics:
                        MODEL_RECALL.labels(model_name=model_name, run_id=run_id).set(metrics['test_recall'])
                    if 'test_f1_score' in metrics:
                        MODEL_F1_SCORE.labels(model_name=model_name, run_id=run_id).set(metrics['test_f1_score'])
                    if 'test_roc_auc' in metrics:
                        MODEL_ROC_AUC.labels(model_name=model_name, run_id=run_id).set(metrics['test_roc_auc'])
                    
                    # Also try alternative metric names
                    if 'accuracy' in metrics:
                        MODEL_ACCURACY.labels(model_name=model_name, run_id=run_id).set(metrics['accuracy'])
                    if 'precision' in metrics:
                        MODEL_PRECISION.labels(model_name=model_name, run_id=run_id).set(metrics['precision'])
                    if 'recall' in metrics:
                        MODEL_RECALL.labels(model_name=model_name, run_id=run_id).set(metrics['recall'])
                    if 'f1_score' in metrics:
                        MODEL_F1_SCORE.labels(model_name=model_name, run_id=run_id).set(metrics['f1_score'])
                    if 'roc_auc' in metrics:
                        MODEL_ROC_AUC.labels(model_name=model_name, run_id=run_id).set(metrics['roc_auc'])
            
            logger.debug(f"Updated MLflow metrics for {len(experiments)} experiments")
            
        except Exception as e:
            logger.error(f"Error collecting MLflow metrics: {str(e)}")
    
    def check_model_serving_health(self):
        """Check model serving endpoint health"""
        try:
            start_time = time.time()
            response = requests.get(f"{self.model_serving_url}/health", timeout=5)
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                MODEL_SERVING_STATUS.set(1)
                MODEL_SERVING_RESPONSE_TIME.observe(response_time)
                logger.debug("Model serving endpoint is healthy")
            else:
                MODEL_SERVING_STATUS.set(0)
                logger.warning(f"Model serving endpoint returned status {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            MODEL_SERVING_STATUS.set(0)
            logger.warning(f"Model serving endpoint is unreachable: {str(e)}")
        except Exception as e:
            MODEL_SERVING_STATUS.set(0)
            logger.error(f"Error checking model serving health: {str(e)}")
    
    def simulate_data_drift_monitoring(self):
        """Simulate data drift monitoring (placeholder)"""
        # This is a placeholder for actual data drift detection
        # In production, you would implement proper drift detection
        import random
        
        features = ['person_age', 'person_income', 'loan_amnt', 'credit_score']
        
        for feature in features:
            # Simulate drift score (0-1, where 1 is high drift)
            drift_score = random.uniform(0, 0.3)  # Low drift simulation
            DATA_DRIFT_SCORE.labels(feature=feature).set(drift_score)
    
    def collect_prediction_distribution(self):
        """Collect prediction distribution from serving logs"""
        # This is a placeholder for actual prediction logging
        # In production, you would collect this from your prediction logs
        import random
        
        # Simulate prediction distribution
        for _ in range(10):
            prediction_confidence = random.uniform(0.5, 1.0)
            PREDICTION_DISTRIBUTION.observe(prediction_confidence)
    
    def collect_all_metrics(self):
        """Collect all types of metrics"""
        try:
            # MLflow metrics
            self.collect_mlflow_metrics()
            
            # Model serving health
            self.check_model_serving_health()
            
            # System metrics
            self.system_collector.collect_system_metrics()
            self.system_collector.collect_process_metrics()
            
            # Request metrics
            self.request_collector.collect_request_metrics()
            self.request_collector.simulate_queue_metrics()
            
            # Data drift and prediction distribution
            self.simulate_data_drift_monitoring()
            self.collect_prediction_distribution()
            
        except Exception as e:
            logger.error(f"Error in collect_all_metrics: {str(e)}")
    
    def run_collection_loop(self, interval: int = 30):
        """Run metrics collection in a loop"""
        logger.info(f"Starting comprehensive metrics collection loop with {interval}s interval")
        
        while True:
            try:
                self.collect_all_metrics()
                self.last_update = datetime.now()
                
                # Log current status
                logger.info(f"Metrics updated at {self.last_update.strftime('%Y-%m-%d %H:%M:%S')}")
                
                time.sleep(interval)
                
            except KeyboardInterrupt:
                logger.info("Metrics collection stopped by user")
                break
            except Exception as e:
                logger.error(f"Error in metrics collection loop: {str(e)}")
                time.sleep(interval)

def main():
    """Main function to start the metrics exporter with enhanced monitoring"""
    # Configuration - FIXED: default to port 5001
    tracking_uri = os.getenv('MLFLOW_TRACKING_URI', 'file:../modelling/mlruns')
    model_serving_url = os.getenv('MODEL_SERVING_URL', 'http://localhost:5001')
    prometheus_port = int(os.getenv('PROMETHEUS_EXPORTER_PORT', 8000))
    collection_interval = int(os.getenv('METRICS_COLLECTION_INTERVAL', 30))
    
    logger.info("Starting Enhanced MLflow Metrics Exporter")
    logger.info(f"MLflow URI: {tracking_uri}")
    logger.info(f"Model Serving URL: {model_serving_url}")
    logger.info(f"Prometheus Port: {prometheus_port}")
    logger.info(f"Collection Interval: {collection_interval}s")
    
    # Start Prometheus HTTP server
    start_http_server(prometheus_port)
    logger.info(f"Prometheus metrics server started on port {prometheus_port}")
    
    # Initialize and start metrics exporter
    exporter = MLflowMetricsExporter(tracking_uri, model_serving_url)
    
    # Run metrics collection in a separate thread
    collection_thread = threading.Thread(
        target=exporter.run_collection_loop,
        args=(collection_interval,),
        daemon=True
    )
    collection_thread.start()
    
    logger.info("Enhanced metrics exporter started successfully")
    logger.info("Monitoring:")
    logger.info("  ✓ MLflow experiments and runs")
    logger.info("  ✓ Model performance metrics")
    logger.info("  ✓ System resources (CPU, Memory, Disk)")
    logger.info("  ✓ Process metrics")
    logger.info("  ✓ HTTP request metrics")
    logger.info("  ✓ Model serving health")
    logger.info("  ✓ Data drift simulation")
    logger.info(f"  ✓ Metrics available at: http://localhost:{prometheus_port}/metrics")
    
    # Keep the main thread alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down enhanced metrics exporter")

if __name__ == '__main__':
    main()