import os
import time
import threading
import psutil
import signal
import sys
from contextlib import contextmanager
from typing import Dict, Any, Optional
from functools import wraps
import logging

import mlflow
from prometheus_client import (
    Gauge, Counter, Histogram, Summary, Info, 
    generate_latest, REGISTRY, CollectorRegistry,
    multiprocess, process_collector
)
from flask import Flask, jsonify, request, g
from prometheus_flask_exporter import PrometheusMetrics

# Import custom config
from config import get_config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load configuration
config = get_config()

# Production-ready metrics class
class ProductionMetrics:
    """Centralized metrics management with production best practices"""
    
    def __init__(self, service_name: str = "mlflow_server"):
        self.service_name = service_name
        self.registry = CollectorRegistry()
        
        # Process metrics (automatically collects CPU, memory, etc.)
        process_collector.ProcessCollector(
            registry=self.registry,
            namespace=self.service_name
        )
        
        # Business metrics
        self.requests_total = Counter(
            'http_requests_total',
            'Total HTTP requests',
            ['method', 'endpoint', 'status'],
            registry=self.registry
        )
        
        self.request_duration = Histogram(
            'http_request_duration_seconds',
            'HTTP request latency in seconds',
            ['method', 'endpoint'],
            buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10),
            registry=self.registry
        )
        
        self.request_size = Histogram(
            'http_request_size_bytes',
            'HTTP request size in bytes',
            ['method', 'endpoint'],
            buckets=(100, 1000, 10000, 100000, 1000000),
            registry=self.registry
        )
        
        self.response_size = Histogram(
            'http_response_size_bytes',
            'HTTP response size in bytes',
            ['method', 'endpoint'],
            buckets=(100, 1000, 10000, 100000, 1000000),
            registry=self.registry
        )
        
        # Model-specific metrics
        self.model_info = Info(
            'model_info',
            'Model metadata information',
            registry=self.registry
        )
        
        self.model_status = Gauge(
            'model_status',
            'Model status (1=loaded, 0=error)',
            ['model_name', 'version'],
            registry=self.registry
        )
        
        self.model_load_time = Gauge(
            'model_load_seconds',
            'Time taken to load model',
            ['model_name'],
            registry=self.registry
        )
        
        self.model_load_timestamp = Gauge(
            'model_load_timestamp_seconds',
            'Timestamp when model was last loaded',
            ['model_name'],
            registry=self.registry
        )
        
        self.model_prediction_latency = Histogram(
            'model_prediction_seconds',
            'Model prediction latency',
            ['model_name', 'prediction_type'],
            buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1),
            registry=self.registry
        )
        
        self.predictions_total = Counter(
            'model_predictions_total',
            'Total predictions made',
            ['model_name', 'prediction_class', 'status'],
            registry=self.registry
        )
        
        self.prediction_confidence = Histogram(
            'model_prediction_confidence',
            'Model prediction confidence scores',
            ['model_name'],
            buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
            registry=self.registry
        )
        
        # Error metrics
        self.errors_total = Counter(
            'errors_total',
            'Total errors by type',
            ['error_type', 'endpoint'],
            registry=self.registry
        )
        
        # System metrics (collected periodically)
        self.cpu_usage = Gauge(
            'system_cpu_usage_percent',
            'System CPU usage percentage',
            ['core'],
            registry=self.registry
        )
        
        self.memory_usage = Gauge(
            'system_memory_usage_bytes',
            'System memory usage in bytes',
            ['type'],
            registry=self.registry
        )
        
        self.disk_usage = Gauge(
            'system_disk_usage_bytes',
            'System disk usage in bytes',
            ['mount', 'type'],
            registry=self.registry
        )
        
        self.active_connections = Gauge(
            'active_connections',
            'Number of active connections',
            registry=self.registry
        )
        
        # Uptime tracking
        self.start_time = time.time()
        self.uptime = Gauge(
            'server_uptime_seconds',
            'Server uptime in seconds',
            registry=self.registry
        )
        
        # Batch processing metrics
        self.batch_size = Histogram(
            'batch_size',
            'Size of prediction batches',
            ['model_name'],
            buckets=(1, 2, 5, 10, 20, 50, 100),
            registry=self.registry
        )
        
        self.request_queue_time = Histogram(
            'request_queue_time_seconds',
            'Time request spends in queue',
            ['endpoint'],
            buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1),
            registry=self.registry
        )
        
        # Start background metrics collection
        self._start_background_collection()
        
        # Set model info
        self.model_info.info({
            'service': service_name,
            'version': config.get('VERSION', '1.0.0'),
            'environment': config.get('ENVIRONMENT', 'production')
        })
        
        logger.info(f"Metrics initialized for service: {service_name}")
    
    def _start_background_collection(self):
        """Start background thread for collecting system metrics"""
        def collect_system_metrics():
            while True:
                try:
                    # CPU metrics (per core)
                    cpu_percents = psutil.cpu_percent(interval=1, percpu=True)
                    for i, cpu_percent in enumerate(cpu_percents):
                        self.cpu_usage.labels(core=str(i)).set(cpu_percent)
                    
                    # Memory metrics
                    memory = psutil.virtual_memory()
                    self.memory_usage.labels(type='available').set(memory.available)
                    self.memory_usage.labels(type='used').set(memory.used)
                    self.memory_usage.labels(type='total').set(memory.total)
                    
                    # Disk metrics
                    for partition in psutil.disk_partitions():
                        try:
                            usage = psutil.disk_usage(partition.mountpoint)
                            self.disk_usage.labels(
                                mount=partition.mountpoint, 
                                type='used'
                            ).set(usage.used)
                            self.disk_usage.labels(
                                mount=partition.mountpoint, 
                                type='free'
                            ).set(usage.free)
                        except:
                            pass
                    
                    # Uptime
                    self.uptime.set(time.time() - self.start_time)
                    
                except Exception as e:
                    logger.error(f"Error collecting system metrics: {e}")
                
                time.sleep(15)  # Collect every 15 seconds
        
        thread = threading.Thread(target=collect_system_metrics, daemon=True)
        thread.start()
    
    def track_request(self, method: str, endpoint: str, status_code: int, duration: float):
        """Track HTTP request metrics"""
        status = self._get_status_group(status_code)
        self.requests_total.labels(
            method=method, 
            endpoint=endpoint, 
            status=status
        ).inc()
        self.request_duration.labels(
            method=method, 
            endpoint=endpoint
        ).observe(duration)
    
    def track_error(self, error_type: str, endpoint: str):
        """Track errors"""
        self.errors_total.labels(error_type=error_type, endpoint=endpoint).inc()
    
    def track_model_prediction(self, model_name: str, duration: float, 
                              prediction_class: Any, confidence: Optional[float] = None,
                              batch_size_val: int = 1, status: str = "success"):
        """Track model prediction metrics"""
        self.model_prediction_latency.labels(
            model_name=model_name,
            prediction_type='single' if batch_size_val == 1 else 'batch'
        ).observe(duration)
        
        self.predictions_total.labels(
            model_name=model_name,
            prediction_class=str(prediction_class),
            status=status
        ).inc()
        
        if batch_size_val > 1:
            self.batch_size.labels(model_name=model_name).observe(batch_size_val)
        
        if confidence is not None:
            self.prediction_confidence.labels(model_name=model_name).observe(confidence)
    
    def track_model_load(self, model_name: str, load_time: float, version: str = "unknown"):
        """Track model loading metrics"""
        self.model_load_time.labels(model_name=model_name).set(load_time)
        self.model_load_timestamp.labels(model_name=model_name).set(time.time())
        self.model_status.labels(
            model_name=model_name, 
            version=version
        ).set(1)
    
    def track_request_size(self, method: str, endpoint: str, size: int):
        """Track request size"""
        self.request_size.labels(method=method, endpoint=endpoint).observe(size)
    
    def track_response_size(self, method: str, endpoint: str, size: int):
        """Track response size"""
        self.response_size.labels(method=method, endpoint=endpoint).observe(size)
    
    def track_queue_time(self, endpoint: str, duration: float):
        """Track request queue time"""
        self.request_queue_time.labels(endpoint=endpoint).observe(duration)
    
    @staticmethod
    def _get_status_group(status_code: int) -> str:
        """Categorize status codes into groups"""
        if status_code < 200:
            return '1xx'
        elif status_code < 300:
            return '2xx'
        elif status_code < 400:
            return '3xx'
        elif status_code < 500:
            return '4xx'
        else:
            return '5xx'
    
    def get_metrics(self):
        """Get current metrics"""
        return generate_latest(self.registry)


# Decorator for tracking prediction metrics
def track_prediction(model_name: str):
    """Decorator to track prediction metrics"""
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = f(*args, **kwargs)
                duration = time.time() - start_time
                
                # Extract prediction info
                if isinstance(result, dict):
                    prediction = result.get('prediction')
                    confidence = result.get('confidence')
                else:
                    prediction = result
                    confidence = None
                
                # Track metrics
                g.metrics.track_model_prediction(
                    model_name=model_name,
                    duration=duration,
                    prediction_class=prediction,
                    confidence=confidence
                )
                return result
            except Exception as e:
                duration = time.time() - start_time
                g.metrics.track_model_prediction(
                    model_name=model_name,
                    duration=duration,
                    prediction_class='error',
                    status='error'
                )
                raise
        return wrapper
    return decorator


# Context manager for tracking operations
@contextmanager
def track_operation(metrics: ProductionMetrics, operation: str, endpoint: str):
    """Context manager for tracking operations"""
    start_time = time.time()
    queue_start = getattr(g, 'request_received_time', start_time)
    
    try:
        # Track queue time
        metrics.track_queue_time(endpoint, start_time - queue_start)
        yield
    except Exception as e:
        metrics.track_error(type(e).__name__, endpoint)
        raise
    finally:
        duration = time.time() - start_time
        if hasattr(g, 'method') and hasattr(g, 'endpoint'):
            metrics.track_request(g.method, g.endpoint, getattr(g, 'status_code', 500), duration)


# Production Flask server
class ProductionMLflowServer:
    """Production-ready MLflow server with comprehensive monitoring"""
    
    def __init__(self):
        self.app = Flask(__name__)
        self.config = config
        self.model = None
        self.metrics = ProductionMetrics(service_name="credit_risk_model")
        self._setup_middleware()
        self._setup_routes()
        self._setup_health_checks()
        self._setup_graceful_shutdown()
        
    def _setup_middleware(self):
        """Setup request tracking middleware"""
        
        @self.app.before_request
        def before_request():
            g.start_time = time.time()
            g.request_received_time = time.time()
            g.method = request.method
            g.endpoint = request.endpoint or request.path
            
            # Track active connections
            self.metrics.active_connections.inc()
            
            # Track request size
            request_size = request.content_length or 0
            self.metrics.track_request_size(
                g.method, 
                g.endpoint, 
                request_size
            )
        
        @self.app.after_request
        def after_request(response):
            g.status_code = response.status_code
            
            # Track response size
            response_size = len(response.get_data())
            self.metrics.track_response_size(
                g.method, 
                g.endpoint, 
                response_size
            )
            
            # Track request duration
            duration = time.time() - g.start_time
            self.metrics.track_request(
                g.method,
                g.endpoint,
                response.status_code,
                duration
            )
            
            # Decrement active connections
            self.metrics.active_connections.dec()
            
            return response
        
        @self.app.teardown_request
        def teardown_request(error=None):
            if error:
                self.metrics.track_error(
                    type(error).__name__,
                    request.endpoint or request.path
                )
    
    def _setup_routes(self):
        """Setup API routes"""
        
        @self.app.route('/health', methods=['GET'])
        def health_check():
            """Enhanced health check endpoint"""
            health_status = {
                'status': 'healthy' if self.model else 'unhealthy',
                'model_loaded': bool(self.model),
                'timestamp': time.time(),
                'uptime': time.time() - self.metrics.start_time,
                'active_connections': self.metrics.active_connections._value.get(),
                'metrics': {
                    'total_requests': self.metrics.requests_total._value.get(),
                    'error_rate': self._calculate_error_rate()
                } if self.config.get('METRICS_ENABLED', True) else None
            }
            
            http_status = 200 if self.model else 503
            return jsonify(health_status), http_status
        
        @self.app.route('/metrics', methods=['GET'])
        def metrics():
            """Prometheus metrics endpoint"""
            if not self.config.get('METRICS_ENABLED', True):
                return jsonify({'error': 'Metrics disabled'}), 400
            
            content_type = request.headers.get('Content-Type', '')
            if 'application/json' in content_type:
                # JSON format for debugging
                metrics_data = self._get_metrics_json()
                return jsonify(metrics_data), 200
            else:
                # Prometheus format
                return self.metrics.get_metrics(), 200, {
                    'Content-Type': 'text/plain; version=0.0.4; charset=utf-8'
                }
        
        @self.app.route('/predict', methods=['POST'])
        @track_prediction(model_name='credit_risk_model')
        def predict():
            """Prediction endpoint with comprehensive tracking"""
            with track_operation(self.metrics, 'prediction', '/predict'):
                # Validate input
                data = request.get_json(silent=True)
                if data is None:
                    self.metrics.track_error('InvalidJSON', '/predict')
                    return jsonify({'error': 'Invalid JSON'}), 400
                
                if not data or 'features' not in data:
                    self.metrics.track_error('MissingFeatures', '/predict')
                    return jsonify({'error': 'Missing features field'}), 400
                
                # Check model
                if not self.model:
                    self.metrics.track_error('ModelNotLoaded', '/predict')
                    return jsonify({'error': 'Model not loaded'}), 503
                
                try:
                    # Extract features
                    features = data['features']
                    
                    # Validate batch or single prediction
                    is_batch = isinstance(features[0], (list, tuple)) if features else False
                    
                    if is_batch:
                        # Batch prediction
                        import numpy as np
                        features_array = np.array(features)
                        start_time = time.time()
                        predictions = self.model.predict(features_array)
                        prediction_time = time.time() - start_time
                        
                        # Get probabilities if available
                        probabilities = None
                        if hasattr(self.model, "predict_proba"):
                            probabilities = self.model.predict_proba(features_array).tolist()
                        
                        # Track batch metrics
                        self.metrics.track_model_prediction(
                            model_name='credit_risk_model',
                            duration=prediction_time,
                            prediction_class='batch',
                            batch_size_val=len(features)
                        )
                        
                        response = {
                            'predictions': predictions.tolist(),
                            'probabilities': probabilities,
                            'batch_size': len(features),
                            'latency': prediction_time
                        }
                    else:
                        # Single prediction
                        import numpy as np
                        features_array = np.array([features])
                        start_time = time.time()
                        prediction = self.model.predict(features_array)
                        prediction_time = time.time() - start_time
                        prediction_value = prediction[0].item()
                        
                        # Get probability
                        probability = None
                        confidence = None
                        if hasattr(self.model, "predict_proba"):
                            proba = self.model.predict_proba(features_array)
                            probability = proba[0].tolist()
                            confidence = max(probability) if probability else None
                        
                        # Track single prediction metrics
                        self.metrics.track_model_prediction(
                            model_name='credit_risk_model',
                            duration=prediction_time,
                            prediction_class=prediction_value,
                            confidence=confidence
                        )
                        
                        response = {
                            'prediction': prediction_value,
                            'probability': probability,
                            'confidence': confidence,
                            'latency': prediction_time
                        }
                    
                    return jsonify(response)
                    
                except Exception as e:
                    logger.error(f"Prediction error: {str(e)}", exc_info=True)
                    self.metrics.track_error('PredictionError', '/predict')
                    return jsonify({'error': str(e)}), 500
        
        @self.app.route('/debug/metrics', methods=['GET'])
        def debug_metrics():
            """Debug endpoint for detailed metrics"""
            if not self.config.get('DEBUG', False):
                return jsonify({'error': 'Debug mode disabled'}), 403
            
            metrics_data = {
                'start_time': self.metrics.start_time,
                'uptime': time.time() - self.metrics.start_time,
                'model_loaded': self.model is not None,
                'metrics': self._get_metrics_json()
            }
            return jsonify(metrics_data), 200
        
        @self.app.route('/reload-model', methods=['POST'])
        def reload_model():
            """Endpoint to reload model without restarting server"""
            if not self.config.get('ALLOW_MODEL_RELOAD', False):
                return jsonify({'error': 'Model reload not allowed'}), 403
            
            try:
                success = self.load_model()
                if success:
                    return jsonify({'status': 'Model reloaded successfully'}), 200
                else:
                    return jsonify({'error': 'Failed to reload model'}), 500
            except Exception as e:
                logger.error(f"Model reload error: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/', methods=['GET'])
        def index():
            """Root endpoint"""
            return jsonify({
                'service': 'Production MLflow Model Server',
                'version': self.config.get('VERSION', '1.0.0'),
                'endpoints': {
                    'health': '/health',
                    'metrics': '/metrics',
                    'predict': '/predict',
                    'debug': '/debug/metrics' if self.config.get('DEBUG', False) else None,
                    'reload': '/reload-model' if self.config.get('ALLOW_MODEL_RELOAD', False) else None
                },
                'documentation': '/docs' if self.config.get('ENABLE_DOCS', False) else None
            })
        
        # Optional: Add Swagger/OpenAPI documentation
        if self.config.get('ENABLE_DOCS', False):
            self._setup_documentation()
    
    def _setup_health_checks(self):
        """Setup periodic health checks"""
        def periodic_health_check():
            while True:
                try:
                    # Check model health
                    if self.model is None:
                        logger.warning("Model not loaded in health check")
                        self.metrics.model_status.labels(
                            model_name='credit_risk_model',
                            version=self.config.get('MODEL_VERSION', 'unknown')
                        ).set(0)
                    else:
                        self.metrics.model_status.labels(
                            model_name='credit_risk_model',
                            version=self.config.get('MODEL_VERSION', 'unknown')
                        ).set(1)
                    
                    # Check system resources
                    if psutil.virtual_memory().percent > 90:
                        logger.warning("High memory usage detected")
                    
                except Exception as e:
                    logger.error(f"Health check error: {e}")
                
                time.sleep(30)  # Run every 30 seconds
        
        thread = threading.Thread(target=periodic_health_check, daemon=True)
        thread.start()
    
    def _setup_graceful_shutdown(self):
        """Setup graceful shutdown handlers"""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, shutting down gracefully...")
            sys.exit(0)
        
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)
    
    def _setup_documentation(self):
        """Setup API documentation endpoint"""
        @self.app.route('/docs', methods=['GET'])
        def docs():
            return jsonify({
                'openapi': '3.0.0',
                'info': {
                    'title': 'MLflow Model Server API',
                    'version': self.config.get('VERSION', '1.0.0')
                },
                'paths': {
                    '/predict': {
                        'post': {
                            'summary': 'Make predictions',
                            'parameters': [{
                                'name': 'body',
                                'in': 'body',
                                'required': True,
                                'schema': {
                                    'type': 'object',
                                    'properties': {
                                        'features': {
                                            'type': 'array',
                                            'items': {'type': 'number'},
                                            'description': 'Feature vector for prediction'
                                        }
                                    }
                                }
                            }]
                        }
                    }
                }
            })
    
    def _calculate_error_rate(self) -> float:
        """Calculate error rate from metrics"""
        total_requests = self.metrics.requests_total._value.get()
        if total_requests == 0:
            return 0.0
        
        error_requests = sum(
            self.metrics.requests_total.labels(
                method=method, 
                endpoint=endpoint, 
                status=status
            )._value.get()
            for method in ['GET', 'POST']
            for endpoint in ['/predict', '/health']
            for status in ['4xx', '5xx']
        )
        
        return error_requests / total_requests
    
    def _get_metrics_json(self) -> Dict[str, Any]:
        """Get metrics in JSON format for debugging"""
        return {
            'requests': {
                'total': self.metrics.requests_total._value.get(),
                'error_rate': self._calculate_error_rate()
            },
            'model': {
                'loaded': self.model is not None,
                'load_time': self.metrics.model_load_time._value.get() if self.model else None
            },
            'system': {
                'cpu_percent': psutil.cpu_percent(),
                'memory_percent': psutil.virtual_memory().percent,
                'uptime': time.time() - self.metrics.start_time
            }
        }
    
    def load_model(self) -> bool:
        """Load model with comprehensive tracking"""
        try:
            start_time = time.time()
            model_path = self.config.get('MODEL_PATH')
            
            if not model_path or not os.path.exists(model_path):
                raise FileNotFoundError(f"Model not found at {model_path}")
            
            logger.info(f"Loading model from {model_path}")
            self.model = mlflow.sklearn.load_model(model_path)
            
            load_time = time.time() - start_time
            
            # Track metrics
            self.metrics.track_model_load(
                model_name='credit_risk_model',
                load_time=load_time,
                version=self.config.get('MODEL_VERSION', 'unknown')
            )
            
            logger.info(f"Model loaded successfully in {load_time:.2f} seconds")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}", exc_info=True)
            self.metrics.model_status.labels(
                model_name='credit_risk_model',
                version=self.config.get('MODEL_VERSION', 'unknown')
            ).set(0)
            return False
    
    def run(self):
        """Run the production server"""
        # Load model on startup
        if not self.load_model():
            logger.warning("Model failed to load on startup")
            if self.config.get('EXIT_ON_MODEL_FAILURE', False):
                logger.error("Exiting due to model load failure")
                sys.exit(1)
        
        # Get server configuration
        host = self.config.get('SERVER_HOST', '0.0.0.0')
        port = self.config.get('SERVER_PORT', 5000)
        workers = self.config.get('WORKERS', 1)
        
        logger.info(f"Starting server on {host}:{port} with {workers} workers")
        
        # Use production WSGI server if available
        try:
            from gunicorn.app.base import BaseApplication
            
            class GunicornApp(BaseApplication):
                def __init__(self, app, options=None):
                    self.options = options or {}
                    self.application = app
                    super().__init__()
                
                def load_config(self):
                    for key, value in self.options.items():
                        self.cfg.set(key, value)
                
                def load(self):
                    return self.application
            
            options = {
                'bind': f'{host}:{port}',
                'workers': workers,
                'worker_class': 'sync',
                'timeout': self.config.get('TIMEOUT', 30),
                'graceful_timeout': self.config.get('GRACEFUL_TIMEOUT', 30),
                'max_requests': self.config.get('MAX_REQUESTS', 1000),
                'max_requests_jitter': self.config.get('MAX_REQUESTS_JITTER', 100),
                'accesslog': '-',
                'errorlog': '-',
                'loglevel': self.config.get('LOG_LEVEL', 'info')
            }
            
            GunicornApp(self.app, options).run()
            
        except ImportError:
            logger.warning("Gunicorn not available, using Flask development server")
            logger.warning("This is not recommended for production!")
            
            self.app.run(
                host=host,
                port=port,
                debug=self.config.get('DEBUG', False),
                threaded=True
            )


if __name__ == '__main__':
    server = ProductionMLflowServer()
    server.run()