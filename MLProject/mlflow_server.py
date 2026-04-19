import os
import mlflow
from prometheus_client import Gauge, Counter, Histogram, generate_latest
from flask import Flask, jsonify, request

# Import custom config
from config import get_config

# Load configuration
config = get_config()

# Metrics
class Metrics:
    requests_total = Counter('model_requests_total', 'Total number of model requests')
    requests_latency = Histogram('model_requests_latency_seconds', 'Request latency in seconds')
    model_load_time = Gauge('model_load_time_seconds', 'Model load time in seconds')
    model_size = Gauge('model_size_bytes', 'Model size in bytes')
    model_status = Gauge('model_status', 'Model status (1=loaded, 0=error)', ['model_name'])
    predictions_total = Counter('model_predictions_total', 'Total number of predictions', ['class'])
    predictions_correct = Counter('model_predictions_correct_total', 'Number of correct predictions', ['class'])
    feature_histogram = Histogram('model_feature_values', 'Distribution of feature values', ['feature_name'])
    uptime_seconds = Gauge('server_uptime_seconds', 'Server uptime in seconds')
    request_size_bytes = Histogram('model_request_size_bytes', 'Size of request payload in bytes')
    response_size_bytes = Histogram('model_response_size_bytes', 'Size of response payload in bytes')

# Flask app
class MLflowServer:
    def __init__(self):
        self.app = Flask(__name__)
        self.config = config
        self.model = None
        self._setup_routes()
        
    def _setup_routes(self):
        # Health check endpoint
        @self.app.route('/health', methods=['GET'])
        def health_check():
            status = 'healthy' if self.model else 'unhealthy'
            return jsonify({
                'status': status,
                'model_loaded': bool(self.model),
                'timestamp': self._get_timestamp()
            })
        
        # Metrics endpoint
        @self.app.route('/metrics', methods=['GET'])
        def metrics():
            if not self.config.METRICS_ENABLED:
                return jsonify({'error': 'Metrics disabled'}), 400
            
            metrics_output = generate_latest()
            return metrics_output, 200, {'Content-Type': 'text/plain; version=0.0.4; charset=utf-8'}
        
        # Model prediction endpoint
        @self.app.route('/predict', methods=['POST'])
        def predict():
            start_time = self._get_timestamp()
            
            try:
                data = request.get_json()
                if data is None:
                    return jsonify({'error': 'Invalid JSON'}), 400
                if not data or 'features' not in data:
                    return jsonify({'error': 'Invalid input data'}), 400
                
                if not self.model:
                    return jsonify({'error': 'Model not loaded'}), 500
                
                # Record request
                Metrics.requests_total.inc()
                
                # Record request size
                request_size = len(str(data))
                Metrics.request_size_bytes.observe(request_size)
                
                # Make prediction
                features = data['features']
                
                # Record feature distribution
                for i, feature_value in enumerate(features):
                    Metrics.feature_histogram.labels(feature_name=f'feature_{i}').observe(feature_value)
                
                prediction = self.model.predict([features])[0]
                probability = self.model.predict_proba([features])[0].tolist()
                
                # Record latency
                latency = self._get_timestamp() - start_time
                Metrics.requests_latency.observe(latency)
                
                # Record prediction metrics
                Metrics.predictions_total.labels(class_=str(prediction)).inc()
                # Note: For correct predictions, we'd need the true label which we don't have in serving
                # This would typically be done in a feedback loop or separate validation endpoint
                
                response_data = {
                    'prediction': prediction,
                    'probability': probability,
                    'latency': latency
                }
                
                # Record response size
                response_size = len(str(response_data))
                Metrics.response_size_bytes.observe(response_size)
                
                return jsonify(response_data)
                
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        # Root endpoint
        @self.app.route('/', methods=['GET'])
        def index():
            return jsonify({
                'service': 'MLflow Model Server',
                'version': '1.0.0',
                'endpoints': {
                    'health': '/health',
                    'metrics': '/metrics',
                    'predict': '/predict'
                }
            })
    
    def load_model(self):
        try:
            start_time = self._get_timestamp()
            self.model = mlflow.sklearn.load_model(self.config.MODEL_PATH)
            load_time = self._get_timestamp() - start_time
            
            # Update metrics
            Metrics.model_load_time.set(load_time)
            Metrics.model_status.labels(model_name='credit_risk').set(1)
            
            print(f"Model loaded successfully in {load_time:.2f} seconds")
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            Metrics.model_status.labels(model_name='credit_risk').set(0)
            return False
    
    def _get_timestamp(self):
        import time
        return time.time()
    
    def run(self):
        # Load model on startup
        if not self.load_model():
            print("Warning: Model failed to load on startup")
        
        # Start Flask server
        self.app.run(
            host=self.config.SERVER_HOST,
            port=self.config.SERVER_PORT,
            debug=False
        )

if __name__ == '__main__':
    server = MLflowServer()
    server.run()