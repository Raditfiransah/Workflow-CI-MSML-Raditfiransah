import unittest
import json
import os
import tempfile
import shutil
from unittest.mock import patch, MagicMock

# Add the MLProject directory to the path
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'MLProject'))

class TestMLflowServer(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        from mlflow_server import MLflowServer, Metrics
        from config import Config
        
        self.server = MLflowServer()
        self.app = self.server.app.test_client()
        
        # Create a temporary directory for model
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Tear down test fixtures after each test method."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('mlflow.sklearn.load_model')
    def test_health_endpoint_when_model_loaded(self, mock_load_model):
        """Test health endpoint when model is loaded."""
        # Mock model loading
        mock_model = MagicMock()
        mock_load_model.return_value = mock_model
        
        # Load model
        self.server.load_model()
        
        # Test health endpoint
        response = self.app.get('/health')
        data = json.loads(response.data)
        
        self.assertEqual(response.status_code, 200)
        self.assertEqual(data['status'], 'healthy')
        self.assertTrue(data['model_loaded'])
        
    @patch('mlflow.sklearn.load_model')
    def test_health_endpoint_when_model_not_loaded(self, mock_load_model):
        """Test health endpoint when model is not loaded."""
        # Simulate model loading failure
        mock_load_model.side_effect = Exception("Failed to load model")
        
        # Try to load model (will fail)
        self.server.load_model()
        
        # Test health endpoint
        response = self.app.get('/health')
        data = json.loads(response.data)
        
        self.assertEqual(response.status_code, 200)
        self.assertEqual(data['status'], 'unhealthy')
        self.assertFalse(data['model_loaded'])
        
    def test_metrics_endpoint_disabled(self):
        """Test metrics endpoint when disabled."""
        with patch.object(self.server.config, 'METRICS_ENABLED', False):
            response = self.app.get('/metrics')
            data = json.loads(response.data)
            
            self.assertEqual(response.status_code, 400)
            self.assertIn('error', data)
            self.assertEqual(data['error'], 'Metrics disabled')
            
    @patch('mlflow.sklearn.load_model')
    def test_predict_endpoint_no_features(self, mock_load_model):
        """Test predict endpoint with no features provided."""
        # Mock model loading
        mock_model = MagicMock()
        mock_load_model.return_value = mock_model
        
        # Load model
        self.server.load_model()
        
        # Test predict endpoint with no features
        response = self.app.post('/predict', 
                                data=json.dumps({}),
                                content_type='application/json')
        data = json.loads(response.data)
        
        self.assertEqual(response.status_code, 400)
        self.assertIn('error', data)
        
    @patch('mlflow.sklearn.load_model')
    def test_predict_endpoint_invalid_json(self, mock_load_model):
        """Test predict endpoint with invalid JSON."""
        # Mock model loading
        mock_model = MagicMock()
        mock_load_model.return_value = mock_model
        
        # Load model
        self.server.load_model()
        
        # Test predict endpoint with invalid JSON
        response = self.app.post('/predict', 
                                data='invalid json',
                                content_type='application/json')
        data = json.loads(response.data)
        
        self.assertEqual(response.status_code, 400)
        self.assertIn('error', data)
        
    def test_root_endpoint(self):
        """Test root endpoint."""
        response = self.app.get('/')
        data = json.loads(response.data)
        
        self.assertEqual(response.status_code, 200)
        self.assertEqual(data['service'], 'MLflow Model Server')
        self.assertEqual(data['version'], '1.0.0')
        self.assertIn('endpoints', data)
        self.assertIn('health', data['endpoints'])
        self.assertIn('metrics', data['endpoints'])
        self.assertIn('predict', data['endpoints'])

if __name__ == '__main__':
    unittest.main()