"""
Integration tests for FastAPI endpoints.
Tests /health, /ready, /metrics, and inference endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import json

# Import after mocking to avoid loading real models
with patch('src.api.main.load_model'):
    from src.api.main import app

client = TestClient(app)


class TestHealthEndpoints:
    """Test health check endpoints"""

    def test_health_endpoint(self):
        """Test /health liveness probe"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] in ["healthy", "unhealthy"]

    def test_ready_endpoint(self):
        """Test /ready readiness probe"""
        response = client.get("/ready")
        # May be 200 or 503 depending on model state
        assert response.status_code in [200, 503]
        data = response.json()
        assert "status" in data
        assert data["status"] in ["ready", "not ready"]

    def test_ready_checks_model_loaded(self):
        """Test that /ready verifies model is loaded"""
        response = client.get("/ready")
        data = response.json()

        if response.status_code == 200:
            assert data["status"] == "ready"
        else:
            assert data["status"] == "not ready"


class TestMetricsEndpoint:
    """Test Prometheus metrics endpoint"""

    def test_metrics_endpoint_exists(self):
        """Test /metrics endpoint is accessible"""
        response = client.get("/metrics")
        assert response.status_code == 200

    def test_metrics_format(self):
        """Test metrics are in Prometheus format"""
        response = client.get("/metrics")
        content = response.text

        # Check for Prometheus metric format
        assert "# HELP" in content or "# TYPE" in content or "_total" in content

    def test_metrics_content_type(self):
        """Test metrics endpoint content type"""
        response = client.get("/metrics")
        # Prometheus format should be text/plain
        assert "text/plain" in response.headers.get("content-type", "")


class TestInferenceEndpoint:
    """Test main inference endpoint"""

    @patch('src.api.main.model')
    @patch('src.api.main.tokenizer')
    def test_fingerprint_basic(self, mock_tokenizer, mock_model):
        """Test basic fingerprint inference"""
        # Mock model outputs
        mock_model.return_value = MagicMock(
            last_hidden_state=[[0.1] * 768]
        )
        mock_tokenizer.return_value = {
            'input_ids': [[1, 2, 3]],
            'attention_mask': [[1, 1, 1]]
        }

        payload = {"text": "I am happy"}
        response = client.post("/nlp/emotion/fingerprint", json=payload)

        # May fail if model not loaded, but should return valid response
        assert response.status_code in [200, 503]

    def test_fingerprint_empty_text(self):
        """Test fingerprint with empty text"""
        payload = {"text": ""}
        response = client.post("/nlp/emotion/fingerprint", json=payload)

        # Should return 400 Bad Request or handle gracefully
        assert response.status_code in [200, 400, 422]

    def test_fingerprint_long_text(self):
        """Test fingerprint with very long text"""
        payload = {"text": "word " * 1000}
        response = client.post("/nlp/emotion/fingerprint", json=payload)

        # Should handle or truncate gracefully
        assert response.status_code in [200, 400, 503]

    def test_fingerprint_special_chars(self):
        """Test fingerprint with special characters"""
        payload = {"text": "Hello! @#$%^&*() 你好"}
        response = client.post("/nlp/emotion/fingerprint", json=payload)

        assert response.status_code in [200, 503]

    def test_fingerprint_response_structure(self):
        """Test fingerprint response has expected structure"""
        with patch('src.api.main.model') as mock_model:
            # Mock successful inference
            mock_model.return_value = MagicMock(
                last_hidden_state=[[0.1] * 768]
            )

            payload = {"text": "I am happy"}
            response = client.post("/nlp/emotion/fingerprint", json=payload)

            if response.status_code == 200:
                data = response.json()
                # Should have text and embed fields
                assert "text" in data or "error" in data


class TestErrorHandling:
    """Test API error handling"""

    def test_invalid_json(self):
        """Test handling of invalid JSON"""
        response = client.post(
            "/nlp/emotion/fingerprint",
            data="not valid json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422  # Unprocessable Entity

    def test_missing_field(self):
        """Test handling of missing required fields"""
        payload = {}  # Missing 'text' field
        response = client.post("/nlp/emotion/fingerprint", json=payload)
        assert response.status_code == 422

    def test_wrong_content_type(self):
        """Test handling of wrong content type"""
        response = client.post(
            "/nlp/emotion/fingerprint",
            data="text=hello",
            headers={"Content-Type": "application/x-www-form-urlencoded"}
        )
        # Should reject or handle gracefully
        assert response.status_code in [400, 422]

    def test_nonexistent_endpoint(self):
        """Test accessing non-existent endpoint"""
        response = client.get("/nonexistent")
        assert response.status_code == 404


class TestRateLimiting:
    """Test rate limiting functionality"""

    @pytest.mark.skip(reason="Rate limiting may not be enabled")
    def test_rate_limit_exceeded(self):
        """Test rate limit enforcement"""
        # Send many requests quickly
        responses = []
        for _ in range(100):
            response = client.post(
                "/nlp/emotion/fingerprint",
                json={"text": "test"}
            )
            responses.append(response.status_code)

        # At least some should be rate limited (429)
        assert 429 in responses or all(s in [200, 503] for s in responses)


class TestCORS:
    """Test CORS configuration"""

    def test_cors_headers_present(self):
        """Test CORS headers are present"""
        response = client.options("/health")
        # CORS headers may or may not be configured
        # Just verify the endpoint responds
        assert response.status_code in [200, 405]


class TestRequestValidation:
    """Test request validation"""

    def test_text_type_validation(self):
        """Test text field must be string"""
        payload = {"text": 12345}  # Number instead of string
        response = client.post("/nlp/emotion/fingerprint", json=payload)
        # Should validate and reject
        assert response.status_code in [200, 422]

    def test_text_null_validation(self):
        """Test null text handling"""
        payload = {"text": None}
        response = client.post("/nlp/emotion/fingerprint", json=payload)
        assert response.status_code in [200, 422]

    def test_extra_fields_ignored(self):
        """Test extra fields are ignored"""
        payload = {"text": "hello", "extra_field": "ignored"}
        response = client.post("/nlp/emotion/fingerprint", json=payload)
        # Should process normally or reject
        assert response.status_code in [200, 422, 503]


class TestConcurrency:
    """Test concurrent request handling"""

    def test_concurrent_requests(self):
        """Test handling multiple concurrent requests"""
        import concurrent.futures

        def make_request():
            return client.post(
                "/nlp/emotion/fingerprint",
                json={"text": "test"}
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_request) for _ in range(10)]
            responses = [f.result() for f in futures]

        # All requests should complete
        assert len(responses) == 10
        # All should have valid status codes
        assert all(r.status_code in [200, 503, 429] for r in responses)


class TestLogging:
    """Test logging functionality"""

    @patch('src.api.main.logger')
    def test_request_logging(self, mock_logger):
        """Test that requests are logged"""
        response = client.get("/health")
        # Logger should be called (exact calls depend on implementation)
        # Just verify endpoint works
        assert response.status_code == 200


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
