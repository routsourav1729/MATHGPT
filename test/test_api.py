"""
Test FastAPI endpoints for MATHGPT.
Minimal tests for API functionality.
"""

import pytest
from fastapi.testclient import TestClient
import sys
import os

# Add paths for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import the FastAPI app
from api.main import app

# Create test client
client = TestClient(app)


def test_root_endpoint():
    """Test the root health check endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "status" in data
    assert data["status"] == "healthy"


def test_health_endpoint():
    """Test the detailed health check"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_status" in data


def test_solve_endpoint_format():
    """Test solve endpoint with valid request format"""
    test_request = {
        "question": "What is 2 + 2?",
        "max_tokens": 100
    }
    
    response = client.post("/solve", json=test_request)
    
    # Should return 200 or 503 (if model not loaded)
    assert response.status_code in [200, 503]
    
    if response.status_code == 200:
        data = response.json()
        assert "question" in data
        assert "reasoning" in data
        assert "final_answer" in data
        assert "confidence" in data
        assert data["question"] == test_request["question"]


def test_solve_endpoint_invalid_request():
    """Test solve endpoint with invalid request"""
    # Missing required field
    invalid_request = {
        "max_tokens": 100
        # Missing "question" field
    }
    
    response = client.post("/solve", json=invalid_request)
    assert response.status_code == 422  # Validation error


def test_solve_endpoint_empty_question():
    """Test solve endpoint with empty question"""
    test_request = {
        "question": "",
        "max_tokens": 50
    }
    
    response = client.post("/solve", json=test_request)
    # Should handle empty questions gracefully
    assert response.status_code in [200, 422, 503]


def test_batch_solve_endpoint():
    """Test batch solve endpoint"""
    test_questions = [
        "What is 1 + 1?",
        "What is 3 + 4?"
    ]
    
    response = client.post("/batch-solve", json=test_questions)
    
    # Should return 200 or 503 (if model not loaded)
    assert response.status_code in [200, 503]
    
    if response.status_code == 200:
        data = response.json()
        assert "results" in data
        assert len(data["results"]) == len(test_questions)


def test_solve_max_tokens_validation():
    """Test max_tokens parameter validation"""
    # Test with invalid max_tokens (too high)
    test_request = {
        "question": "Simple question?",
        "max_tokens": 1000  # Above the 500 limit
    }
    
    response = client.post("/solve", json=test_request)
    assert response.status_code == 422  # Validation error
    
    # Test with valid max_tokens
    test_request = {
        "question": "Simple question?",
        "max_tokens": 100  # Within limits
    }
    
    response = client.post("/solve", json=test_request)
    assert response.status_code in [200, 503]  # Valid request


def test_api_documentation():
    """Test that API documentation is available"""
    response = client.get("/docs")
    assert response.status_code == 200
    
    response = client.get("/openapi.json")
    assert response.status_code == 200
    data = response.json()
    assert "info" in data
    assert "paths" in data


def test_cors_headers():
    """Test CORS headers are present"""
    response = client.options("/solve")
    # Should handle OPTIONS request for CORS
    assert response.status_code in [200, 405]  # Depending on FastAPI version