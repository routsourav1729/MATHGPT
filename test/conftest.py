"""
Test configuration and fixtures for MATHGPT tests.
"""

import pytest
import sys
import os
from pathlib import Path

# Add project paths to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root / "reason"))
sys.path.insert(0, str(project_root / "api"))


@pytest.fixture
def sample_questions():
    """Fixture providing sample math questions for testing"""
    return [
        "What is 2 + 2?",
        "Sarah has 5 apples and buys 3 more. How many apples does she have?",
        "A train travels 60 miles in 2 hours. What is its speed?",
        "If I have 10 cookies and eat 3, how many are left?",
        "What is 15 divided by 3?"
    ]


@pytest.fixture
def sample_question_request():
    """Fixture providing a sample QuestionRequest object"""
    from api.models import QuestionRequest
    return QuestionRequest(
        question="What is 7 + 8?",
        max_tokens=100
    )


@pytest.fixture
def sample_solution_response():
    """Fixture providing a sample SolutionResponse object"""
    from api.models import SolutionResponse
    return SolutionResponse(
        question="What is 7 + 8?",
        reasoning="7 + 8 = 15. Therefore, the answer is 15.",
        final_answer="15",
        confidence=0.95
    )


@pytest.fixture
def mock_model_path():
    """Fixture providing mock model path for testing"""
    return str(project_root / "reason" / "logs" / "gsm8k_model_final.pt")


@pytest.fixture(scope="session")
def test_client():
    """Fixture providing FastAPI test client"""
    try:
        from fastapi.testclient import TestClient
        from api.main import app
        return TestClient(app)
    except ImportError:
        pytest.skip("FastAPI not available")


# Configure pytest
def pytest_configure(config):
    """Configure pytest settings"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically"""
    for item in items:
        # Mark API tests as integration tests
        if "test_api" in item.nodeid:
            item.add_marker(pytest.mark.integration)
        
        # Mark model tests as unit tests
        if "test_model" in item.nodeid:
            item.add_marker(pytest.mark.unit)
        
        # Mark slow tests if they involve model loading
        if "solver" in item.name.lower() or "inference" in item.name.lower():
            item.add_marker(pytest.mark.slow)


# Skip tests if dependencies are missing
def pytest_runtest_setup(item):
    """Setup for each test - skip if dependencies missing"""
    
    # Skip model tests if torch is not available
    if "test_model" in item.nodeid:
        try:
            import torch
        except ImportError:
            pytest.skip("PyTorch not available")
    
    # Skip API tests if FastAPI is not available
    if "test_api" in item.nodeid:
        try:
            import fastapi
        except ImportError:
            pytest.skip("FastAPI not available")


# Error handling
@pytest.fixture(autouse=True)
def setup_test_environment():
    """Setup test environment before each test"""
    # Ensure clean state
    os.environ.pop("MODEL_PATH", None)
    os.environ.pop("API_PORT", None)
    
    yield
    
    # Cleanup after test
    # Add any cleanup logic here if needed
    pass