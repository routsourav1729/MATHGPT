"""
Test model functionality and data validation.
Tests Pydantic models and basic inference logic.
"""

import pytest
import sys
import os

# Add paths for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from api.models import QuestionRequest, SolutionResponse


def test_question_request_valid():
    """Test QuestionRequest with valid data"""
    request = QuestionRequest(
        question="What is 2 + 2?",
        max_tokens=100
    )
    
    assert request.question == "What is 2 + 2?"
    assert request.max_tokens == 100


def test_question_request_default_max_tokens():
    """Test QuestionRequest with default max_tokens"""
    request = QuestionRequest(question="Simple question?")
    
    assert request.question == "Simple question?"
    assert request.max_tokens == 200  # Default value


def test_question_request_max_tokens_limits():
    """Test max_tokens validation limits"""
    
    # Test minimum limit
    with pytest.raises(ValueError):
        QuestionRequest(question="Test?", max_tokens=10)  # Below 50 minimum
    
    # Test maximum limit
    with pytest.raises(ValueError):
        QuestionRequest(question="Test?", max_tokens=1000)  # Above 500 maximum
    
    # Test valid range
    request = QuestionRequest(question="Test?", max_tokens=150)
    assert request.max_tokens == 150


def test_question_request_empty_question():
    """Test QuestionRequest with empty question"""
    # Should still create the object (validation handled by API)
    request = QuestionRequest(question="")
    assert request.question == ""


def test_solution_response_valid():
    """Test SolutionResponse with valid data"""
    response = SolutionResponse(
        question="What is 2 + 2?",
        reasoning="2 + 2 equals 4",
        final_answer="4",
        confidence=0.95
    )
    
    assert response.question == "What is 2 + 2?"
    assert response.reasoning == "2 + 2 equals 4"
    assert response.final_answer == "4"
    assert response.confidence == 0.95


def test_solution_response_confidence_limits():
    """Test confidence score validation"""
    
    # Test minimum confidence
    with pytest.raises(ValueError):
        SolutionResponse(
            question="Test?",
            reasoning="Test reasoning",
            final_answer="42",
            confidence=-0.1  # Below 0.0
        )
    
    # Test maximum confidence
    with pytest.raises(ValueError):
        SolutionResponse(
            question="Test?",
            reasoning="Test reasoning", 
            final_answer="42",
            confidence=1.1  # Above 1.0
        )
    
    # Test valid confidence
    response = SolutionResponse(
        question="Test?",
        reasoning="Test reasoning",
        final_answer="42",
        confidence=0.85
    )
    assert response.confidence == 0.85


def test_solution_response_optional_answer():
    """Test SolutionResponse with optional final_answer"""
    response = SolutionResponse(
        question="Complex question?",
        reasoning="This is complex reasoning...",
        final_answer=None,  # Optional field
        confidence=0.7
    )
    
    assert response.final_answer is None
    assert response.reasoning == "This is complex reasoning..."


def test_model_serialization():
    """Test model JSON serialization"""
    request = QuestionRequest(
        question="What is 5 + 3?",
        max_tokens=150
    )
    
    # Test conversion to dict
    request_dict = request.model_dump()
    assert isinstance(request_dict, dict)
    assert request_dict["question"] == "What is 5 + 3?"
    assert request_dict["max_tokens"] == 150
    
    # Test JSON serialization
    request_json = request.model_dump_json()
    assert isinstance(request_json, str)
    assert "What is 5 + 3?" in request_json


def test_answer_extraction_patterns():
    """Test basic answer extraction logic"""
    # This tests the logic that would be in inference_service.py
    import re
    
    test_cases = [
        ("Therefore, the answer is 42.", "42"),
        ("The answer is 15.5", "15.5"),
        ("#### 123", "123"),
        ("Final result = 99", "99"),
        ("No clear answer here", None)
    ]
    
    # Simple answer extraction patterns (similar to inference_service.py)
    patterns = [
        r"Therefore,?\s*the answer is\s*([+-]?\d+(?:\.\d+)?)",
        r"The answer is\s*([+-]?\d+(?:\.\d+)?)",
        r"####\s*([+-]?\d+(?:\.\d+)?)",
        r"=\s*([+-]?\d+(?:\.\d+)?)(?:\s|$|\.)",
    ]
    
    for text, expected in test_cases:
        found_answer = None
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                try:
                    found_answer = str(float(matches[-1]))
                    break
                except ValueError:
                    continue
        
        if expected is None:
            assert found_answer is None, f"Expected None for '{text}' but got '{found_answer}'"
        else:
            assert found_answer == str(float(expected)), f"Expected '{expected}' for '{text}' but got '{found_answer}'"


def test_question_preprocessing():
    """Test question text preprocessing"""
    # Test basic text cleaning
    test_cases = [
        ("  What is 2+2?  ", "What is 2+2?"),
        ("WHAT IS 5+3?", "WHAT IS 5+3?"),  # Case preservation
        ("Question: What is 1+1?", "Question: What is 1+1?"),  # Format preservation
    ]
    
    for input_text, expected in test_cases:
        cleaned = input_text.strip()
        assert cleaned == expected