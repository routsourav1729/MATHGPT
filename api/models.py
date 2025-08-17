"""
Pydantic models for API request/response validation.
"""

from pydantic import BaseModel, Field
from typing import Optional


class QuestionRequest(BaseModel):
    """Request model for math problem solving"""
    question: str = Field(
        ..., 
        description="Mathematical word problem to solve",
        example="Sarah has 5 apples and buys 3 more. How many apples does she have?"
    )
    max_tokens: Optional[int] = Field(
        default=200,
        description="Maximum tokens to generate for reasoning",
        ge=50,
        le=500
    )


class SolutionResponse(BaseModel):
    """Response model for solved math problem"""
    question: str = Field(description="Original question")
    reasoning: str = Field(description="Step-by-step reasoning process")
    final_answer: Optional[str] = Field(description="Extracted numerical answer")
    confidence: float = Field(description="Model confidence score", ge=0.0, le=1.0)


class ErrorResponse(BaseModel):
    """Error response model"""
    error: str = Field(description="Error message")
    detail: Optional[str] = Field(description="Detailed error information")


class HealthResponse(BaseModel):
    """Health check response"""
    status: str = Field(description="Service status")
    model_status: str = Field(description="Model loading status")