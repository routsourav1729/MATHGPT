"""
FastAPI service for Chain-of-Thought mathematical reasoning.
Minimal implementation for deployment.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from models import QuestionRequest, SolutionResponse
from inference_service import MathSolver
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="MATHGPT - Chain-of-Thought Reasoning",
    description="GPT-2 fine-tuned for step-by-step mathematical reasoning",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize solver (loads model on startup)
solver = None

@app.on_event("startup")
async def load_model():
    """Load the fine-tuned model on startup"""
    global solver
    try:
        solver = MathSolver()
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "MATHGPT API is running",
        "status": "healthy",
        "model_loaded": solver is not None
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "model_status": "loaded" if solver else "not_loaded"
    }

@app.post("/solve", response_model=SolutionResponse)
async def solve_math_problem(request: QuestionRequest):
    """
    Solve a mathematical word problem with step-by-step reasoning.
    
    Args:
        request: QuestionRequest containing the math problem
        
    Returns:
        SolutionResponse with reasoning steps and final answer
    """
    if not solver:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Solve the problem using existing inference code
        result = solver.solve(request.question, request.max_tokens)
        
        return SolutionResponse(
            question=request.question,
            reasoning=result["reasoning"],
            final_answer=result["answer"],
            confidence=result.get("confidence", 0.95)
        )
        
    except Exception as e:
        logger.error(f"Error solving problem: {e}")
        raise HTTPException(status_code=500, detail=f"Error solving problem: {str(e)}")

@app.post("/batch-solve")
async def solve_batch(questions: list[str]):
    """Solve multiple questions in batch"""
    if not solver:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    results = []
    for question in questions[:5]:  # Limit to 5 questions
        try:
            result = solver.solve(question, max_tokens=150)
            results.append({
                "question": question,
                "reasoning": result["reasoning"],
                "answer": result["answer"]
            })
        except Exception as e:
            results.append({
                "question": question,
                "error": str(e)
            })
    
    return {"results": results}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)