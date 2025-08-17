"""
Hugging Face Spaces entry point for MATHGPT API.
Simple wrapper to run FastAPI app on HF Spaces.
"""

import sys
import os
from pathlib import Path

# Add project paths
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "src"))
sys.path.append(str(project_root / "reason"))
sys.path.append(str(project_root / "api"))

# Import FastAPI app
from api.main import app

# For Hugging Face Spaces
if __name__ == "__main__":
    import uvicorn
    
    # Get port from environment (HF Spaces uses port 7860)
    port = int(os.getenv("PORT", 7860))
    
    print(f"🚀 Starting MATHGPT API on port {port}")
    print(f"📚 API docs will be available at http://localhost:{port}/docs")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )