#!/bin/bash
# Deployment script for MATHGPT

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to display usage
usage() {
    echo "Usage: $0 [hf|render|docker|local]"
    echo ""
    echo "Deployment options:"
    echo "  hf       - Deploy to Hugging Face Spaces"
    echo "  render   - Deploy to Render" 
    echo "  docker   - Build and run Docker container"
    echo "  local    - Run locally"
    echo ""
    exit 1
}

# Deploy to Hugging Face Spaces
deploy_hf() {
    print_status "Preparing Hugging Face Spaces deployment..."
    
    # Check if HF CLI is installed
    if ! command -v huggingface-cli &> /dev/null; then
        print_warning "Hugging Face CLI not found. Installing..."
        pip install huggingface_hub
    fi
    
    # Copy necessary files
    print_status "Copying files for HF Spaces..."
    cp -r ../src ./hf_spaces/
    cp -r ../reason ./hf_spaces/
    cp -r ../api ./hf_spaces/
    cp ../requirements.txt ./hf_spaces/
    
    print_success "Files ready for HF Spaces deployment"
    print_status "Next steps:"
    echo "1. Create a new Space on https://huggingface.co/new-space"
    echo "2. Choose 'Gradio' or 'Docker' as SDK"
    echo "3. Upload files from deployment/hf_spaces/ directory"
    echo "4. Set app.py as main file"
}

# Deploy to Render
deploy_render() {
    print_status "Preparing Render deployment..."
    
    # Check if render.yaml exists
    if [ ! -f "./render/render.yaml" ]; then
        print_error "render.yaml not found!"
        exit 1
    fi
    
    print_success "Render configuration ready"
    print_status "Next steps:"
    echo "1. Connect your GitHub repo to Render"
    echo "2. Create new Web Service"
    echo "3. Point to your repository"
    echo "4. Render will use the render.yaml configuration"
}

# Build and run Docker
deploy_docker() {
    print_status "Building Docker container..."
    
    cd ..
    docker build -f docker/Dockerfile.api -t mathgpt-api:latest .
    
    print_success "Docker image built successfully"
    print_status "Starting container..."
    
    docker run -d \
        --name mathgpt-api \
        -p 8000:8000 \
        -v $(pwd)/reason/logs:/app/reason/logs \
        mathgpt-api:latest
    
    print_success "Container started on http://localhost:8000"
    print_status "API docs available at http://localhost:8000/docs"
}

# Run locally
run_local() {
    print_status "Starting local development server..."
    
    cd ../api
    
    # Check if dependencies are installed
    python -c "import fastapi, uvicorn" 2>/dev/null || {
        print_warning "Installing dependencies..."
        pip install -r ../requirements.txt
    }
    
    print_status "Starting FastAPI server..."
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
}

# Main script logic
case "${1:-}" in
    hf)
        deploy_hf
        ;;
    render)
        deploy_render
        ;;
    docker)
        deploy_docker
        ;;
    local)
        run_local
        ;;
    *)
        usage
        ;;
esac