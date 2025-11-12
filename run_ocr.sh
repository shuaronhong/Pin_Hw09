#!/bin/bash

# Menu Translator - OCR Service Startup Script
# This script automatically sets up and starts the OCR service
# Port: 5001

set -e  # Exit on error

echo "🔍 Menu Translator - Starting OCR Service"
echo "============================================="

# Get the directory where the script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OCR_DIR="$SCRIPT_DIR/OCR"

cd "$OCR_DIR"

# Check if python3 is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: python3 is not installed!"
    echo "Please install Python 3 first."
    exit 1
fi

# Check if python3-venv is available
if ! python3 -m venv --help &> /dev/null; then
    echo "❌ Error: python3-venv is not installed!"
    echo "Installing python3-venv package..."
    
    if command -v apt &> /dev/null; then
        if [ "$EUID" -ne 0 ]; then
            sudo apt update && sudo apt install -y python3-venv
        else
            apt update && apt install -y python3-venv
        fi
    elif command -v yum &> /dev/null; then
        if [ "$EUID" -ne 0 ]; then
            sudo yum install -y python3-virtualenv
        else
            yum install -y python3-virtualenv
        fi
    else
        echo "Please install python3-venv manually for your system"
        exit 1
    fi
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
    echo "✅ Virtual environment created"
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate
export PYTHONPATH="$VIRTUAL_ENV/lib/python*/site-packages:$PYTHONPATH"

# Check if dependencies need to be installed/updated
NEEDS_INSTALL=false

if ! python -c "import paddle" 2>/dev/null; then
    NEEDS_INSTALL=true
elif [ requirements.txt -nt venv/lib/python*/site-packages ]; then
    echo "📦 Requirements.txt has been updated"
    NEEDS_INSTALL=true
fi

# Install/update dependencies if needed
if [ "$NEEDS_INSTALL" = true ]; then
    echo "📥 Installing dependencies (this may take a while)..."
    pip install --upgrade pip setuptools wheel
    
    # Check for GPU and cuDNN availability
    GPU_AVAILABLE=false
    if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
        if ldconfig -p | grep -q libcudnn; then
            GPU_AVAILABLE=true
            echo "🎮 GPU and cuDNN detected! Installing GPU-accelerated version..."
        else
            echo "⚠️  GPU detected but cuDNN not found"
            echo "   Run ./install_cudnn.sh to enable GPU acceleration"
            echo "   Installing CPU version for now..."
        fi
    else
        echo "💻 No GPU detected, installing CPU version..."
    fi
    
    # Install PaddlePaddle (GPU or CPU version)
    if [ "$GPU_AVAILABLE" = true ]; then
        echo "📦 Installing PaddlePaddle GPU..."
        pip uninstall -y paddlepaddle paddlepaddle-gpu 2>/dev/null || true
        pip install paddlepaddle-gpu -i https://pypi.tuna.tsinghua.edu.cn/simple
    else
        echo "📦 Installing PaddlePaddle CPU..."
        pip uninstall -y paddlepaddle paddlepaddle-gpu 2>/dev/null || true
        pip install paddlepaddle
    fi
    
    # Install other dependencies from requirements.txt (skip paddlepaddle line)
    echo "📦 Installing other dependencies..."
    grep -v "^paddlepaddle" requirements.txt | pip install -r /dev/stdin
    
    # Install PaddleOCR without dependencies to skip problematic PyMuPDF/pdf2docx
    # (these are only needed for PDF files, not images)
    echo "📦 Installing PaddleOCR..."
    pip install paddleocr --no-deps
    
    echo ""
    echo "✅ Dependencies installed successfully"
    if [ "$GPU_AVAILABLE" = true ]; then
        echo "🚀 GPU acceleration ENABLED! Performance will be 10-100x faster."
    else
        echo "ℹ️  Running in CPU mode"
        echo "   To enable GPU: run ./install_cudnn.sh and reinstall"
    fi
else
    echo "✅ Dependencies already installed"
fi

# Start the OCR service
echo ""
echo "🚀 Starting OCR service on http://localhost:5001"
echo "============================================="
exec ./start_ocr.sh

