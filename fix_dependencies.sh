#!/bin/bash

echo "🔧 Fixing dependency compatibility issues..."

# Activate virtual environment
source .venv/bin/activate

echo "📦 Uninstalling problematic packages..."
pip uninstall -y torch torchvision numpy

echo "📦 Installing compatible versions..."
pip install numpy==1.24.3
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --index-url https://download.pytorch.org/whl/cu118

echo "✅ Dependencies fixed! You can now restart your server."
echo "🚀 Run: python -m uvicorn app:app --host 0.0.0.0 --port 8079"
