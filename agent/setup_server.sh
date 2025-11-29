#!/bin/bash
# Setup script for server deployment

echo "Setting up GPyTorch project on server..."

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv .venv

# Activate environment
source .venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Install sofa library (adjust path as needed)
echo "Installing sofa library..."
SOFA_PATH="${SOFA_PATH:-/home/cesar/Documents/Universidad/SOFA/repository/sofa-lib}"
if [ -d "$SOFA_PATH" ]; then
    pip install -e "$SOFA_PATH"
    echo "Sofa library installed from $SOFA_PATH"
else
    echo "WARNING: Sofa library not found at $SOFA_PATH"
    echo "Please set SOFA_PATH environment variable or install manually"
fi

# Create necessary directories
mkdir -p models
mkdir -p results
mkdir -p logs

echo "Setup complete!"
echo ""
echo "Next steps:"
echo "1. Link or copy your data: ln -s /path/to/data ./data"
echo "2. Run training: python train.py --subfolder 0km_0dBm --epochs 10"
