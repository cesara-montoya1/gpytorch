#!/bin/bash
# Reinstall PyTorch with correct CUDA version
# Usage: bash reinstall_pytorch.sh [cuda_version]
# Example: bash reinstall_pytorch.sh 11.8
#          bash reinstall_pytorch.sh 12.1

CUDA_VERSION=${1:-"12.1"}  # Default to CUDA 12.1

echo "========================================="
echo "Reinstalling PyTorch for CUDA $CUDA_VERSION"
echo "========================================="
echo ""

# Activate virtual environment
source .venv/bin/activate

# Uninstall existing PyTorch
echo "Uninstalling existing PyTorch..."
pip uninstall -y torch torchvision torchaudio

# Install PyTorch based on CUDA version
echo ""
echo "Installing PyTorch for CUDA $CUDA_VERSION..."

case $CUDA_VERSION in
    11.8)
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
        ;;
    12.1)
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
        ;;
    12.4)
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
        ;;
    cpu)
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
        ;;
    *)
        echo "Unsupported CUDA version: $CUDA_VERSION"
        echo "Supported versions: 11.8, 12.1, 12.4, cpu"
        echo ""
        echo "Installing latest stable version..."
        pip install torch torchvision torchaudio
        ;;
esac

# Verify installation
echo ""
echo "========================================="
echo "Verifying PyTorch installation..."
echo "========================================="
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}'); print(f'Number of GPUs: {torch.cuda.device_count() if torch.cuda.is_available() else 0}')"

echo ""
echo "Done!"
