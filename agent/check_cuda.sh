#!/bin/bash
# Check CUDA and PyTorch compatibility

echo "========================================="
echo "CUDA and PyTorch Compatibility Check"
echo "========================================="
echo ""

# Check CUDA version
echo "1. CUDA Version:"
if command -v nvcc &> /dev/null; then
    nvcc --version | grep "release"
else
    echo "   nvcc not found in PATH"
fi

if command -v nvidia-smi &> /dev/null; then
    nvidia-smi | grep "CUDA Version"
else
    echo "   nvidia-smi not found"
fi
echo ""

# Check GPU compute capability
echo "2. GPU Information:"
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader
echo ""

# Check PyTorch CUDA version
echo "3. PyTorch CUDA Version:"
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
VENV_PYTHON="${SCRIPT_DIR}/.venv/bin/python"

if [ -f "$VENV_PYTHON" ]; then
    $VENV_PYTHON -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version (compiled): {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"
else
    echo "   Virtual environment not found"
fi
echo ""

echo "========================================="
echo "If PyTorch CUDA version doesn't match your GPU's CUDA version,"
echo "you need to reinstall PyTorch with the correct CUDA version."
echo ""
echo "Visit: https://pytorch.org/get-started/locally/"
echo "Or see: reinstall_pytorch.sh"
echo "========================================="
