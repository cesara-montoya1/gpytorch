# Multi-output Gaussian Process for OSNR and Overlap Prediction

## Project Overview
This project implements a Multi-output Gaussian Process using GPyTorch to predict:
1. **OSNR** (Continuous) - Optical Signal-to-Noise Ratio
2. **Channel Overlap** (Binary) - Whether channels overlap (spacing < 35.2 GHz)

## Dataset
- Total: ~123M rows across 3 subfolders
- `0km_0dBm`: 51.8M rows
- `270km_0dBm`: 27.9M rows  
- `270km_9dBm`: 43.3M rows

## Installation

### Using pip
```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Using uv (faster)
```bash
# Create virtual environment
uv venv .venv
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt
```

## Usage

### Training
```bash
python scripts/train.py --subfolder "0km_0dBm" --subsample 0.26 --epochs 10
```

**Parameters:**
- `--subfolder`: Dataset subfolder (`0km_0dBm`, `270km_0dBm`, `270km_9dBm`)
- `--subsample`: Fraction of data to use (e.g., `0.26` for ~10k samples)
- `--epochs`: Number of training epochs

### Evaluation
```bash
python scripts/evaluate.py --subfolder "0km_0dBm" --subsample 0.26
```

**Parameters:**
- `--subfolder`: Dataset subfolder to evaluate
- `--subsample`: Same fraction used during training
- `--limit`: (Optional) Limit number of test samples

## Model Architecture
- **Type**: Variational Gaussian Process (Sparse GP)
- **Inducing Points**: 500 (learnable)
- **Kernel**: RBF with automatic relevance determination
- **Likelihoods**: 
  - Gaussian for OSNR
  - Bernoulli for Overlap
- **Training**: Mini-batch with ELBO loss

## Project Structure
```
gpytorch-test/
├── checkpoints/                 # Saved model weights (.pth files)
├── configs/                     # Training configurations (JSON)
├── data/                        # Dataset (symlinked or local)
├── docs/                        # Additional documentation
├── results/                     # Evaluation results
├── scripts/                     # Execution scripts
│   ├── train.py                # Training script
│   └── evaluate.py             # Evaluation script
├── src/                         # Source code
│   ├── models/                 # Model definitions
│   └── utils/                  # Utilities (data loading, etc.)
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## Output
- **Checkpoints**: Saved to `checkpoints/{subfolder}/mixed_gp_model.pth`
- **Configs**: Saved to `configs/{subfolder}/training_config.json`
- **Results**: Saved to `results/{subfolder}/evaluation_results_*.json`

## Documentation
- [README.md](README.md) - This file
- [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) - Deployment guide

## Notes
- The project uses CUDA 12.1 PyTorch wheels (compatible with CUDA 12.4)
- Data is expected in `./data/` directory (can be a symlink)
- SOFA library is installed directly from GitHub
