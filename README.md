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

## Methodology

### Multi-output Gaussian Process with Linear Model of Coregionalization (LMC)

This project uses a **Sparse Variational Gaussian Process (SVGP)** with **Linear Model of Coregionalization (LMC)** for scalable multi-output learning. The implementation is built on [GPyTorch](https://github.com/cornellius-gp/gpytorch), a highly efficient GP library.

#### Why LMC for Correlated Outputs?

The I/Q constellation features are affected by **both** degradation sources:
- **Low OSNR** (noise) → constellation spread + centroid shifts
- **Spectral overlap** → constellation spread + centroid shifts

Since the same I/Q patterns can result from different causes, the outputs are **correlated**. LMC models this by using **shared latent functions** that are linearly combined to produce both outputs.

**Key Advantage**: The model learns that certain latent patterns predict both low OSNR *and* high overlap simultaneously, capturing the joint distribution rather than treating tasks independently.

#### Why Variational Inference?

Standard GPs have O(N³) complexity, making them infeasible for large datasets. **Sparse Variational GPs** reduce this to O(M²N) where M is the number of inducing points (M << N):

- **Inducing Points**: 500 learnable pseudo-inputs that summarize the dataset
- **Mini-batch Training**: Process data in batches (256 samples) for memory efficiency
- **Scalability**: Linear complexity in dataset size (N)

For 123M rows, this approach is **essential** - standard GP would require ~1.8 exabytes of memory!

#### LMC Architecture

**Latent Functions**: 3 shared Gaussian Processes model underlying patterns:
- Each latent GP captures different aspects of constellation degradation
- Tasks (OSNR, Overlap) are linear combinations of these latent functions
- The mixing weights are learned during training

**Likelihoods**:
1. **OSNR (Continuous)**: Gaussian likelihood
2. **Channel Overlap (Binary)**: Bernoulli likelihood

Handled via `LikelihoodList` for mixed output types.

#### Model Components

- **Kernel**: RBF (Radial Basis Function) with automatic relevance determination per latent function
- **Mean Function**: Constant mean per latent function
- **Variational Distribution**: Cholesky parameterization for numerical stability
- **Optimization**: ELBO (Evidence Lower Bound) with Adam optimizer

#### Training Process

```
Loss = -ELBO = -(E[log p(y|f)] - KL[q(u)||p(u)])
```

Where:
- `E[log p(y|f)]`: Expected log-likelihood (data fit)
- `KL[q(u)||p(u)]`: KL divergence (regularization)

### References

- **GPyTorch**: [https://github.com/cornellius-gp/gpytorch](https://github.com/cornellius-gp/gpytorch)
- **SVGP Paper**: [Hensman et al. (2013) - "Gaussian Processes for Big Data"](https://arxiv.org/abs/1309.6835)
- **LMC Paper**: [Journel & Huijbregts (1978) - "Mining Geostatistics"](https://scholar.google.com/scholar?q=linear+model+of+coregionalization)
- **GPyTorch Examples**: [Scalable GP Regression](https://docs.gpytorch.ai/en/stable/examples/02_Scalable_Exact_GPs/index.html)

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

## Configuration Files

Training automatically generates a configuration file at `configs/{subfolder}/training_config.json` with the following structure:

```json
{
  "subfolder": "0km_0dBm",
  "subsample": 0.26,
  "epochs": 10,
  "batch_size": 256,
  "lr": 0.01,
  "train_size": 414600,
  "test_size": 103651,
  "num_inducing": 500,
  "timestamp": "2025-11-29T09:33:17.180867"
}
```

**Fields:**
- `subfolder`: Dataset used for training
- `subsample`: Fraction of data used
- `epochs`: Number of training epochs
- `batch_size`: Mini-batch size (default: 256)
- `lr`: Learning rate (default: 0.01)
- `train_size`: Number of training samples
- `test_size`: Number of test samples
- `num_inducing`: Number of inducing points (default: 500)
- `timestamp`: Training timestamp

These files are automatically created during training and are useful for tracking experiments and reproducing results.

## Documentation
- [README.md](README.md) - This file
- [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) - Deployment guide

## Notes
- The project uses CUDA 12.1 PyTorch wheels (compatible with CUDA 12.4)
- Data is expected in `./data/` directory (can be a symlink)
- SOFA library is installed directly from GitHub
