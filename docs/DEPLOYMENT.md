# Deployment Guide

## Overview
This guide covers deploying the project to a remote server (e.g., for GPU training).

## Initial Setup

### 1. Transfer Project to Server
```bash
# From your local machine
rsync -av --exclude='.venv' --exclude='data' --exclude='*.pth' --exclude='__pycache__' \
    ./gpytorch-test/ user@server:/path/to/gpytorch-test/
```

### 2. Setup Environment on Server
```bash
# SSH to server
ssh user@server
cd /path/to/gpytorch-test

# Create virtual environment (using pip)
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Or using uv (faster)
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

### 3. Link or Copy Data
```bash
# Option 1: Symlink (if data is already on server)
ln -s /path/to/server/data ./data

# Option 2: Transfer data (if needed)
rsync -av user@local:/path/to/data/ ./data/
```

## Running Training

### Interactive Training
```bash
# Activate virtual environment
source .venv/bin/activate

# Run training
python scripts/train.py --subfolder "0km_0dBm" --subsample 0.26 --epochs 10
```

### Background Training (using tmux)
```bash
# Start tmux session
tmux new -s gp_training

# Activate venv and run training
source .venv/bin/activate
python scripts/train.py --subfolder "0km_0dBm" --subsample 0.26 --epochs 10

# Detach from tmux: Ctrl+b, then d
# Reattach later: tmux attach -t gp_training
```

### Using Job Schedulers (SLURM, etc.)
If your server uses a job scheduler, create your own submission scripts. Example for SLURM:

```bash
#!/bin/bash
#SBATCH --job-name=gp_train
#SBATCH --output=logs/train_%j.out
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00

# Activate environment
source /path/to/gpytorch-test/.venv/bin/activate

# Run training
python /path/to/gpytorch-test/scripts/train.py \
    --subfolder "0km_0dBm" \
    --subsample 0.26 \
    --epochs 10
```

## Monitoring

### GPU Usage
```bash
# Real-time monitoring
watch -n 1 nvidia-smi

# Check CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Training Progress
Training progress is displayed in real-time. You'll see:
- Epoch-by-epoch loss updates
- Evaluation metrics after training

### Output Files
```
checkpoints/{subfolder}/mixed_gp_model.pth    # Trained model
configs/{subfolder}/training_config.json       # Training metadata
results/{subfolder}/evaluation_results_*.json  # Evaluation metrics
```

## Transferring Results

### Download Results to Local Machine
```bash
# From your local machine
rsync -av user@server:/path/to/gpytorch-test/checkpoints/ ./checkpoints/
rsync -av user@server:/path/to/gpytorch-test/results/ ./results/
```

## Troubleshooting

### GPU Not Available
```bash
# Check if GPU is visible
nvidia-smi

# Verify PyTorch sees GPU
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Out of Memory
Reduce memory usage by:
1. Reducing batch size (edit `train.py` or add `--batch-size` argument)
2. Reducing subsample fraction (e.g., `--subsample 0.15`)

### Training Interrupted
If training is interrupted, simply re-run the command. The model will train from scratch (checkpointing not yet implemented).

## Expected Runtime
With GPU (e.g., V100, A100):
- **Per epoch**: ~40 minutes
- **10 epochs**: ~6-7 hours
- **All 3 subfolders**: ~20 hours total

## Tmux Tips
```bash
# Create session
tmux new -s gp_training

# Detach: Ctrl+b, then d
# Reattach: tmux attach -t gp_training
# List sessions: tmux ls
# Kill session: tmux kill-session -t gp_training

# Split panes for monitoring
# Horizontal: Ctrl+b, then "
# Vertical: Ctrl+b, then %
# Switch panes: Ctrl+b, then arrow keys
```
