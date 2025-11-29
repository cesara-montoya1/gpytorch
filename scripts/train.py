import torch
import gpytorch
import numpy as np
from src.utils.data_loader import load_dataset
from src.models.mixed_gp import MixedGPModel
from torch.utils.data import TensorDataset, DataLoader
import tqdm
from pathlib import Path
import json
from datetime import datetime

def train(epochs=50, batch_size=256, lr=0.01, smoke_test=False, subfolder="0km_0dBm", subsample=0.26):
    # Load Data
    print(f"Loading data from {subfolder} with subsample {subsample}...")
    train_x, train_y, test_x, test_y, scaler_x, scaler_y = load_dataset(subfolder=subfolder, subsample_fraction=subsample)
    
    if smoke_test:
        print("Smoke test: Using subset of data.")
        train_x = train_x[:1000]
        train_y = train_y[:1000]
        test_x = test_x[:100]
        test_y = test_y[:100]
    
    print(f"Train size: {train_x.shape}")
    
    # Create DataLoader
    train_dataset = TensorDataset(train_x, train_y)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize Model
    # Inducing points: subset of training data (e.g., 500 points)
    num_inducing = 500
    inducing_idx = torch.randperm(train_x.size(0))[:num_inducing]
    inducing_points = train_x[inducing_idx].clone()
    
    model = MixedGPModel(inducing_points)
    
    # Likelihoods
    likelihood_osnr = gpytorch.likelihoods.GaussianLikelihood()
    likelihood_overlap = gpytorch.likelihoods.BernoulliLikelihood()
    
    # Move to GPU if available
    if torch.cuda.is_available():
        train_x = train_x.cuda()
        train_y = train_y.cuda()
        test_x = test_x.cuda()
        test_y = test_y.cuda()
        model = model.cuda()
        likelihood_osnr = likelihood_osnr.cuda()
        likelihood_overlap = likelihood_overlap.cuda()
    
    model.train()
    likelihood_osnr.train()
    likelihood_overlap.train()
    
    optimizer = torch.optim.Adam([
        {'params': model.parameters()},
        {'params': likelihood_osnr.parameters()},
        # Bernoulli likelihood has no parameters usually, but good practice
    ], lr=lr)
    
    # MLLs
    # We manually compute the loss: - (E[log p(y1|f1)] + E[log p(y2|f2)] - KL)
    
    print("Starting training...")
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_x, batch_y in tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            if torch.cuda.is_available():
                batch_x = batch_x.cuda()
                batch_y = batch_y.cuda()
            
            optimizer.zero_grad()
            
            output = model(batch_x)
            # Extract marginals
            # The output mean and variance are (batch, 2)
            # We construct univariate distributions for each task
            
            mean = output.mean
            var = output.variance
            
            dist_osnr = gpytorch.distributions.MultivariateNormal(mean[:, 0], torch.diag_embed(var[:, 0]))
            dist_overlap = gpytorch.distributions.MultivariateNormal(mean[:, 1], torch.diag_embed(var[:, 1]))
            
            # Expected Log Prob
            # We need to scale by num_data / batch_size for correct ELBO scaling
            num_data = train_x.size(0)
            scale = num_data / batch_x.size(0)
            
            log_prob_osnr = likelihood_osnr.expected_log_prob(batch_y[:, 0], dist_osnr).sum()
            log_prob_overlap = likelihood_overlap.expected_log_prob(batch_y[:, 1], dist_overlap).sum()
            
            # KL Divergence
            kl_div = model.variational_strategy.kl_divergence().sum()
            
            # Variational ELBO = E[log p] - KL
            # We minimize -ELBO
            loss = - (scale * (log_prob_osnr + log_prob_overlap) - kl_div)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs} Loss: {avg_loss:.4f}")
        
    # Save model with metadata
    checkpoint_dir = Path(f"checkpoints/{subfolder}")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = checkpoint_dir / "mixed_gp_model.pth"
    state = {
        'model': model.state_dict(),
        'likelihood_osnr': likelihood_osnr.state_dict(),
        'likelihood_overlap': likelihood_overlap.state_dict(),
        'metadata': {
            'subfolder': subfolder,
            'subsample': subsample,
            'epochs': epochs,
            'batch_size': batch_size,
            'lr': lr,
            'train_size': train_x.shape[0],
            'test_size': test_x.shape[0],
            'num_inducing': num_inducing,
            'timestamp': datetime.now().isoformat()
        }
    }
    torch.save(state, model_path)
    print(f"Model saved to {model_path}")
    
    # Save training config
    config_dir = Path(f"configs/{subfolder}")
    config_dir.mkdir(parents=True, exist_ok=True)
    
    config_path = config_dir / "training_config.json"
    with open(config_path, 'w') as f:
        json.dump(state['metadata'], f, indent=2)
    print(f"Config saved to {config_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--subfolder", type=str, default="0km_0dBm", help="Subfolder to train on")
    parser.add_argument("--subsample", type=float, default=0.26, help="Fraction of data to use (10k/38k ~= 0.26)")
    args = parser.parse_args()
    
    train(epochs=args.epochs, smoke_test=args.smoke_test, subfolder=args.subfolder, subsample=args.subsample)
