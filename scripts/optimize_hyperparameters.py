import torch
import gpytorch
import numpy as np
import optuna
from optuna.pruners import MedianPruner
from src.utils.data_loader import load_dataset
from src.models.mixed_gp import MixedGPModel
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_squared_error, accuracy_score
import tqdm
import json
from pathlib import Path
from datetime import datetime
import gc

def train_and_evaluate(num_latents, num_inducing, lr, batch_size, epochs, train_x, train_y, val_x, val_y, scaler_y):
    """
    Train model with given hyperparameters and return validation metrics.
    """
    try:
        # Create DataLoader
        train_dataset = TensorDataset(train_x, train_y)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Initialize Model
        # Ensure we don't request more inducing points than available data
        actual_num_inducing = min(num_inducing, train_x.size(0))
        inducing_idx = torch.randperm(train_x.size(0))[:actual_num_inducing]
        inducing_points = train_x[inducing_idx].clone()
        
        model = MixedGPModel(inducing_points, num_latents=num_latents, num_tasks=2)
        
        # Likelihoods
        likelihood = gpytorch.likelihoods.LikelihoodList(
            gpytorch.likelihoods.GaussianLikelihood(),
            gpytorch.likelihoods.BernoulliLikelihood()
        )
        
        # Move to GPU if available
        if torch.cuda.is_available():
            model = model.cuda()
            likelihood = likelihood.cuda()
            # Note: Data tensors are already on GPU if passed that way, or we move them here if not
            if train_x.device.type != 'cuda':
                train_x = train_x.cuda()
                train_y = train_y.cuda()
                val_x = val_x.cuda()
                val_y = val_y.cuda()
        
        model.train()
        likelihood.train()
        
        optimizer = torch.optim.Adam([
            {'params': model.parameters()},
            {'params': likelihood.parameters()},
        ], lr=lr)
        
        # Training loop
        for epoch in range(epochs):
            pbar = tqdm.tqdm(train_loader, desc=f"Trial Epoch {epoch+1}/{epochs}", leave=False)
            for batch_x, batch_y in pbar:
                if torch.cuda.is_available() and batch_x.device.type != 'cuda':
                    batch_x = batch_x.cuda()
                    batch_y = batch_y.cuda()
                
                optimizer.zero_grad()
                output = model(batch_x)
                
                mean = output.mean
                var = output.variance
                
                dist_osnr = gpytorch.distributions.MultivariateNormal(mean[:, 0], torch.diag_embed(var[:, 0]))
                dist_overlap = gpytorch.distributions.MultivariateNormal(mean[:, 1], torch.diag_embed(var[:, 1]))
                
                num_data = train_x.size(0)
                scale = num_data / batch_x.size(0)
                
                log_prob_osnr = likelihood.likelihoods[0].expected_log_prob(batch_y[:, 0], dist_osnr).sum()
                log_prob_overlap = likelihood.likelihoods[1].expected_log_prob(batch_y[:, 1], dist_overlap).sum()
                
                kl_div = model.variational_strategy.kl_divergence().sum()
                loss = -(scale * (log_prob_osnr + log_prob_overlap) - kl_div)
                
                loss.backward()
                optimizer.step()
                
                pbar.set_postfix({'loss': loss.item()})
        
        # Validation
        model.eval()
        likelihood.eval()
        
        with torch.no_grad():
            # Process validation in batches to avoid OOM if validation set is large
            val_dataset = TensorDataset(val_x, val_y)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            
            all_pred_osnr = []
            all_target_osnr = []
            all_pred_overlap = []
            all_target_overlap = []
            
            for batch_val_x, batch_val_y in val_loader:
                if torch.cuda.is_available() and batch_val_x.device.type != 'cuda':
                    batch_val_x = batch_val_x.cuda()
                    batch_val_y = batch_val_y.cuda()
                
                output = model(batch_val_x)
                mean = output.mean
                var = output.variance
                
                # OSNR predictions
                pred_osnr_norm = mean[:, 0].cpu().numpy()
                target_osnr_norm = batch_val_y[:, 0].cpu().numpy()
                
                # Inverse transform
                pred_osnr = scaler_y.inverse_transform(pred_osnr_norm.reshape(-1, 1)).flatten()
                target_osnr = scaler_y.inverse_transform(target_osnr_norm.reshape(-1, 1)).flatten()
                
                all_pred_osnr.extend(pred_osnr)
                all_target_osnr.extend(target_osnr)
                
                # Overlap predictions
                dist_overlap = gpytorch.distributions.MultivariateNormal(mean[:, 1], torch.diag_embed(var[:, 1]))
                probs_overlap = likelihood.likelihoods[1](dist_overlap).mean.cpu().numpy()
                pred_overlap = (probs_overlap > 0.5).astype(float)
                target_overlap = batch_val_y[:, 1].cpu().numpy()
                
                all_pred_overlap.extend(pred_overlap)
                all_target_overlap.extend(target_overlap)
            
            # Metrics
            rmse_osnr = np.sqrt(mean_squared_error(all_target_osnr, all_pred_osnr))
            acc_overlap = accuracy_score(all_target_overlap, all_pred_overlap)
        
        return rmse_osnr, acc_overlap

    finally:
        # Explicit cleanup
        del model
        del likelihood
        del optimizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

def objective(trial, train_x, train_y, val_x, val_y, scaler_y, epochs):
    """
    Optuna objective function.
    """
    # Suggest hyperparameters
    num_latents = trial.suggest_int('num_latents', 2, 5)
    num_inducing = trial.suggest_int('num_inducing', 300, 1000, step=100)
    lr = trial.suggest_float('lr', 1e-3, 1e-1, log=True)
    batch_size = trial.suggest_categorical('batch_size', [128, 256, 512])
    
    # Train and evaluate
    try:
        rmse_osnr, acc_overlap = train_and_evaluate(
            num_latents=num_latents,
            num_inducing=num_inducing,
            lr=lr,
            batch_size=batch_size,
            epochs=epochs,
            train_x=train_x,
            train_y=train_y,
            val_x=val_x,
            val_y=val_y,
            scaler_y=scaler_y
        )
        
        # Combined metric: normalize and combine
        # Lower RMSE is better, higher accuracy is better
        # Normalize RMSE to [0, 1] range (assume max RMSE ~ 10)
        normalized_rmse = rmse_osnr / 10.0
        # Convert accuracy to error rate
        error_rate = 1.0 - acc_overlap
        
        # Combined metric (lower is better)
        combined_metric = 0.5 * normalized_rmse + 0.5 * error_rate
        
        return combined_metric
        
    except Exception as e:
        print(f"Trial failed with error: {e}")
        # Clean up if failure occurred
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        return float('inf')

def optimize_hyperparameters(n_trials=50, subfolder="0km_0dBm", subsample=0.1, epochs=5):
    """
    Run Optuna hyperparameter optimization.
    
    Args:
        n_trials: Number of optimization trials
        subfolder: Dataset subfolder
        subsample: Fraction of data to use (smaller = faster)
        epochs: Number of training epochs per trial
    """
    print(f"Starting hyperparameter optimization with {n_trials} trials")
    print(f"Dataset: {subfolder}, Subsample: {subsample}, Epochs: {epochs}")
    
    # Load Data ONCE
    print("Loading dataset...")
    train_x, train_y, test_x, test_y, scaler_x, scaler_y = load_dataset(
        subfolder=subfolder, 
        subsample_fraction=subsample
    )
    
    # Split training data into train/val (80/20)
    n_train = int(0.8 * train_x.size(0))
    indices = torch.randperm(train_x.size(0))
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]
    
    val_x = train_x[val_idx]
    val_y = train_y[val_idx]
    train_x = train_x[train_idx]
    train_y = train_y[train_idx]
    
    # Pre-move to GPU if available to save transfer time, 
    # BUT be careful if dataset is huge. 
    # Given the user said "depleted memory", keeping it on CPU and moving batches might be safer 
    # if GPU memory is the bottleneck. However, if RAM is the bottleneck, moving to GPU frees RAM.
    # Let's keep data on CPU and move batches to GPU in the loop to be safe against GPU OOM,
    # as we have a lot of trials.
    
    print(f"Data loaded. Train size: {train_x.size(0)}, Val size: {val_x.size(0)}")
    
    # Create study
    study = optuna.create_study(
        direction='minimize',
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=2),
        study_name=f'gp_optimization_{subfolder}'
    )
    
    # Optimize
    study.optimize(
        lambda trial: objective(
            trial, 
            train_x=train_x, 
            train_y=train_y, 
            val_x=val_x, 
            val_y=val_y, 
            scaler_y=scaler_y, 
            epochs=epochs
        ),
        n_trials=n_trials,
        show_progress_bar=True
    )
    
    # Results
    print("\n" + "="*50)
    print("Optimization Complete!")
    print("="*50)
    print(f"\nBest trial: {study.best_trial.number}")
    print(f"Best value (combined metric): {study.best_value:.4f}")
    print("\nBest hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    # Save results
    results_dir = Path(f"configs/{subfolder}")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    best_params_path = results_dir / "best_hyperparameters.json"
    with open(best_params_path, 'w') as f:
        json.dump({
            'best_params': study.best_params,
            'best_value': study.best_value,
            'n_trials': n_trials,
            'subfolder': subfolder,
            'subsample': subsample,
            'epochs': epochs,
            'timestamp': datetime.now().isoformat()
        }, f, indent=2)
    
    print(f"\nBest hyperparameters saved to {best_params_path}")
    
    # Save study
    study_path = results_dir / "optuna_study.pkl"
    import pickle
    with open(study_path, 'wb') as f:
        pickle.dump(study, f)
    print(f"Study saved to {study_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Hyperparameter optimization with Optuna')
    parser.add_argument('--n-trials', type=int, default=50, help='Number of optimization trials')
    parser.add_argument('--subfolder', type=str, default='0km_0dBm', help='Dataset subfolder')
    parser.add_argument('--subsample', type=float, default=0.1, help='Fraction of data to use')
    parser.add_argument('--epochs', type=int, default=5, help='Epochs per trial')
    args = parser.parse_args()
    
    optimize_hyperparameters(
        n_trials=args.n_trials,
        subfolder=args.subfolder,
        subsample=args.subsample,
        epochs=args.epochs
    )
