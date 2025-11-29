import torch
import gpytorch
import numpy as np
from src.utils.data_loader import load_dataset
from src.models.mixed_gp import MixedGPModel
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, classification_report
import tqdm
import json
from pathlib import Path
from datetime import datetime

def evaluate(model_path=None, batch_size=1024, limit=None, subfolder="0km_0dBm", subsample=0.26):
    # Auto-detect model path if not provided
    if model_path is None:
        model_path = Path(f"checkpoints/{subfolder}/mixed_gp_model.pth")
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}")
    
    # Load Data (Test set only ideally, but load_dataset returns both)
    print(f"Loading data for evaluation from {subfolder} with subsample {subsample}...")
    # We use a limit if specified to avoid waiting too long during dev
    _, _, test_x, test_y, scaler_x, scaler_y = load_dataset(limit=limit, subfolder=subfolder, subsample_fraction=subsample)
    
    print(f"Test size: {test_x.shape}")
    
    # Load Model
    # Load metadata to get num_latents
    state = torch.load(model_path)
    metadata = state.get('metadata', {})
    num_inducing = metadata.get('num_inducing', 500)
    num_latents = metadata.get('num_latents', 3)
    
    dummy_inducing = torch.zeros(num_inducing, 2)
    model = MixedGPModel(dummy_inducing, num_latents=num_latents, num_tasks=2)
    
    # Likelihoods - use LikelihoodList
    likelihood = gpytorch.likelihoods.LikelihoodList(
        gpytorch.likelihoods.GaussianLikelihood(),  # OSNR
        gpytorch.likelihoods.BernoulliLikelihood()  # Overlap
    )

    # Load state dict
    try:
        model.load_state_dict(state['model'])
        likelihood.load_state_dict(state['likelihood'])
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    if torch.cuda.is_available():
        test_x = test_x.cuda()
        test_y = test_y.cuda()
        model = model.cuda()
        likelihood = likelihood.cuda()
        
    model.eval()
    likelihood.eval()
    
    test_dataset = TensorDataset(test_x, test_y)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    all_preds_osnr = []
    all_preds_overlap = []
    all_vars_osnr = []
    all_probs_overlap = []
    all_targets_osnr = []
    all_targets_overlap = []
    
    print("Running inference...")
    with torch.no_grad():
        for batch_x, batch_y in tqdm.tqdm(test_loader):
            output = model(batch_x)
            # output is MultitaskMultivariateNormal
            mean = output.mean
            var = output.variance
            
            # OSNR (Task 0)
            # Inverse transform scaler
            # We predict in normalized space, then convert back.
            pred_osnr_norm = mean[:, 0]
            var_osnr_norm = var[:, 0]
            
            # Overlap (Task 1) - use likelihood to get probabilities
            dist_overlap = gpytorch.distributions.MultivariateNormal(mean[:, 1], torch.diag_embed(var[:, 1]))
            probs_overlap = likelihood.likelihoods[1](dist_overlap).mean
            
            all_preds_osnr.append(pred_osnr_norm.cpu().numpy())
            all_vars_osnr.append(var_osnr_norm.cpu().numpy())
            all_probs_overlap.append(probs_overlap.cpu().numpy())
            
            all_targets_osnr.append(batch_y[:, 0].cpu().numpy())
            all_targets_overlap.append(batch_y[:, 1].cpu().numpy())
            
    # Concatenate
    preds_osnr_norm = np.concatenate(all_preds_osnr)
    vars_osnr_norm = np.concatenate(all_vars_osnr)
    probs_overlap = np.concatenate(all_probs_overlap)
    targets_osnr_norm = np.concatenate(all_targets_osnr)
    targets_overlap = np.concatenate(all_targets_overlap)
    
    # Inverse transform OSNR
    # We need the scaler.
    preds_osnr = scaler_y.inverse_transform(preds_osnr_norm.reshape(-1, 1)).flatten()
    targets_osnr = scaler_y.inverse_transform(targets_osnr_norm.reshape(-1, 1)).flatten()
    # Variance scales by std^2
    vars_osnr = vars_osnr_norm * (scaler_y.scale_[0] ** 2)
    std_osnr = np.sqrt(vars_osnr)
    
    # Metrics
    mae_osnr = mean_absolute_error(targets_osnr, preds_osnr)
    rmse_osnr = np.sqrt(mean_squared_error(targets_osnr, preds_osnr))
    
    preds_overlap_binary = (probs_overlap > 0.5).astype(float)
    acc_overlap = accuracy_score(targets_overlap, preds_overlap_binary)
    
    print("\n=== Evaluation Results ===")
    print(f"OSNR MAE: {mae_osnr:.4f}")
    print(f"OSNR RMSE: {rmse_osnr:.4f}")
    print(f"Overlap Accuracy: {acc_overlap:.4f}")
    print("\nOverlap Classification Report:")
    print(classification_report(targets_overlap, preds_overlap_binary))
    
    print("\n=== Confidence Examples ===")
    print("OSNR (Pred +/- 2*Std | True):")
    num_examples = min(5, len(preds_osnr))
    for i in range(num_examples):
        print(f"{preds_osnr[i]:.2f} +/- {2*std_osnr[i]:.2f} | {targets_osnr[i]:.2f}")
        
    print("\nOverlap (Prob | True):")
    for i in range(num_examples):
        print(f"{probs_overlap[i]:.4f} | {targets_overlap[i]}")
    
    # Save results
    results_dir = Path(f"results/{subfolder}")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        'subfolder': subfolder,
        'subsample': subsample,
        'test_size': len(preds_osnr),
        'timestamp': datetime.now().isoformat(),
        'osnr_metrics': {
            'mae': float(mae_osnr),
            'rmse': float(rmse_osnr),
            'mean_confidence_2sigma': float(np.mean(2*std_osnr))
        },
        'overlap_metrics': {
            'accuracy': float(acc_overlap),
            'mean_probability_confidence': float(np.mean(np.abs(probs_overlap - 0.5)))
        }
    }
    
    results_path = results_dir / f"evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--subfolder", type=str, default="0km_0dBm")
    parser.add_argument("--subsample", type=float, default=0.26)
    args = parser.parse_args()
    evaluate(limit=args.limit, subfolder=args.subfolder, subsample=args.subsample)
