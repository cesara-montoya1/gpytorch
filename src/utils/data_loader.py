import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from pathlib import Path
import polars as pl
try:
    from sofa.data_io import load_32gbaud_db
except ImportError:
    load_32gbaud_db = None

def load_dataset(data_path="./data", test_size=0.2, seed=42, subfolder=None, limit=None, subsample_fraction=None):
    """
    Load dataset using sofa library.
    
    Args:
        data_path (str): Path to data directory.
        test_size (float): Fraction of data to use for testing.
        seed (int): Random seed.
        subfolder (str, optional): Specific subfolder to load. If None, loads all.
        limit (int, optional): Limit number of rows to load (first N).
        subsample_fraction (float, optional): Fraction of data to keep (random subsample).
        
    Returns:
        train_x, train_y, test_x, test_y, scaler_x, scaler_y_osnr
    """
    if load_32gbaud_db is None:
        raise ImportError("sofa library is required. Please install it.")
        
    path = Path(data_path)
    
    if subfolder:
        print(f"Loading subfolder: {subfolder}")
        df = load_32gbaud_db(path, full=False, subfolder=subfolder)
    else:
        # Load all
        dfs = []
        for sf in ["0km_0dBm", "270km_0dBm", "270km_9dBm"]:
            try:
                print(f"Loading subfolder: {sf}")
                dfs.append(load_32gbaud_db(path, full=False, subfolder=sf))
            except Exception as e:
                print(f"Warning: Could not load {sf}: {e}")
        
        if not dfs:
            raise ValueError("No data loaded.")
        df = pl.concat(dfs)

    if limit:
        df = df.head(limit)
        
    if subsample_fraction:
        print(f"Subsampling with fraction: {subsample_fraction}")
        # Polars sample is efficient
        df = df.sample(fraction=subsample_fraction, seed=seed)

    # Convert to pandas/numpy for easier handling with sklearn/torch
    # Polars to numpy
    I = df["I"].to_numpy()
    Q = df["Q"].to_numpy()
    OSNR = df["OSNR"].to_numpy()
    Spacing = df["Spacing"].to_numpy()
    
    # Inputs: I, Q
    X = np.stack([I, Q], axis=1)
    
    # Targets
    # 1. OSNR (Continuous)
    # 2. Overlap (Binary). Threshold 35.2 GHz.
    # Assuming Spacing < 35.2 means Overlap (1).
    y_overlap = (Spacing < 35.2).astype(float)
    
    # Stack targets
    y = np.stack([OSNR, y_overlap], axis=1)
    
    # Split
    # Stratify by overlap to maintain class distribution
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y_overlap)
    
    # Normalize Inputs
    scaler_x = StandardScaler()
    X_train = scaler_x.fit_transform(X_train)
    X_test = scaler_x.transform(X_test)
    
    # Normalize OSNR (Target 0)
    # We don't normalize Binary target
    scaler_y_osnr = StandardScaler()
    y_train_osnr = scaler_y_osnr.fit_transform(y_train[:, 0].reshape(-1, 1)).flatten()
    y_test_osnr = scaler_y_osnr.transform(y_test[:, 0].reshape(-1, 1)).flatten()
    
    # Reassemble y
    y_train[:, 0] = y_train_osnr
    y_test[:, 0] = y_test_osnr
    
    # Convert to Tensor
    train_x = torch.tensor(X_train).float()
    train_y = torch.tensor(y_train).float()
    test_x = torch.tensor(X_test).float()
    test_y = torch.tensor(y_test).float()
    
    return train_x, train_y, test_x, test_y, scaler_x, scaler_y_osnr
