"""
Training module for metric-only anomaly detection model.
Supports GPU acceleration.
"""

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from models.layers import MetricAutoRegressor, create_dataloader_metric_AR


def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    return device


def train_model(samples, config, verbose=True):
    """
    Train the metric-only auto-regressor model with GPU support.
    
    Args:
        samples: List of sample dicts with keys ['problem_id', 'timestamp', 'features']
        config: Configuration dict with model parameters
        verbose: Print training progress
    
    Returns:
        Trained model (on CPU for compatibility)
    """
    device = get_device()
    
    # Hyperparameters
    batch_size = config.get('batch_size', 128)
    epochs = config.get('epochs', 100)
    instance_dim = config['instance_dim']
    channel_dim = config['channel_dim']
    gru_hidden_dim = config.get('gru_hidden_dim', 32)
    gru_layers = config.get('gru_layers', 1)
    dropout = config.get('dropout', 0.3)
    learning_rate = config.get('learning_rate', 0.001)
    
    PATIENCE = 10
    early_stop_threshold = 1e-4
    prev_loss = np.inf
    stop_count = 0
    
    # Model initialization
    model = MetricAutoRegressor(
        instance_dim=instance_dim,
        channel_dim=channel_dim,
        gru_hidden_dim=gru_hidden_dim,
        dropout=dropout,
        gru_layers=gru_layers
    ).to(device)
    
    best_state_dict = model.state_dict()
    
    # Loss and optimizer
    Loss = nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.1, patience=5)
    
    # Create dataloader
    dataloader = create_dataloader_metric_AR(
        samples, 
        window_size=6, 
        max_gap_ns=60 * 1e9,
        batch_size=batch_size, 
        shuffle=True,
        num_workers=4,
        pin_memory=torch.cuda.is_available()
    )
    
    if dataloader is None:
        print("Error: Could not create dataloader.")
        return None
    
    if verbose:
        print(f"Training with {len(dataloader)} batches per epoch on {device}")
    
    # Training loop
    iterator = tqdm(range(epochs), desc="Training") if verbose else range(epochs)
    for epoch in iterator:
        model.train()
        running_loss = []
        
        for batch_ts, batched_feats, batched_targets in dataloader:
            batched_feats = batched_feats.to(device)
            batched_targets = batched_targets.to(device)
            batched_targets_log = model.transform_target(batched_targets)
            
            opt.zero_grad()
            z, h = model(batched_feats)
            loss = Loss(h, batched_targets_log)
            loss.backward()
            opt.step()
            running_loss.append(loss.item())
        
        epoch_loss = np.mean(running_loss)
        
        # Early stopping
        if prev_loss - epoch_loss < early_stop_threshold:
            stop_count += 1
            if stop_count == PATIENCE:
                if verbose:
                    print(f'\nEarly stopping at epoch {epoch}')
                model.load_state_dict(best_state_dict)
                break
        else:
            best_state_dict = model.state_dict()
            stop_count = 0
            prev_loss = epoch_loss
        
        if verbose and epoch % 10 == 0:
            print(f'\nEpoch {epoch} loss: {epoch_loss:.6f}')
        
        scheduler.step(epoch_loss)
    
    model.load_state_dict(best_state_dict)
    model = model.cpu()
    
    if verbose:
        print(f"Training completed. Final loss: {prev_loss:.6f}")
    
    return model

