"""
Neural network layers for metric-only anomaly detection.

Optimized for large instance counts by avoiding O(n²) attention.
Uses MLP + GRU instead of Transformer + GNN.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np


class MetricExtractor(nn.Module):
    """
    Lightweight feature extractor for metric-only data.
    Uses MLP + GRU to extract temporal patterns efficiently.
    Includes input normalization to handle large value ranges.
    """
    
    def __init__(self, instance_dim, channel_dim, gru_hidden_dim, 
                 dropout=0.3, gru_layers=1):
        super(MetricExtractor, self).__init__()
        self.instance_dim = instance_dim
        self.channel_dim = channel_dim
        
        # Input normalization (LayerNorm on channel dimension)
        self.input_norm = nn.LayerNorm(channel_dim)
        
        # MLP for channel mixing
        self.channel_mixer = nn.Sequential(
            nn.Linear(channel_dim, channel_dim * 4),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(channel_dim * 4, channel_dim),
            nn.LeakyReLU()
        )
        
        # GRU for temporal encoding
        self.GRUEncoder = nn.GRU(
            channel_dim, gru_hidden_dim, gru_layers, 
            bias=False, batch_first=True, 
            dropout=dropout if gru_layers > 1 else 0
        )
        
        # Feature projection
        self.feature_proj = nn.Linear(gru_hidden_dim, gru_hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, features):
        """
        Args:
            features: (batch_size, seq_len, instance_dim, channel_dim)
        Returns:
            h: (batch_size, instance_dim, gru_hidden_dim)
        """
        batch_size, seq_len, instance_dim, channel_dim = features.shape
        
        # Log transform for large values, then normalize
        h = torch.log1p(features.abs()) * features.sign()
        h = self.input_norm(h)
        
        # Apply channel mixer
        h = h.reshape(-1, channel_dim)
        h = self.channel_mixer(h)
        
        # Reshape for GRU
        h = h.view(batch_size, seq_len, instance_dim, channel_dim)
        h = h.permute(0, 2, 1, 3)  # (B, I, T, C)
        
        # Apply GRU for temporal encoding per instance
        h = h.reshape(-1, seq_len, channel_dim)  # (B*I, T, C)
        output, h_n = self.GRUEncoder(h)
        h = F.leaky_relu(h_n[-1])  # (B*I, gru_hidden_dim)
        
        # Reshape to (batch_size, instance_dim, gru_hidden_dim)
        h = h.view(batch_size, instance_dim, -1)
        h = self.dropout(self.feature_proj(h))
        
        return h


class MetricRegressor(nn.Module):
    """MLP regressor for prediction."""
    
    def __init__(self, in_dim, out_dim):
        super(MetricRegressor, self).__init__()
        self.mlp = nn.Linear(in_dim, out_dim)
        
    def forward(self, features):
        return F.leaky_relu(self.mlp(features))


class MetricAutoRegressor(nn.Module):
    """
    Auto-regressor model for metric-only anomaly detection.
    Predicts the next time step's features based on previous time steps.
    """
    
    def __init__(self, instance_dim, channel_dim, gru_hidden_dim=32, 
                 dropout=0.3, gru_layers=1, **kwargs):
        super(MetricAutoRegressor, self).__init__()
        self.extractor = MetricExtractor(
            instance_dim, channel_dim, gru_hidden_dim, dropout, gru_layers
        )
        self.regressor = MetricRegressor(gru_hidden_dim, channel_dim)
        
    def forward(self, features):
        """
        Args:
            features: (batch_size, seq_len, instance_dim, channel_dim)
        Returns:
            z: extracted features (batch_size, instance_dim, gru_hidden_dim)
            h: predicted features in log space (batch_size, instance_dim, channel_dim)
        """
        z = self.extractor(features)
        h = self.regressor(z)
        return z, h
    
    @staticmethod
    def transform_target(targets):
        """Transform targets to log space for loss computation."""
        return torch.log1p(targets.abs()) * targets.sign()


def collate_metric_AR(samples):
    """Collate function for metric-only auto-regressor."""
    timestamps, feats, targets = map(list, zip(*samples))
    return torch.stack(timestamps), torch.stack(feats), torch.stack(targets)


def create_dataloader_metric_AR(samples, window_size=6, max_gap_ns=60*1e9, 
                                 batch_size=128, shuffle=False, 
                                 num_workers=0, pin_memory=False):
    """
    Create a DataLoader for metric-only auto-regressor.
    
    Args:
        samples: List of sample dicts with keys ['problem_id', 'timestamp', 'features']
        window_size: Number of time steps in each sequence
        max_gap_ns: Maximum gap between consecutive samples in nanoseconds
        batch_size: Batch size
        shuffle: Whether to shuffle the data
        num_workers: Number of workers for data loading
        pin_memory: Whether to pin memory for faster GPU transfer
    
    Returns:
        DataLoader object
    """
    # Sort samples by timestamp
    samples = sorted(samples, key=lambda x: x['timestamp'])
    
    # Group samples by problem_id
    problem_groups = {}
    for sample in samples:
        pid = sample['problem_id']
        if pid not in problem_groups:
            problem_groups[pid] = []
        problem_groups[pid].append(sample)
    
    # Create sliding windows within each problem group
    series_samples = []
    for pid, group in problem_groups.items():
        group = sorted(group, key=lambda x: x['timestamp'])
        for i in range(len(group) - window_size + 1):
            window = group[i:i + window_size]
            valid = all(
                abs(window[j+1]['timestamp'] - window[j]['timestamp']) <= max_gap_ns
                for j in range(len(window) - 1)
            )
            if valid:
                series_samples.append(window)
    
    if len(series_samples) == 0:
        print(f"Warning: No valid series samples created. Total samples: {len(samples)}")
        return None
    
    # Create dataset
    dataset = []
    for series_sample in series_samples:
        timestamp = torch.tensor(series_sample[-1]['timestamp'])
        feats_seq = torch.stack([
            torch.tensor(step['features'], dtype=torch.float32) 
            for step in series_sample[:-1]
        ])
        target = torch.tensor(series_sample[-1]['features'], dtype=torch.float32)
        dataset.append([timestamp, feats_seq, target])
    
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        collate_fn=collate_metric_AR,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

