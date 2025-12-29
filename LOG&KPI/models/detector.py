"""
Anomaly Detector class that wraps all functionality.
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from datetime import datetime
from models.layers import MetricAutoRegressor, create_dataloader_metric_AR
from models.trainer import train_model, get_device


class AnomalyDetector:
    """
    Anomaly Detector for metric-only time series data.
    
    Usage:
        detector = AnomalyDetector(config)
        detector.fit(train_samples)
        results = detector.detect(test_samples, anomaly_cases)
    """
    
    def __init__(self, config):
        """
        Initialize the detector with configuration.
        
        Args:
            config: Dict with model_param and downstream_param
        """
        self.config = config
        self.model = None
        self.device = get_device()
        
    def fit(self, samples, verbose=True):
        """
        Train the model on samples.
        
        Args:
            samples: List of sample dicts with keys ['problem_id', 'timestamp', 'features']
            verbose: Print training progress
        """
        # Auto-detect dimensions from data
        sample = samples[0]
        instance_dim, channel_dim = sample['features'].shape
        
        model_config = self.config['model_param'].copy()
        model_config['instance_dim'] = instance_dim
        model_config['channel_dim'] = channel_dim
        
        if verbose:
            print(f"Auto-detected dimensions: {instance_dim} instances, {channel_dim} channels")
        
        self.model = train_model(samples, model_config, verbose)
        return self
    
    def save(self, path):
        """Save the trained model."""
        import pickle
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"Model saved to {path}")
    
    def load(self, path):
        """Load a trained model."""
        import pickle
        with open(path, 'rb') as f:
            self.model = pickle.load(f)
        print(f"Model loaded from {path}")
        return self
    
    def _compute_deviations(self, samples, method='num', t_value=3):
        """Compute system-level deviations for samples."""
        device = self.device
        model = self.model.to(device)
        mse = nn.MSELoss(reduction='none')
        
        dataloader = create_dataloader_metric_AR(
            samples, window_size=6, max_gap_ns=60*1e9,
            batch_size=128, shuffle=False,
            num_workers=4, pin_memory=torch.cuda.is_available()
        )
        
        if dataloader is None:
            return pd.DataFrame()
        
        results = []
        model.eval()
        with torch.no_grad():
            for batch_ts, batched_feats, batched_targets in dataloader:
                batched_feats = batched_feats.to(device)
                batched_targets = batched_targets.to(device)
                batched_targets_log = model.transform_target(batched_targets)
                
                z, h = model(batched_feats)
                loss = mse(h, batched_targets_log)
                
                if method == 'num':
                    instance_deviation = torch.sum(loss, dim=-1)
                    k = min(t_value, instance_deviation.shape[1])
                    topk_values, topk_indices = torch.topk(instance_deviation, k=k, dim=-1)
                    mask = torch.zeros_like(instance_deviation)
                    mask = mask.scatter_(1, topk_indices, 1).unsqueeze(-1)
                    system_deviation = torch.sum(loss * mask, dim=1)
                else:
                    system_deviation = torch.sum(loss, dim=(1, 2))
                
                for i, ts in enumerate(batch_ts.numpy()):
                    results.append({
                        'timestamp': ts,
                        'deviation': system_deviation[i].cpu().sum().item()
                    })
        
        return pd.DataFrame(results)
    
    @staticmethod
    def _parse_time_range(time_range_str):
        """Parse time range string to nanosecond timestamps."""
        parts = time_range_str.split(' ~ ')
        start_dt = datetime.strptime(parts[0].strip(), "%Y-%m-%d %H:%M:%S")
        end_dt = datetime.strptime(parts[1].strip(), "%Y-%m-%d %H:%M:%S")
        return int(start_dt.timestamp() * 1e9), int(end_dt.timestamp() * 1e9)
    
    @staticmethod
    def _get_threshold(values, q=0.1, level=0.90):
        """Compute threshold using SPOT or quantile."""
        try:
            from utils.spot import SPOT
            if len(set(values)) > 1:
                model = SPOT(q)
                model.fit(values, [])
                model.initialize(level=level, verbose=False)
                return model.extreme_quantile
        except:
            pass
        return np.percentile(values, level * 100)
    
    def detect(self, train_samples, test_samples, anomaly_cases=None, verbose=True):
        """
        Detect anomalies in the data.
        
        Args:
            train_samples: Training samples
            test_samples: Test samples to evaluate
            anomaly_cases: Optional list of ground truth cases for evaluation
            verbose: Print detailed results
        
        Returns:
            Dict with 'intervals', 'precision', 'recall', 'f1'
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        ad_config = self.config['downstream_param']['AD']
        split_ratio = ad_config.get('split_ratio', 0.6)
        method = ad_config.get('method', 'num')
        t_value = ad_config.get('t_value', 3)
        q = ad_config.get('q', 0.1)
        level = ad_config.get('level', 0.90)
        delay = ad_config.get('delay', 600)
        
        # Combine and sort samples
        total_samples = train_samples + test_samples
        total_samples.sort(key=lambda x: x['timestamp'])
        ts_list = [s['timestamp'] for s in total_samples]
        split_ts = ts_list[int(len(ts_list) * split_ratio)]
        
        # Get eval samples
        eval_samples = [s for s in total_samples if s['timestamp'] > split_ts]
        
        if verbose:
            print(f"Eval samples: {len(eval_samples)}")
        
        # Compute deviations
        deviations = self._compute_deviations(eval_samples, method, t_value)
        
        if len(deviations) == 0:
            return {'intervals': [], 'precision': 0, 'recall': 0, 'f1': 0}
        
        # Self-adaptive threshold
        baseline_idx = int(len(deviations) * 0.3)
        baseline_vals = deviations['deviation'].values[:baseline_idx]
        threshold = self._get_threshold(baseline_vals, q, level)
        
        if verbose:
            print(f"Threshold: {threshold:.4f}")
            print(f"Deviation range: [{deviations['deviation'].min():.4f}, {deviations['deviation'].max():.4f}]")
        
        # Mark outliers
        deviations['outlier'] = (deviations['deviation'] > threshold).astype(int)
        
        if verbose:
            outlier_ratio = deviations['outlier'].mean()
            print(f"Outlier ratio: {outlier_ratio:.4f}")
        
        # Get intervals
        intervals = self._get_intervals(deviations, delay * 1e9)
        
        # Evaluate if ground truth provided
        precision, recall, f1 = 0, 0, 0
        if anomaly_cases:
            filtered_cases = []
            for case in anomaly_cases:
                start_ns, _ = self._parse_time_range(case['time_range'])
                if start_ns > split_ts:
                    filtered_cases.append(case)
            
            precision, recall, f1 = self._evaluate(intervals, filtered_cases, verbose)
        
        return {
            'intervals': intervals,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'threshold': threshold,
            'deviations': deviations
        }
    
    @staticmethod
    def _get_intervals(deviations, delay_ns):
        """Extract continuous anomaly intervals."""
        outliers = deviations[deviations['outlier'] == 1].copy()
        if len(outliers) == 0:
            return []
        
        outliers = outliers.sort_values('timestamp')
        outliers['diff'] = [0] + np.diff(outliers['timestamp']).tolist()
        
        intervals = []
        start_ts, end_ts = None, None
        
        for _, row in outliers.iterrows():
            if start_ts is None:
                start_ts = int(row['timestamp'])
            if row['diff'] >= delay_ns:
                intervals.append((start_ts, end_ts))
                start_ts = int(row['timestamp'])
            end_ts = int(row['timestamp'])
        
        intervals.append((start_ts, end_ts))
        return [(s, e) for s, e in intervals if s != e]
    
    def _evaluate(self, pred_intervals, gt_cases, verbose=True):
        """Evaluate detection results against ground truth."""
        if not pred_intervals or not gt_cases:
            return 0, 0, 0
        
        # Parse ground truth
        gt_intervals = []
        for case in gt_cases:
            start_ns, end_ns = self._parse_time_range(case['time_range'])
            gt_intervals.append({
                'id': case['problem_id'],
                'start': start_ns,
                'end': end_ns
            })
        
        # Match
        pred_matched = {tuple(p): set() for p in pred_intervals}
        gt_matched = {gt['id']: set() for gt in gt_intervals}
        
        for pred_s, pred_e in pred_intervals:
            for gt in gt_intervals:
                if not (gt['start'] > pred_e or gt['end'] < pred_s):
                    pred_matched[(pred_s, pred_e)].add(gt['id'])
                    gt_matched[gt['id']].add((pred_s, pred_e))
        
        TP = sum(1 for v in gt_matched.values() if v)
        FP = sum(1 for v in pred_matched.values() if not v)
        FN = sum(1 for v in gt_matched.values() if not v)
        
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        if verbose:
            print(f"TP={TP}, FP={FP}, FN={FN}")
            print(f"Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")
        
        return round(precision, 4), round(recall, 4), round(f1, 4)

