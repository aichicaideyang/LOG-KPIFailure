#!/usr/bin/env python3
"""
Main entry point for Metric-Only Anomaly Detection.

Usage:
    python main.py --config config/default.yaml
    python main.py --train data/train_samples.pkl --test data/test_samples.pkl
"""

import os
import sys
import argparse
import yaml

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.detector import AnomalyDetector
from utils.io import load_pkl, save_pkl, load_jsonl, save_json


def main():
    parser = argparse.ArgumentParser(description='Metric-Only Anomaly Detection')
    parser.add_argument('--config', type=str, default='config/default.yaml',
                        help='Path to configuration file')
    parser.add_argument('--train', type=str, help='Path to training samples')
    parser.add_argument('--test', type=str, help='Path to test samples')
    parser.add_argument('--cases', type=str, help='Path to anomaly cases (JSONL)')
    parser.add_argument('--model', type=str, help='Path to save/load model')
    parser.add_argument('--output', type=str, default='results',
                        help='Output directory for results')
    args = parser.parse_args()
    
    # Load configuration
    config_path = args.config
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"Loaded config from {config_path}")
    else:
        print(f"Config file not found: {config_path}")
        config = {
            'model_param': {
                'gru_hidden_dim': 32,
                'gru_layers': 1,
                'dropout': 0.3,
                'epochs': 100,
                'batch_size': 128,
                'learning_rate': 0.001
            },
            'downstream_param': {
                'AD': {
                    'split_ratio': 0.6,
                    'method': 'num',
                    't_value': 3,
                    'q': 0.1,
                    'level': 0.90,
                    'delay': 600
                }
            }
        }
    
    # Resolve data paths
    train_path = args.train or config.get('path', {}).get('train_samples', 'data/train_samples.pkl')
    test_path = args.test or config.get('path', {}).get('test_samples', 'data/test_samples.pkl')
    cases_path = args.cases or config.get('path', {}).get('case_path', 'data/B榜题目.jsonl')
    model_path = args.model or os.path.join(args.output, 'model.pkl')
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Load data
    print("\n" + "="*60)
    print("Loading data...")
    print("="*60)
    
    train_samples = load_pkl(train_path)
    test_samples = load_pkl(test_path)
    print(f"  Train samples: {len(train_samples)}")
    print(f"  Test samples: {len(test_samples)}")
    
    anomaly_cases = None
    if os.path.exists(cases_path):
        anomaly_cases = load_jsonl(cases_path)
        print(f"  Anomaly cases: {len(anomaly_cases)}")
    
    # Initialize detector
    detector = AnomalyDetector(config)
    
    # Train or load model
    print("\n" + "="*60)
    print("Model Training/Loading...")
    print("="*60)
    
    if os.path.exists(model_path):
        detector.load(model_path)
    else:
        detector.fit(train_samples)
        detector.save(model_path)
    
    # Detect anomalies
    print("\n" + "="*60)
    print("Anomaly Detection...")
    print("="*60)
    
    results = detector.detect(train_samples, test_samples, anomaly_cases)
    
    # Save results
    result_path = os.path.join(args.output, 'results.json')
    save_json(result_path, {
        'intervals': [(int(s), int(e)) for s, e in results['intervals']],
        'precision': results['precision'],
        'recall': results['recall'],
        'f1': results['f1'],
        'threshold': float(results['threshold'])
    })
    
    print("\n" + "="*60)
    print("Results Summary")
    print("="*60)
    print(f"  Predicted intervals: {len(results['intervals'])}")
    print(f"  Precision: {results['precision']:.4f}")
    print(f"  Recall: {results['recall']:.4f}")
    print(f"  F1 Score: {results['f1']:.4f}")
    print(f"\nResults saved to: {result_path}")


if __name__ == '__main__':
    main()

