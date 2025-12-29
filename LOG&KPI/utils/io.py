"""I/O utility functions."""

import json
import pickle


def load_pkl(filepath):
    """Load pickle file."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def save_pkl(filepath, data):
    """Save pickle file."""
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)


def load_json(filepath):
    """Load JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(filepath, data, indent=2):
    """Save JSON file."""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def load_jsonl(filepath):
    """Load JSONL file (one JSON per line)."""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

