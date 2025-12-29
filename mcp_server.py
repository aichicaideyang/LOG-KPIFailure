#!/usr/bin/env python3
"""
MCP (Model Context Protocol) Server for Metric-Only Anomaly Detection.

This server exposes the anomaly detection functionality as MCP tools.

To run:
    python mcp_server.py

To use with Claude Desktop, add to your config:
{
    "mcpServers": {
        "metric_only_ad": {
            "command": "python",
            "args": ["/path/to/metric_only_ad/mcp_server.py"]
        }
    }
}
"""

import os
import sys
import json
import asyncio
from typing import Any

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import Tool, TextContent
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    print("Warning: MCP package not installed. Install with: pip install mcp")

from models.detector import AnomalyDetector
from utils.io import load_pkl, load_jsonl, save_json, save_pkl
import yaml


# Global detector instance
_detector = None
_config = None


def get_detector():
    """Get or create the detector instance."""
    global _detector, _config
    if _detector is None:
        config_path = os.path.join(os.path.dirname(__file__), 'config', 'default.yaml')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                _config = yaml.safe_load(f)
        else:
            _config = {
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
        _detector = AnomalyDetector(_config)
    return _detector, _config


async def train_model(train_data_path: str, model_save_path: str = None) -> dict:
    """
    Train the anomaly detection model.
    
    Args:
        train_data_path: Path to training samples (pickle file)
        model_save_path: Optional path to save the trained model
    
    Returns:
        Status message
    """
    detector, config = get_detector()
    
    if not os.path.exists(train_data_path):
        return {"error": f"Training data not found: {train_data_path}"}
    
    train_samples = load_pkl(train_data_path)
    detector.fit(train_samples)
    
    if model_save_path:
        os.makedirs(os.path.dirname(model_save_path) or '.', exist_ok=True)
        detector.save(model_save_path)
        return {"status": "success", "message": f"Model trained and saved to {model_save_path}", "samples": len(train_samples)}
    
    return {"status": "success", "message": "Model trained", "samples": len(train_samples)}


async def load_model(model_path: str) -> dict:
    """
    Load a pre-trained model.
    
    Args:
        model_path: Path to the model file
    
    Returns:
        Status message
    """
    detector, _ = get_detector()
    
    if not os.path.exists(model_path):
        return {"error": f"Model not found: {model_path}"}
    
    detector.load(model_path)
    return {"status": "success", "message": f"Model loaded from {model_path}"}


async def detect_anomalies(
    train_data_path: str,
    test_data_path: str,
    cases_path: str = None,
    output_path: str = None
) -> dict:
    """
    Detect anomalies in the data.
    
    Args:
        train_data_path: Path to training samples
        test_data_path: Path to test samples
        cases_path: Optional path to ground truth cases (JSONL)
        output_path: Optional path to save results
    
    Returns:
        Detection results with intervals and metrics
    """
    detector, _ = get_detector()
    
    if detector.model is None:
        return {"error": "Model not trained. Please train or load a model first."}
    
    if not os.path.exists(train_data_path):
        return {"error": f"Training data not found: {train_data_path}"}
    if not os.path.exists(test_data_path):
        return {"error": f"Test data not found: {test_data_path}"}
    
    train_samples = load_pkl(train_data_path)
    test_samples = load_pkl(test_data_path)
    
    anomaly_cases = None
    if cases_path and os.path.exists(cases_path):
        anomaly_cases = load_jsonl(cases_path)
    
    results = detector.detect(train_samples, test_samples, anomaly_cases, verbose=False)
    
    output = {
        "intervals": [(int(s), int(e)) for s, e in results['intervals']],
        "num_intervals": len(results['intervals']),
        "precision": results['precision'],
        "recall": results['recall'],
        "f1": results['f1'],
        "threshold": float(results['threshold'])
    }
    
    if output_path:
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        save_json(output_path, output)
        output["saved_to"] = output_path
    
    return output


async def get_config() -> dict:
    """Get current configuration."""
    _, config = get_detector()
    return config


async def update_config(param_path: str, value: Any) -> dict:
    """
    Update a configuration parameter.
    
    Args:
        param_path: Dot-separated path to parameter (e.g., 'downstream_param.AD.level')
        value: New value for the parameter
    
    Returns:
        Updated configuration
    """
    global _config, _detector
    _, config = get_detector()
    
    parts = param_path.split('.')
    obj = config
    for part in parts[:-1]:
        obj = obj[part]
    obj[parts[-1]] = value
    
    # Recreate detector with new config
    _detector = AnomalyDetector(_config)
    
    return {"status": "success", "updated": param_path, "value": value}


if MCP_AVAILABLE:
    # Create MCP server
    server = Server("metric_only_ad")

    @server.list_tools()
    async def list_tools():
        """List available tools."""
        return [
            Tool(
                name="train_model",
                description="Train the anomaly detection model on training data",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "train_data_path": {
                            "type": "string",
                            "description": "Path to training samples pickle file"
                        },
                        "model_save_path": {
                            "type": "string",
                            "description": "Optional path to save the trained model"
                        }
                    },
                    "required": ["train_data_path"]
                }
            ),
            Tool(
                name="load_model",
                description="Load a pre-trained anomaly detection model",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "model_path": {
                            "type": "string",
                            "description": "Path to the model file"
                        }
                    },
                    "required": ["model_path"]
                }
            ),
            Tool(
                name="detect_anomalies",
                description="Detect anomalies in time series data",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "train_data_path": {
                            "type": "string",
                            "description": "Path to training samples pickle file"
                        },
                        "test_data_path": {
                            "type": "string",
                            "description": "Path to test samples pickle file"
                        },
                        "cases_path": {
                            "type": "string",
                            "description": "Optional path to ground truth cases (JSONL file)"
                        },
                        "output_path": {
                            "type": "string",
                            "description": "Optional path to save detection results"
                        }
                    },
                    "required": ["train_data_path", "test_data_path"]
                }
            ),
            Tool(
                name="get_config",
                description="Get current model configuration",
                inputSchema={
                    "type": "object",
                    "properties": {}
                }
            ),
            Tool(
                name="update_config",
                description="Update a configuration parameter",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "param_path": {
                            "type": "string",
                            "description": "Dot-separated path to parameter (e.g., 'downstream_param.AD.level')"
                        },
                        "value": {
                            "description": "New value for the parameter"
                        }
                    },
                    "required": ["param_path", "value"]
                }
            )
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict):
        """Handle tool calls."""
        if name == "train_model":
            result = await train_model(
                arguments["train_data_path"],
                arguments.get("model_save_path")
            )
        elif name == "load_model":
            result = await load_model(arguments["model_path"])
        elif name == "detect_anomalies":
            result = await detect_anomalies(
                arguments["train_data_path"],
                arguments["test_data_path"],
                arguments.get("cases_path"),
                arguments.get("output_path")
            )
        elif name == "get_config":
            result = await get_config()
        elif name == "update_config":
            result = await update_config(
                arguments["param_path"],
                arguments["value"]
            )
        else:
            result = {"error": f"Unknown tool: {name}"}
        
        return [TextContent(type="text", text=json.dumps(result, indent=2))]


    async def main():
        """Run the MCP server."""
        async with stdio_server() as (read_stream, write_stream):
            await server.run(read_stream, write_stream, server.create_initialization_options())


    if __name__ == "__main__":
        asyncio.run(main())

else:
    # Fallback for when MCP is not available
    if __name__ == "__main__":
        print("MCP package not installed.")
        print("Install with: pip install mcp")
        print("\nYou can still use this module programmatically:")
        print("  from mcp_server import train_model, detect_anomalies")
        print("  await train_model('data/train.pkl', 'model.pkl')")
        print("  result = await detect_anomalies('data/train.pkl', 'data/test.pkl')")

