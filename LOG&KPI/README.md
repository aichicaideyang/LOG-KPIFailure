# Metric-Only Anomaly Detection

基于纯指标的时间序列异常检测工具，改编自ART模型，移除了图神经网络部分，专注于指标数据的异常检测。

## 特点

- **无图依赖**：不需要服务调用图，只使用时间序列指标
- **GPU加速**：支持CUDA加速训练和推理
- **自适应阈值**：使用SPOT算法自动确定异常阈值
- **MCP工具**：支持作为MCP服务器使用

## 目录结构

```
metric_only_ad/
├── config/
│   └── default.yaml      # 默认配置文件
├── models/
│   ├── __init__.py
│   ├── layers.py         # 神经网络层
│   ├── trainer.py        # 训练模块
│   └── detector.py       # 异常检测器
├── utils/
│   ├── __init__.py
│   ├── io.py             # I/O工具
│   └── spot.py           # SPOT算法
├── data/                 # 数据目录
├── results/              # 结果输出目录
├── main.py               # 命令行入口
├── mcp_server.py         # MCP服务器
├── requirements.txt      # 依赖
└── README.md
```

## 安装

```bash
# 安装依赖
pip install -r requirements.txt

# 可选：安装MCP支持
pip install mcp
```

## 使用方法

### 1. 命令行方式

```bash
# 使用默认配置
python main.py --train data/train_samples.pkl --test data/test_samples.pkl

# 指定配置文件
python main.py --config config/default.yaml

# 完整参数
python main.py \
    --train data/train_samples.pkl \
    --test data/test_samples.pkl \
    --cases data/B榜题目.jsonl \
    --model results/model.pkl \
    --output results/
```

### 2. Python API

```python
from models.detector import AnomalyDetector
from utils.io import load_pkl, load_jsonl
import yaml

# 加载配置
with open('config/default.yaml', 'r') as f:
    config = yaml.safe_load(f)

# 加载数据
train_samples = load_pkl('data/train_samples.pkl')
test_samples = load_pkl('data/test_samples.pkl')
anomaly_cases = load_jsonl('data/B榜题目.jsonl')

# 创建检测器
detector = AnomalyDetector(config)

# 训练模型
detector.fit(train_samples)

# 保存/加载模型
detector.save('model.pkl')
# detector.load('model.pkl')

# 检测异常
results = detector.detect(train_samples, test_samples, anomaly_cases)

print(f"检测到 {len(results['intervals'])} 个异常时段")
print(f"Precision: {results['precision']:.4f}")
print(f"Recall: {results['recall']:.4f}")
print(f"F1: {results['f1']:.4f}")
```

### 3. MCP工具

作为MCP服务器运行：

```bash
python mcp_server.py
```

在Claude Desktop配置中添加：

```json
{
    "mcpServers": {
        "metric_only_ad": {
            "command": "python",
            "args": ["/path/to/metric_only_ad/mcp_server.py"]
        }
    }
}
```

可用的MCP工具：
- `train_model`: 训练模型
- `load_model`: 加载已训练的模型
- `detect_anomalies`: 检测异常
- `get_config`: 获取配置
- `update_config`: 更新配置参数

## 数据格式

### 样本格式 (pickle)

```python
[
    {
        'problem_id': '002',
        'timestamp': 1758064233000000000,  # 纳秒级时间戳
        'features': np.array([...])        # shape: (instance_dim, channel_dim)
    },
    ...
]
```

### 异常标签格式 (JSONL)

```json
{"problem_id": "002", "time_range": "2025-09-16 23:20:33 ~ 2025-09-16 23:30:33", ...}
{"problem_id": "003", "time_range": "2025-09-16 23:44:03 ~ 2025-09-17 00:00:05", ...}
```

## 配置参数

主要参数说明：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `model_param.gru_hidden_dim` | 32 | GRU隐藏层维度 |
| `model_param.epochs` | 100 | 训练轮数 |
| `model_param.batch_size` | 128 | 批大小 |
| `downstream_param.AD.level` | 0.90 | 阈值分位数 |
| `downstream_param.AD.delay` | 600 | 异常合并间隔(秒) |

## 模型架构

```
Input (B, T, I, C)
    ↓
Log Transform + LayerNorm
    ↓
Channel MLP Mixer
    ↓
GRU Encoder
    ↓
Feature Projection
    ↓
Regressor (预测下一时刻特征)
```

- B: batch_size
- T: 时间窗口长度 (默认5)
- I: 实例数量
- C: 特征通道数

## License

MIT

