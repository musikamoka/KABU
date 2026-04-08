<img width="1887" height="909" alt="image" src="https://github.com/user-attachments/assets/627f41be-4575-48ce-bc5e-12a056946bff" />
<img width="1662" height="813" alt="image" src="https://github.com/user-attachments/assets/b18d2ea9-cdda-4df6-89ec-4db2c02f98c1" />


<div align="center">

<img src="figures/logo.png" width="120" alt="KRONOS Logo"/>

# KRONOS Terminal

**A professional K-line forecasting terminal powered by the Kronos Foundation Model**

[![Model](https://img.shields.io/badge/Model-Kronos--base%20102.3M-f39c12?style=flat-square)](https://huggingface.co/NeoQuasar/Kronos-base)
[![Python](https://img.shields.io/badge/Python-3.10+-3498db?style=flat-square)](https://www.python.org)
[![License](https://img.shields.io/badge/License-MIT-2ecc71?style=flat-square)](LICENSE)
[![AAAI](https://img.shields.io/badge/AAAI-2026-e74c3c?style=flat-square)](https://arxiv.org/abs/2508.02739)

[中文](#中文) · [English](#english) · [日本語](#日本語)

</div>

---

## English

### What is this?

KRONOS Terminal is a local web application that wraps the **Kronos Foundation Model** — the first open-source foundation model for financial candlestick (K-line) data, pre-trained on 12 billion+ K-line records from 45 global exchanges and accepted at **AAAI 2026**.

This tool provides a Bloomberg-style terminal interface where you can:
- Input any stock ticker supported by Yahoo Finance
- Run Kronos-base (102.3M parameters) on your local GPU
- Visualize historical + predicted K-lines with MACD, RSI, Bollinger Bands
- Zoom, pan, and interact with the chart in real time

### Quick Start

**Requirements:** Anaconda, NVIDIA GPU (optional but recommended)

```bash
# 1. Clone both repos
git clone https://github.com/shiyu-coder/Kronos
git clone https://github.com/YOUR_USERNAME/kronos-terminal
cp kronos-terminal/* Kronos/

# 2. Setup conda environment
conda create -n kronos python=3.10
conda activate kronos
cd Kronos
pip install -r requirements.txt
pip install flask flask-cors yfinance matplotlib

# 3. Install PyTorch with CUDA (check nvidia-smi for your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 4. Launch (Windows)
double-click start.bat

# 4. Launch (macOS / Linux)
bash start.sh
```

Select your language → two servers start automatically → browser opens.

### Architecture

```
start.bat / start.sh
    ├── kronos_server.py  (Flask API · port 5000 · conda kronos env)
    │       └── Kronos-base model (HuggingFace: NeoQuasar/Kronos-base)
    └── python -m http.server 8080
            └── kronos_terminal.html  (Browser UI)
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| History Window | 60 days | How many trading days of data to feed the model |
| Forecast Days | 5 days | Number of days to predict (≤10 recommended) |
| Samples | 50 | Monte Carlo paths — more = more stable, slower |
| Temperature | 0.70 | Sampling temperature — lower = more conservative |

### Disclaimer

This tool is for **academic research and technical demonstration only**. Predictions do not constitute investment advice. Financial markets involve significant uncertainty.

---

## 中文

### 这是什么？

KRONOS Terminal 是一个本地 Web 应用，集成了 **Kronos 基础模型** —— 首个面向金融K线图的开源基础模型，基于全球 45 家交易所超过 120 亿条K线记录预训练，已被 **AAAI 2026** 接收。

本工具提供彭博终端风格的界面，功能包括：
- 输入任意 Yahoo Finance 支持的股票代码
- 在本地 GPU 上运行 Kronos-base（102.3M 参数）
- 可视化历史 + 预测K线，含 MACD、RSI、布林带
- 图表缩放、平移、十字线实时交互

### 快速开始

**环境要求：** Anaconda、NVIDIA GPU（可选但推荐）

```bash
# 1. 克隆仓库
git clone https://github.com/shiyu-coder/Kronos
cd Kronos

# 将本项目文件复制到 Kronos 目录下

# 2. 创建 conda 环境
conda create -n kronos python=3.10
conda activate kronos
pip install -r requirements.txt
pip install flask flask-cors yfinance matplotlib

# 3. 安装 PyTorch（根据 nvidia-smi 显示的 CUDA 版本选择）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 4. 一键启动（Windows）
双击 start.bat，选择语言，自动打开浏览器
```

### 注意事项

- 首次运行会从 HuggingFace 下载约 **409MB** 的模型权重
- 如果下载慢，可设置镜像：`set HF_ENDPOINT=https://hf-mirror.com`
- 预测天数建议 **≤ 10 天**，越短越准确
- 红涨绿跌配色（A股风格）

### 免责声明

本工具仅供学术研究和技术演示，预测结果**不构成投资建议**。金融市场存在高度不确定性，请独立做出投资决策。

---

## 日本語

### これは何？

KRONOS Terminal は **Kronos 基盤モデル** を利用したローカル Web アプリです。世界 45 の取引所から 120 億件以上の K 線データで学習した金融 K 線特化型モデルで、**AAAI 2026** に採択されています。

### クイックスタート

```bash
git clone https://github.com/shiyu-coder/Kronos
cd Kronos
conda create -n kronos python=3.10
conda activate kronos
pip install -r requirements.txt
pip install flask flask-cors yfinance matplotlib
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 起動（Windows）
start.bat をダブルクリック → 言語選択 → 自動でブラウザが開きます
```

### 免責事項

本ツールは学術研究・技術デモ目的のみです。予測結果は投資助言ではありません。

---

## Credits

- **Kronos Model** — [shiyu-coder/Kronos](https://github.com/shiyu-coder/Kronos) · [Paper](https://arxiv.org/abs/2508.02739)
- **Data Source** — [Yahoo Finance](https://finance.yahoo.com) via `yfinance`
- **Pre-trained Weights** — [NeoQuasar/Kronos-base](https://huggingface.co/NeoQuasar/Kronos-base)

## License

MIT License — see [LICENSE](LICENSE) for details.

> ⚠️ **Important:** The Kronos model weights are subject to the original repository's license. This terminal is a UI wrapper and is not affiliated with the official Kronos project.
