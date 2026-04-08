<div align="center">

```
██╗  ██╗██████╗  ██████╗ ███╗   ██╗ ██████╗ ███████╗
██║ ██╔╝██╔══██╗██╔═══██╗████╗  ██║██╔═══██╗██╔════╝
█████╔╝ ██████╔╝██║   ██║██╔██╗ ██║██║   ██║███████╗
██╔═██╗ ██╔══██╗██║   ██║██║╚██╗██║██║   ██║╚════██║
██║  ██╗██║  ██║╚██████╔╝██║ ╚████║╚██████╔╝███████║
╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═══╝ ╚═════╝ ╚══════╝
```

**KRONOS Market Intelligence**

AI-powered stock & forex candlestick forecasting system

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.11-ee4c2c?logo=pytorch)](https://pytorch.org)
[![Flask](https://img.shields.io/badge/Flask-3.1-black?logo=flask)](https://flask.palletsprojects.com)
[![CUDA](https://img.shields.io/badge/CUDA-12.8-76b900?logo=nvidia)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

[中文](#中文) · [English](#english) · [日本語](#日本語)

</div>

---

## 中文

### 简介

KRONOS 是基于 [Kronos 时序基础模型](https://www.zdoc.app/zh/shiyu-coder/Kronos) 构建的本地股票预测系统。通过 GUI 启动器一键启动，在浏览器中以交互式 K 线图查看 AI 预测结果。

### 功能特性

- 🤖 **AI 预测** — Kronos-mini / small / base 三档模型，概率采样生成预测 K 线
- 📊 **交互式图表** — 烛台图 + MA5/20/60 + 布林带 + MACD + RSI + 成交量，支持拖拽/缩放
- 🖥️ **GUI 启动器** — tkinter 界面，自动检测 conda、安装依赖、启动服务
- 🌐 **多语言** — 中文 / English / 日本語 界面切换
- 🔒 **本地运行** — 全程离线推理，数据不上传任何服务器

### 快速开始

#### 1. 环境要求
- Windows 10/11
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) 或 Anaconda
- NVIDIA GPU（推荐，CPU 也可运行）

#### 2. 创建 conda 环境并安装依赖

```bash
conda create -n kronos python=3.10 -y
conda activate kronos
pip install flask flask-cors yfinance torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install pandas numpy matplotlib huggingface-hub safetensors einops
```

#### 3. 下载 Kronos 模型

```bash
# 模型来源
# https://www.zdoc.app/zh/shiyu-coder/Kronos
# 将模型仓库克隆到本地，例如 D:\desk\KABUYOSO\Kronos
```

#### 4. 配置并启动

编辑 `kronos_config.ini`：

```ini
CONDA_ENV=kronos
MODEL_DIR=D:/path/to/Kronos
LANG=zh
```

双击运行 `kronos_launcher.py`，点击 **启动** 即可。

### 文件结构

```
KRONOS/
├── kronos_launcher.py      # GUI 启动器
├── kronos_server.py        # Flask REST API 服务器
├── kronos_terminal.html    # 浏览器交互图表 UI
├── kronos_predictor.py     # CLI 预测工具（含 matplotlib 图表）
├── kronos_config.ini       # 配置文件
├── environment.txt         # 依赖包清单
└── .kronos_lang            # 语言传递文件（自动生成）
```

### API 接口

| 端点 | 方法 | 说明 |
|---|---|---|
| `/api/health` | GET | 健康检查 |
| `/api/config` | GET | 当前配置 |
| `/api/predict` | POST | K 线预测 |
| `/api/quote` | GET | 历史行情 |

---

## English

### Overview

KRONOS is a local stock forecasting system built on the [Kronos time-series foundation model](https://www.zdoc.app/zh/shiyu-coder/Kronos). Launch with one click via the GUI launcher, then view AI-predicted candlesticks in an interactive browser chart.

### Features

- 🤖 **AI Forecast** — Kronos-mini / small / base models with probabilistic sampling
- 📊 **Interactive Chart** — Candlestick + MA + Bollinger Bands + MACD + RSI + Volume with drag/zoom
- 🖥️ **GUI Launcher** — Auto-detects conda, installs dependencies, starts services
- 🌐 **Multi-language** — Chinese / English / Japanese UI
- 🔒 **Fully Local** — All inference runs locally, no data sent to external servers

### Quick Start

```bash
# 1. Create environment
conda create -n kronos python=3.10 -y
conda activate kronos

# 2. Install dependencies
pip install flask flask-cors yfinance torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install pandas numpy matplotlib huggingface-hub safetensors einops

# 3. Edit kronos_config.ini, then run:
python kronos_launcher.py
```

---

## 日本語

### 概要

KRONOS は [Kronos 時系列基盤モデル](https://www.zdoc.app/zh/shiyu-coder/Kronos) を使った株式・為替予測システムです。GUI ランチャーからワンクリックで起動し、ブラウザ上でインタラクティブなローソク足チャートで予測結果を確認できます。

### 特徴

- 🤖 **AI 予測** — Kronos-mini / small / base モデル、確率的サンプリングで将来 K 線を生成
- 📊 **インタラクティブチャート** — ローソク足・MA・ボリンジャーバンド・MACD・RSI・出来高、ドラッグ/ズーム対応
- 🖥️ **GUI ランチャー** — conda 自動検出・依存関係インストール・サービス起動を 1 クリックで完結
- 🌐 **多言語対応** — 中国語 / 英語 / 日本語 切り替え
- 🔒 **ローカル完結** — 推論はすべてローカルで実行、データは外部に送信されない

### クイックスタート

```bash
# 1. 仮想環境作成
conda create -n kronos python=3.10 -y
conda activate kronos

# 2. 依存関係インストール
pip install flask flask-cors yfinance torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install pandas numpy matplotlib huggingface-hub safetensors einops

# 3. kronos_config.ini を編集後、起動：
python kronos_launcher.py
```

### 動作確認済み環境

| 項目 | 内容 |
|---|---|
| OS | Windows 11 |
| GPU | NVIDIA GeForce RTX 5070 Ti (16 GB VRAM) |
| CUDA | 12.8 (Driver 591.74) |
| Python | 3.10 |
| PyTorch | 2.11.0+cu128 |

---

## Screenshots

<div align="center">

**Interactive Chart (Japanese UI)**

![KRONOS Chart](demo.png)

**GUI Launcher**

![KRONOS Launcher](demoUI.png)

</div>

---

## Model Credits

Kronos model by [shiyu-coder](https://www.zdoc.app/zh/shiyu-coder/Kronos).
This repository provides only the launcher, API server, and UI — model weights are downloaded separately from HuggingFace.

| Model | HuggingFace |
|---|---|
| Kronos-mini | `NeoQuasar/Kronos-mini` |
| Kronos-small | `NeoQuasar/Kronos-small` |
| Kronos-base | `NeoQuasar/Kronos-base` |
| Tokenizer (2k) | `NeoQuasar/Kronos-Tokenizer-2k` |
| Tokenizer (base) | `NeoQuasar/Kronos-Tokenizer-base` |

---

## License

MIT License — see [LICENSE](LICENSE) for details.

> ⚠️ **免責事項 / Disclaimer / 免责声明**
> 本システムは教育・研究目的です。投資判断には使用しないでください。
> For educational and research purposes only. Not financial advice.
> 本系统仅供教育和研究目的，不构成投资建议。
