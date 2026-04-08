<div align="center">

![KRONOS](demo.png)

# KRONOS Market Intelligence

**株価・為替の AI 予測をブラウザで — ワンクリック起動**

[中文](#中文简介) · [日本語](#日本語紹介) · [English](#english-introduction)

[![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-3.1-black?logo=flask)](https://flask.palletsprojects.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.11_CUDA12.8-ee4c2c?logo=pytorch)](https://pytorch.org)
[![Platform](https://img.shields.io/badge/Platform-Windows-0078D6?logo=windows)](https://www.microsoft.com/windows)
[![License](https://img.shields.io/badge/License-MIT-22c55e)](LICENSE)

</div>

---

## これは何？ / What is this? / 这是什么？

元の [Kronos モデル](https://github.com/shiyu-coder/Kronos) は Python ライブラリとして提供されており、**CLI での利用が前提**でした。

このプロジェクトはそのモデルを**一般ユーザーが使いやすい形で包み直したアプリケーション**です。

| 元の Kronos | このプロジェクト |
|---|---|
| Python ライブラリ / CLI | デスクトップ GUI + Web アプリ |
| コードを書く必要あり | ボタン 1 つで起動 |
| テキスト出力 | リアルタイムインタラクティブチャート |
| 単一言語 | 中文 / English / 日本語 切り替え |
| サーバー機能なし | REST API 付き（外部連携可能） |

---

## 日本語紹介

### スクリーンショット

| チャート画面 | GUI ランチャー |
|---|---|
| ![Chart](demo.png) | ![Launcher](demoUI.png) |

### 主な機能

**🖥️ GUI ランチャー（`kronos_launcher.py`）**
- conda 仮想環境を自動検出・依存パッケージを自動インストール
- ワンクリックでサーバー起動 → ブラウザ自動オープン
- IPv4 / IPv6 デュアルスタック対応のヘルスチェック（60 秒リトライ）
- 起動ログをリアルタイム表示

**📊 インタラクティブチャート（`kronos_terminal.html`）**
- Canvas API でゼロから実装したローソク足エンジン
- MA5 / MA20 / MA60 / ボリンジャーバンド / MACD / RSI / 出来高
- ドラッグ移動・スクロールズーム・ダブルクリックリセット
- ホバーで OHLC + インジケーター値をリアルタイム表示
- 予測ゾーンを点線で区切って視覚的に分離

**🔌 REST API サーバー（`kronos_server.py`）**
- `/api/predict` — Yahoo Finance からデータ取得 → 推論 → 結果返却
- `/api/health` — IPv4 / IPv6 両対応ヘルスチェック
- `/api/config` — 現在のサーバー設定確認
- `/api/quote` — 予測なしの行情取得
- モデルはキャッシュされ 2 回目以降の推論が高速

**📈 CLI 予測ツール（`kronos_predictor.py`）**
- matplotlib による高品質 K 線チャート（ダークテーマ）
- RSI / MACD / ボリンジャーバンド の技術指標シグナル自動サマリー
- 多サンプル中央値集計・FP16 混合精度推論対応

### セットアップ

**1. 必要環境**
- Windows 10 / 11
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
- NVIDIA GPU（推奨）または CPU

**2. 仮想環境作成 & 依存インストール**

```bash
conda create -n kronos python=3.10 -y
conda activate kronos

# GPU（CUDA 12.8）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# 共通依存
pip install flask flask-cors yfinance pandas numpy matplotlib \
            huggingface-hub safetensors einops
```

**3. Kronos モデルのダウンロード**

```bash
git clone https://github.com/shiyu-coder/Kronos.git
# → model/ ディレクトリが含まれる場所のパスを控えておく
```

**4. 設定**

`kronos_config.ini` を編集：

```ini
CONDA_ENV=kronos
MODEL_DIR=D:/path/to/Kronos   # clone した場所
LANG=ja                        # zh / en / ja
```

**5. 起動**

```bash
python kronos_launcher.py
```

GUI が開いたら「起動」をクリック → ブラウザが自動で開きます。

### API リファレンス

```
GET  /api/health
GET  /api/config
GET  /api/quote?ticker=AAPL&period=60
POST /api/predict
```

`/api/predict` のリクエスト例：

```json
{
  "ticker":      "AAPL",
  "period":      60,
  "pred_len":    5,
  "model_size":  "base",
  "samples":     50,
  "temperature": 0.7,
  "top_p":       0.9
}
```

---

## 中文简介

### 这是什么

原版 [Kronos](https://github.com/shiyu-coder/Kronos) 是一个 Python 模型库，需要写代码才能使用。

本项目在此基础上构建了**完整的本地应用**：
- 一键 GUI 启动器（无需手动输命令）
- 浏览器交互式 K 线图（拖拽/缩放/悬停显示指标）
- Flask REST API（支持外部调用）
- 中文 / English / 日本語 界面切换

### 快速开始

```bash
# 1. 创建环境
conda create -n kronos python=3.10 -y
conda activate kronos

# 2. 安装依赖（GPU 版）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install flask flask-cors yfinance pandas numpy matplotlib huggingface-hub safetensors einops

# 3. 下载 Kronos 模型仓库
git clone https://github.com/shiyu-coder/Kronos.git

# 4. 编辑 kronos_config.ini 填写 MODEL_DIR 路径

# 5. 启动
python kronos_launcher.py
```

---

## English Introduction

### What makes this different

The original [Kronos](https://github.com/shiyu-coder/Kronos) is a bare Python library — you need to write Python code to run a single prediction.

This project wraps it into a **full desktop + web application**:

- **One-click launcher** — GUI detects conda, installs missing packages, starts both servers, and opens the browser automatically
- **Interactive browser chart** — candlestick engine built from scratch with the Canvas API; drag to pan, scroll to zoom, hover for OHLC + indicator values; MA / Bollinger Bands / MACD / RSI / Volume all in one view
- **REST API server** — Flask backend with persistent model caching so repeated predictions don't reload weights; fetches live OHLCV data from Yahoo Finance
- **Wick clipping** — post-processes predicted candles to prevent unrealistic shadow lengths based on historical wick-to-body ratios
- **Multilingual UI** — Chinese / English / Japanese, switchable without restarting

### Quick Start

```bash
# 1. Create environment
conda create -n kronos python=3.10 -y
conda activate kronos

# 2. Install deps (GPU build)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install flask flask-cors yfinance pandas numpy matplotlib huggingface-hub safetensors einops

# 3. Clone Kronos model repo
git clone https://github.com/shiyu-coder/Kronos.git

# 4. Edit kronos_config.ini — set MODEL_DIR to the cloned path

# 5. Launch
python kronos_launcher.py
```

---

## ファイル構成

```
KRONOS-Market-Intelligence/
├── kronos_launcher.py      # GUI ランチャー（tkinter）
├── kronos_server.py        # Flask REST API サーバー
├── kronos_terminal.html    # ブラウザ チャート UI
├── kronos_predictor.py     # CLI 予測 + matplotlib チャート
├── kronos_config.ini       # 設定ファイル
├── environment.txt         # 動作確認済みパッケージ一覧
└── .kronos_lang            # 言語引き渡し（自動生成）
```

## 動作確認環境

| 項目 | 内容 |
|---|---|
| OS | Windows 11 |
| GPU | NVIDIA GeForce RTX 5070 Ti (16 GB VRAM) |
| CUDA | 12.8 / Driver 591.74 |
| PyTorch | 2.11.0+cu128 |
| Python | 3.10 |

---

## クレジット / Credits

本プロジェクトは以下のモデルを推論エンジンとして使用しています：

> **Kronos: A Foundation Model for the Language of Financial Markets**
> shiyu-coder et al. — [GitHub](https://github.com/shiyu-coder/Kronos)
>
> モデルの著作権・学術的成果は原著者に帰属します。
> Model weights and academic contributions belong to the original authors.

このリポジトリが提供するのは **GUI / API サーバー / ブラウザ UI レイヤーのみ**です。
モデルの重みファイルはこのリポジトリに含まれません。別途 clone してください。

---

## ライセンス / License

MIT License — see [LICENSE](LICENSE) for details.

> ⚠️ **免責事項 / Disclaimer / 免责声明**
> 本ツールは教育・研究目的のみです。投資判断の根拠として使用しないでください。
> For educational and research use only. Not financial advice.
> 仅供教育研究目的，不构成投资建议。
