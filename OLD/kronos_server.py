#!/usr/bin/env python3
"""
Kronos 本地 API 服务器
========================
在 conda kronos 环境里运行，为 HTML 前端提供真实模型预测接口。

启动方法:
    conda activate kronos
    cd D:\desk\KABUYOSO\Kronos
    python kronos_server.py

然后浏览器打开 kronos_terminal.html 即可使用真实模型。

依赖:
    pip install flask flask-cors yfinance
"""

import sys, os, warnings
warnings.filterwarnings("ignore")

# ─── 将 Kronos 仓库加入路径 ──────────────────────────────────────────────────
KRONOS_DIR = os.path.dirname(os.path.abspath(__file__))
if KRONOS_DIR not in sys.path:
    sys.path.insert(0, KRONOS_DIR)

# ─── 依赖检查 ────────────────────────────────────────────────────────────────
def check_deps():
    missing = []
    for pkg in ["flask", "flask_cors", "yfinance", "torch", "pandas", "numpy"]:
        try: __import__(pkg)
        except ImportError: missing.append(pkg)
    if missing:
        print(f"❌ 缺少依赖: {', '.join(missing)}")
        print(f"   请运行: pip install {' '.join(missing)}")
        sys.exit(1)

check_deps()

import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # 允许浏览器跨域请求

# ─── 全局模型缓存（只加载一次）──────────────────────────────────────────────
_model_cache = {}

def get_model(model_size: str, device: str):
    key = f"{model_size}_{device}"
    if key in _model_cache:
        return _model_cache[key]

    try:
        from model import Kronos, KronosTokenizer, KronosPredictor
    except ImportError:
        raise RuntimeError("未找到 Kronos model 模块，请确保在仓库目录下运行")

    cfg = {
        "mini":  ("NeoQuasar/Kronos-mini",  "NeoQuasar/Kronos-Tokenizer-2k",  2048),
        "small": ("NeoQuasar/Kronos-small", "NeoQuasar/Kronos-Tokenizer-base", 512),
        "base":  ("NeoQuasar/Kronos-base",  "NeoQuasar/Kronos-Tokenizer-base", 512),
    }
    model_name, tok_name, max_ctx = cfg[model_size]

    print(f"  加载 Kronos-{model_size} → device={device} ...")
    tokenizer = KronosTokenizer.from_pretrained(tok_name)
    model     = Kronos.from_pretrained(model_name).to(device)
    model.eval()
    predictor = KronosPredictor(model, tokenizer, max_context=max_ctx)

    _model_cache[key] = {
        "predictor": predictor,
        "max_ctx":   max_ctx,
        "params":    sum(p.numel() for p in model.parameters()) / 1e6,
    }
    print(f"  ✅ 模型加载完成  参数量: {_model_cache[key]['params']:.1f}M")
    return _model_cache[key]

# ─── 设备检测 ────────────────────────────────────────────────────────────────
def auto_device():
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends,"mps") and torch.backends.mps.is_available():
            return "mps"
    except: pass
    return "cpu"

DEVICE = auto_device()
print(f"\n🖥️  推理设备: {DEVICE}")

# ─── Yahoo Finance 数据 ──────────────────────────────────────────────────────
def fetch_data(ticker: str, period_days: int) -> pd.DataFrame:
    import yfinance as yf
    end   = datetime.today()
    start = end - timedelta(days=period_days + 40)
    df = yf.Ticker(ticker).history(
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"))
    if df.empty:
        raise ValueError(f"无法获取 [{ticker}] 数据")
    df = df.reset_index()
    df.columns = [c.lower() for c in df.columns]
    df = df.rename(columns={"date": "timestamps"})
    df["timestamps"] = pd.to_datetime(df["timestamps"]).dt.tz_localize(None)
    return df.tail(period_days).reset_index(drop=True)[
        ["timestamps","open","high","low","close","volume"]
    ]

# ─── 影线裁剪 ────────────────────────────────────────────────────────────────
def clip_wicks(df_hist: pd.DataFrame, pred_df: pd.DataFrame) -> pd.DataFrame:
    bodies = (df_hist["close"] - df_hist["open"]).abs()
    ranges = (df_hist["high"]  - df_hist["low"]).abs()
    ratio  = (ranges / bodies.replace(0, np.nan)).median()
    ratio  = float(np.clip(ratio, 1.5, 3.5))
    for idx in pred_df.index:
        o = pred_df.at[idx,"open"]
        c = pred_df.at[idx,"close"]
        body    = max(abs(c-o), abs(o)*0.002)
        maxWick = body*(ratio-1)/2
        pred_df.at[idx,"high"] = max(o,c) + maxWick
        pred_df.at[idx,"low"]  = min(o,c) - maxWick
    return pred_df

# ─── API 路由 ─────────────────────────────────────────────────────────────────

@app.route("/api/health", methods=["GET"])
def health():
    """健康检查，前端用来确认服务器是否在线"""
    return jsonify({
        "status":  "ok",
        "device":  DEVICE,
        "models":  list(_model_cache.keys()),
        "time":    datetime.now().isoformat(),
    })


@app.route("/api/predict", methods=["POST"])
def predict():
    """
    主预测接口
    请求体 (JSON):
      ticker:      股票代码
      period:      历史天数
      pred_len:    预测天数
      model_size:  mini / small / base
      samples:     采样数
      temperature: 采样温度
      top_p:       nucleus sampling
    """
    data = request.get_json()
    ticker      = data.get("ticker",      "AAPL").upper()
    period      = int(data.get("period",       60))
    pred_len    = int(data.get("pred_len",      5))
    model_size  = data.get("model_size",  "base")
    samples     = int(data.get("samples",      20))
    temperature = float(data.get("temperature", 1.0))
    top_p       = float(data.get("top_p",       0.9))

    print(f"\n📡 预测请求: {ticker}  period={period}  pred={pred_len}  "
          f"model={model_size}  samples={samples}  T={temperature}")

    try:
        # 1. 获取数据
        df = fetch_data(ticker, period)
        print(f"  ✅ 数据: {len(df)} 个交易日")

        # 2. 加载模型
        cached   = get_model(model_size, DEVICE)
        pred_obj = cached["predictor"]
        max_ctx  = cached["max_ctx"]

        if len(df) > max_ctx:
            df = df.tail(max_ctx).reset_index(drop=True)

        # 3. 推理
        x_df        = df[["open","high","low","close","volume"]].copy()
        x_timestamp = df["timestamps"]
        freq        = pd.infer_freq(df["timestamps"]) or "B"
        y_timestamp = pd.Series(pd.date_range(
            start=df["timestamps"].iloc[-1],
            periods=pred_len+1, freq=freq)[1:])

        print(f"  🔮 推理中 (samples={samples})...")
        pred_df = pred_obj.predict(
            df=x_df, x_timestamp=x_timestamp, y_timestamp=y_timestamp,
            pred_len=pred_len, T=temperature, top_p=top_p,
            sample_count=samples)
        pred_df["timestamps"] = y_timestamp.values

        # 4. 影线裁剪
        pred_df = clip_wicks(df, pred_df)
        print("  ✅ 推理完成")

        # 5. 序列化返回
        hist_rows = df.to_dict(orient="records")
        pred_rows = pred_df.to_dict(orient="records")

        # 时间戳转字符串
        for r in hist_rows:
            r["timestamps"] = str(r["timestamps"])[:10]
        for r in pred_rows:
            r["timestamps"] = str(r["timestamps"])[:10]

        return jsonify({
            "ok":     True,
            "ticker": ticker,
            "hist":   hist_rows,
            "pred":   pred_rows,
            "meta": {
                "model":       f"Kronos-{model_size}",
                "device":      DEVICE,
                "params_m":    cached["params"],
                "samples":     samples,
                "temperature": temperature,
            }
        })

    except Exception as e:
        print(f"  ❌ 错误: {e}")
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/quote", methods=["GET"])
def quote():
    """快速获取历史行情（不预测，用于前端实时刷新价格）"""
    ticker = request.args.get("ticker","AAPL").upper()
    period = int(request.args.get("period", 60))
    try:
        df    = fetch_data(ticker, period)
        rows  = df.to_dict(orient="records")
        for r in rows:
            r["timestamps"] = str(r["timestamps"])[:10]
        return jsonify({"ok":True, "ticker":ticker, "hist":rows})
    except Exception as e:
        return jsonify({"ok":False, "error":str(e)}), 500


# ─── 启动 ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "═"*55)
    print("  KRONOS 本地 API 服务器")
    print("  前端连接地址: http://localhost:5000")
    print("  健康检查:     http://localhost:5000/api/health")
    print("═"*55)
    print(f"\n  ✅ 设备: {DEVICE}")
    print("  ⚡ 首次预测时加载模型权重（约5-10秒）\n")

    # 预热：可选，提前加载 base 模型
    # print("  预热加载 Kronos-base...")
    # get_model("base", DEVICE)

    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
