#!/usr/bin/env python3
"""
Kronos 本地 API 服务器  v2
===========================
新增：
  · --model-dir  指定 Kronos 仓库/模型目录（与运行文件目录完全分离）
  · --port       自定义端口（默认 5000）
  · --host       绑定地址（默认 0.0.0.0，局域网可访问）
  · CORS 全开（无安全限制）
  · /api/config  返回当前运行配置

目录结构示例（运行文件 与 模型 分开）：
  D:\Projects\kronos-app\          ← 运行文件目录（脚本所在位置）
      kronos_server.py
      kronos_terminal.html
      start.bat / start.sh

  D:\Models\Kronos\                ← 模型目录（--model-dir 指定）
      model\
          __init__.py
          kronos.py
          ...

启动方法：
  python kronos_server.py --model-dir D:\Models\Kronos
  python kronos_server.py --model-dir /home/user/Kronos --port 5001
"""

import sys, os, warnings, argparse
warnings.filterwarnings("ignore")

# ─── 命令行参数 ───────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Kronos API Server v2")
    p.add_argument(
        "--model-dir",
        default=None,
        help="Kronos 模型仓库目录（包含 model/ 子目录）。"
             "默认与本脚本同级目录。"
    )
    p.add_argument("--port", type=int, default=5000, help="监听端口（默认 5000）")
    p.add_argument("--host", default="0.0.0.0",    help="绑定地址（默认 0.0.0.0）")
    return p.parse_args()

ARGS = parse_args()

# ─── 确定模型目录并加入 sys.path ──────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))   # 运行文件目录
MODEL_DIR  = os.path.abspath(ARGS.model_dir) if ARGS.model_dir else SCRIPT_DIR

if not os.path.isdir(MODEL_DIR):
    print(f"❌ 模型目录不存在: {MODEL_DIR}")
    sys.exit(1)

for d in [MODEL_DIR, SCRIPT_DIR]:
    if d not in sys.path:
        sys.path.insert(0, d)

print(f"\n📂 运行文件目录 : {SCRIPT_DIR}")
print(f"📂 模型目录     : {MODEL_DIR}")

# ─── 依赖检查 ────────────────────────────────────────────────────────────────
def check_deps():
    missing = []
    for pkg in ["flask", "flask_cors", "yfinance", "torch", "pandas", "numpy"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"❌ 缺少依赖: {', '.join(missing)}")
        print(f"   请运行: pip install {' '.join(missing)}")
        sys.exit(1)

check_deps()

import json
import numpy  as np
import pandas as pd
from datetime import datetime, timedelta
from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)

# ─── CORS 全开（无安全限制）────────────────────────────────────────────────────
CORS(app, resources={r"/*": {"origins": "*"}})

# ─── 全局模型缓存 ─────────────────────────────────────────────────────────────
_model_cache = {}
_cache_lock  = __import__("threading").Lock()

def get_model(model_size: str, device: str):
    key = f"{model_size}_{device}"
    with _cache_lock:
        if key in _model_cache:
            return _model_cache[key]

    # 确保模型目录在路径中
    if MODEL_DIR not in sys.path:
        sys.path.insert(0, MODEL_DIR)

    try:
        from model import Kronos, KronosTokenizer, KronosPredictor
    except ImportError as e:
        raise RuntimeError(
            f"未找到 Kronos model 模块（路径: {MODEL_DIR}）\n"
            f"原始错误: {e}\n"
            f"请确认 --model-dir 指向包含 model/ 子目录的 Kronos 仓库"
        )

    cfg = {
        "mini":  ("NeoQuasar/Kronos-mini",  "NeoQuasar/Kronos-Tokenizer-2k",  2048),
        "small": ("NeoQuasar/Kronos-small", "NeoQuasar/Kronos-Tokenizer-base",  512),
        "base":  ("NeoQuasar/Kronos-base",  "NeoQuasar/Kronos-Tokenizer-base",  512),
    }
    model_name, tok_name, max_ctx = cfg[model_size]

    print(f"  加载 Kronos-{model_size} → device={device} ...")
    tokenizer = KronosTokenizer.from_pretrained(tok_name)
    model     = Kronos.from_pretrained(model_name).to(device)
    model.eval()
    predictor = KronosPredictor(model, tokenizer, max_context=max_ctx)
    params    = sum(p.numel() for p in model.parameters()) / 1e6

    entry = {"predictor": predictor, "max_ctx": max_ctx, "params": params}
    with _cache_lock:
        _model_cache[key] = entry
    print(f"  ✅ 模型加载完成  参数量: {params:.1f}M")
    return entry

# ─── 设备自动检测 ─────────────────────────────────────────────────────────────
def auto_device():
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"

DEVICE = auto_device()
print(f"🖥️  推理设备: {DEVICE}\n")

# ─── Yahoo Finance 数据获取 ────────────────────────────────────────────────────
def fetch_data(ticker: str, period_days: int) -> pd.DataFrame:
    import yfinance as yf
    end   = datetime.today()
    start = end - timedelta(days=period_days + 40)
    df = yf.Ticker(ticker).history(
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
    )
    if df.empty:
        raise ValueError(f"无法获取 [{ticker}] 数据，请检查代码是否正确")
    df = df.reset_index()
    df.columns = [c.lower() for c in df.columns]
    df = df.rename(columns={"date": "timestamps"})
    df["timestamps"] = pd.to_datetime(df["timestamps"]).dt.tz_localize(None)
    return (
        df.tail(period_days)
          .reset_index(drop=True)
          [["timestamps", "open", "high", "low", "close", "volume"]]
    )

# ─── 影线裁剪 ────────────────────────────────────────────────────────────────
def clip_wicks(df_hist: pd.DataFrame, pred_df: pd.DataFrame) -> pd.DataFrame:
    bodies = (df_hist["close"] - df_hist["open"]).abs()
    ranges = (df_hist["high"]  - df_hist["low"]).abs()
    ratio  = float(np.clip(
        (ranges / bodies.replace(0, np.nan)).median(), 1.5, 3.5
    ))
    for idx in pred_df.index:
        o = pred_df.at[idx, "open"]
        c = pred_df.at[idx, "close"]
        body    = max(abs(c - o), abs(o) * 0.002)
        maxWick = body * (ratio - 1) / 2
        pred_df.at[idx, "high"] = max(o, c) + maxWick
        pred_df.at[idx, "low"]  = min(o, c) - maxWick
    return pred_df

# ─── 辅助：时间戳序列化 ────────────────────────────────────────────────────────
def rows_to_json(df: pd.DataFrame) -> list:
    rows = df.to_dict(orient="records")
    for r in rows:
        r["timestamps"] = str(r["timestamps"])[:10]
    return rows

# ═══════════════════════════════════════════════════════════════════════════════
#  API 路由
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/api/health", methods=["GET"])
def health():
    """健康检查"""
    return jsonify({
        "status": "ok",
        "device": DEVICE,
        "models": list(_model_cache.keys()),
        "time":   datetime.now().isoformat(),
    })


@app.route("/api/config", methods=["GET"])
def config():
    """返回当前运行配置"""
    return jsonify({
        "script_dir": SCRIPT_DIR,
        "model_dir":  MODEL_DIR,
        "device":     DEVICE,
        "port":       ARGS.port,
        "host":       ARGS.host,
    })


@app.route("/api/predict", methods=["POST"])
def predict():
    """
    主预测接口
    Body (JSON):
      ticker       股票代码
      period       历史天数
      pred_len     预测天数
      model_size   mini / small / base
      samples      采样数
      temperature  采样温度
      top_p        nucleus sampling
    """
    data        = request.get_json() or {}
    ticker      = data.get("ticker",      "AAPL").upper()
    period      = int(data.get("period",         60))
    pred_len    = int(data.get("pred_len",         5))
    model_size  = data.get("model_size",    "base")
    samples     = int(data.get("samples",         20))
    temperature = float(data.get("temperature",  1.0))
    top_p       = float(data.get("top_p",         0.9))

    print(f"\n📡 预测请求: {ticker}  period={period}  pred={pred_len}  "
          f"model={model_size}  samples={samples}  T={temperature}")

    try:
        # 1. 行情数据
        df = fetch_data(ticker, period)
        print(f"  ✅ 数据: {len(df)} 个交易日")

        # 2. 加载模型
        cached   = get_model(model_size, DEVICE)
        pred_obj = cached["predictor"]
        max_ctx  = cached["max_ctx"]

        if len(df) > max_ctx:
            df = df.tail(max_ctx).reset_index(drop=True)

        # 3. 推理
        x_df        = df[["open", "high", "low", "close", "volume"]].copy()
        x_timestamp = df["timestamps"]
        freq        = pd.infer_freq(df["timestamps"]) or "B"
        y_timestamp = pd.Series(
            pd.date_range(
                start=df["timestamps"].iloc[-1],
                periods=pred_len + 1, freq=freq,
            )[1:]
        )

        print(f"  🔮 推理中 (samples={samples})...")
        pred_df = pred_obj.predict(
            df=x_df, x_timestamp=x_timestamp, y_timestamp=y_timestamp,
            pred_len=pred_len, T=temperature, top_p=top_p,
            sample_count=samples,
        )
        pred_df["timestamps"] = y_timestamp.values

        # 4. 影线裁剪
        pred_df = clip_wicks(df, pred_df)
        print("  ✅ 推理完成")

        return jsonify({
            "ok":     True,
            "ticker": ticker,
            "hist":   rows_to_json(df),
            "pred":   rows_to_json(pred_df),
            "meta": {
                "model":       f"Kronos-{model_size}",
                "device":      DEVICE,
                "params_m":    cached["params"],
                "samples":     samples,
                "temperature": temperature,
                "model_dir":   MODEL_DIR,
            },
        })

    except Exception as e:
        import traceback
        print(f"  ❌ 错误: {e}")
        traceback.print_exc()
        # 区分错误类型返回不同状态码
        if "无法获取" in str(e) or "No data" in str(e):
            return jsonify({"ok": False, "error": str(e)}), 404
        if "未找到 Kronos model" in str(e):
            return jsonify({"ok": False, "error": str(e)}), 503
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/quote", methods=["GET"])
def quote():
    """仅获取历史行情，不做预测"""
    ticker = request.args.get("ticker", "AAPL").upper()
    period = int(request.args.get("period", 60))
    try:
        df = fetch_data(ticker, period)
        return jsonify({"ok": True, "ticker": ticker, "hist": rows_to_json(df)})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


# ═══════════════════════════════════════════════════════════════════════════════
#  启动
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("═" * 60)
    print("  KRONOS API 服务器  v2")
    print(f"  监听地址  : http://{ARGS.host}:{ARGS.port}")
    print(f"  健康检查  : http://localhost:{ARGS.port}/api/health")
    print(f"  当前配置  : http://localhost:{ARGS.port}/api/config")
    print("═" * 60)
    print(f"  运行文件目录 : {SCRIPT_DIR}")
    print(f"  模型目录     : {MODEL_DIR}")
    print(f"  推理设备     : {DEVICE}")
    print("  ⚡ 首次预测时加载模型权重（约 5-10 秒）\n")

    app.run(
        host=ARGS.host,
        port=ARGS.port,
        debug=False,
        threaded=True,
    )
