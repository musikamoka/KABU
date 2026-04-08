#!/usr/bin/env python3
"""
Kronos K-Line Predictor
========================
使用 Kronos 基础模型 + Yahoo Finance 数据预测股票K线
配色：红涨绿跌（A股/中国市场风格）

安装依赖:
    pip install yfinance pandas matplotlib torch transformers huggingface_hub

使用方法:
    python kronos_predictor.py --ticker AAPL --demo        # 演示模式
    python kronos_predictor.py --ticker AAPL               # 真实 Kronos 模型
    python kronos_predictor.py --ticker TSLA --period 60 --pred_len 10
    python kronos_predictor.py --ticker 7203.T --period 90 --pred_len 15

真实模型使用前提:
    1. 在 Kronos 仓库目录下运行本脚本（推荐）
       git clone https://github.com/shiyu-coder/Kronos && cd Kronos
    2. 或通过 --kronos_path 指定仓库路径

模型来源: https://www.zdoc.app/zh/shiyu-coder/Kronos
"""

import argparse
import sys
import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from datetime import datetime, timedelta

# ─── 颜色方案（红涨绿跌，A股风格）────────────────────────────────────────────
BG       = "#0d1117"
PANEL_BG = "#161b22"
GRID     = "#21262d"
UP_H     = "#e84040"   # 历史上涨 → 红
DN_H     = "#26a84e"   # 历史下跌 → 绿
UP_P     = "#ff6b6b"   # 预测上涨 → 浅红
DN_P     = "#3fcf70"   # 预测下跌 → 浅绿
ACCENT   = "#58a6ff"
TEXT_PRI = "#e6edf3"
TEXT_SEC = "#8b949e"
GOLD     = "#d29922"

# ─── 参数解析 ────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Kronos K线预测器（红涨绿跌）")
    parser.add_argument("--ticker",      type=str,  default="AAPL",  help="股票代码 (e.g. AAPL, TSLA, 7203.T)")
    parser.add_argument("--period",      type=int,  default=60,      help="历史数据天数 (默认 60)")
    parser.add_argument("--pred_len",    type=int,  default=10,      help="预测天数 (默认 10)")
    parser.add_argument("--model",       type=str,  default="small", choices=["mini","small","base"],
                                                                      help="Kronos 模型大小")
    parser.add_argument("--samples",     type=int,  default=3,       help="采样路径数量 (默认 3)")
    parser.add_argument("--output",      type=str,  default=None,    help="图表保存路径 (默认显示)")
    parser.add_argument("--demo",        action="store_true",        help="演示模式（不需要 GPU，用统计模拟）")
    parser.add_argument("--kronos_path", type=str,  default=None,    help="Kronos 仓库路径（非仓库目录运行时指定）")
    parser.add_argument("--device",      type=str,  default="auto",  help="推理设备: auto / cpu / cuda / cuda:0")
    return parser.parse_args()

# ─── 数据获取 ────────────────────────────────────────────────────────────────

def fetch_data(ticker: str, period_days: int) -> pd.DataFrame:
    """从 Yahoo Finance 获取 OHLCV 数据"""
    try:
        import yfinance as yf
    except ImportError:
        print("❌ 请先安装 yfinance: pip install yfinance")
        sys.exit(1)

    print(f"📡 正在从 Yahoo Finance 获取 [{ticker}] 数据...")
    end   = datetime.today()
    start = end - timedelta(days=period_days + 30)

    ticker_obj = yf.Ticker(ticker)
    df = ticker_obj.history(start=start.strftime("%Y-%m-%d"),
                             end=end.strftime("%Y-%m-%d"))

    if df.empty:
        print(f"❌ 无法获取 [{ticker}] 数据，请检查股票代码")
        sys.exit(1)

    df = df.reset_index()
    df.columns = [c.lower() for c in df.columns]
    df = df.rename(columns={"date": "timestamps"})
    df["timestamps"] = pd.to_datetime(df["timestamps"]).dt.tz_localize(None)
    df = df.tail(period_days).reset_index(drop=True)

    print(f"✅ 获取成功！共 {len(df)} 个交易日  "
          f"[{df['timestamps'].iloc[0].date()} → {df['timestamps'].iloc[-1].date()}]")
    return df[["timestamps", "open", "high", "low", "close", "volume"]]

# ─── Kronos 真实预测 ─────────────────────────────────────────────────────────

def resolve_device(device_arg: str) -> str:
    """解析推理设备"""
    if device_arg != "auto":
        return device_arg
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def kronos_predict(df: pd.DataFrame, pred_len: int, model_size: str,
                   samples: int, kronos_path: str, device_arg: str) -> pd.DataFrame:
    """使用真实 Kronos 模型进行预测"""

    # ── 将 Kronos 仓库加入路径 ──
    search_paths = [kronos_path] if kronos_path else []
    search_paths.append(os.path.dirname(os.path.abspath(__file__)))  # 当前脚本目录
    search_paths.append(os.getcwd())                                  # 当前工作目录

    loaded = False
    for p in search_paths:
        if p and os.path.isdir(os.path.join(p, "model")):
            if p not in sys.path:
                sys.path.insert(0, p)
            loaded = True
            print(f"✅ 找到 Kronos 仓库: {p}")
            break

    try:
        from model import Kronos, KronosTokenizer, KronosPredictor
    except ImportError:
        print("\n⚠️  未找到 Kronos 模型文件！")
        print("   请确保：")
        print("   1. 在 Kronos 仓库目录下运行本脚本，或")
        print("   2. 使用 --kronos_path /your/path/to/Kronos 指定路径")
        print("   仓库克隆：git clone https://github.com/shiyu-coder/Kronos")
        print("\n   → 自动切换到演示模式...\n")
        return None

    model_map = {
        "mini":  ("NeoQuasar/Kronos-mini",  "NeoQuasar/Kronos-Tokenizer-2k",   2048),
        "small": ("NeoQuasar/Kronos-small", "NeoQuasar/Kronos-Tokenizer-base",  512),
        "base":  ("NeoQuasar/Kronos-base",  "NeoQuasar/Kronos-Tokenizer-base",  512),
    }
    model_name, tok_name, max_ctx = model_map[model_size]
    device = resolve_device(device_arg)
    print(f"🤖 加载 Kronos-{model_size}  |  device={device}  |  从 HuggingFace 下载（首次较慢）...")

    try:
        tokenizer = KronosTokenizer.from_pretrained(tok_name)
        model     = Kronos.from_pretrained(model_name)
        model     = model.to(device)
        model.eval()
        predictor = KronosPredictor(model, tokenizer, max_context=max_ctx)
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        print("   → 自动切换到演示模式...")
        return None

    x_df        = df[["open", "high", "low", "close", "volume"]].copy()
    x_timestamp = df["timestamps"]

    last_date   = df["timestamps"].iloc[-1]
    freq        = pd.infer_freq(df["timestamps"]) or "B"
    y_timestamp = pd.date_range(start=last_date, periods=pred_len + 1, freq=freq)[1:]
    y_timestamp = pd.Series(y_timestamp)

    print(f"🔮 生成预测 (pred_len={pred_len}, samples={samples}, T=1.0, top_p=0.9)...")
    try:
        pred_df = predictor.predict(
            df           = x_df,
            x_timestamp  = x_timestamp,
            y_timestamp  = y_timestamp,
            pred_len     = pred_len,
            T            = 1.0,
            top_p        = 0.9,
            sample_count = samples,
        )
    except Exception as e:
        print(f"❌ 推理失败: {e}")
        print("   → 自动切换到演示模式...")
        return None

    pred_df["timestamps"] = y_timestamp.values
    return pred_df

# ─── 演示模式预测（统计模拟）─────────────────────────────────────────────────

def demo_predict(df: pd.DataFrame, pred_len: int, samples: int) -> tuple:
    """用历史统计特征模拟 Kronos 风格预测"""
    print("🎭 演示模式：使用统计方法模拟 Kronos 预测风格")

    closes       = df["close"].values
    returns      = np.diff(closes) / closes[:-1]
    mu           = returns.mean()
    sigma        = returns.std()
    recent_sigma = returns[-20:].std() if len(returns) >= 20 else sigma

    last_date    = df["timestamps"].iloc[-1]
    future_dates = []
    d = last_date
    while len(future_dates) < pred_len:
        d += timedelta(days=1)
        if d.weekday() < 5:
            future_dates.append(d)

    all_paths = []
    for _ in range(samples):
        path = []
        prev_close = closes[-1]
        for i in range(pred_len):
            drift   = mu * 0.3
            shock   = np.random.normal(drift, recent_sigma * (1 + i * 0.02))
            open_p  = prev_close * (1 + np.random.normal(0, sigma * 0.3))
            close_p = prev_close * (1 + shock)
            high_p  = max(open_p, close_p) * (1 + abs(np.random.normal(0, sigma * 0.5)))
            low_p   = min(open_p, close_p) * (1 - abs(np.random.normal(0, sigma * 0.5)))
            path.append({
                "timestamps": future_dates[i],
                "open":  round(open_p,  4),
                "high":  round(high_p,  4),
                "low":   round(low_p,   4),
                "close": round(close_p, 4),
            })
            prev_close = close_p
        all_paths.append(pd.DataFrame(path))

    avg_pred = pd.DataFrame({
        "timestamps": future_dates,
        "open":  np.mean([p["open"].values  for p in all_paths], axis=0),
        "high":  np.mean([p["high"].values  for p in all_paths], axis=0),
        "low":   np.mean([p["low"].values   for p in all_paths], axis=0),
        "close": np.mean([p["close"].values for p in all_paths], axis=0),
    })
    return avg_pred, all_paths

# ─── K线图绘制（红涨绿跌）────────────────────────────────────────────────────

def plot_klines(df: pd.DataFrame, pred_df: pd.DataFrame,
                ticker: str, sample_paths=None, output_path=None,
                is_demo=False):
    """绘制专业风格的历史 + 预测 K 线图（红涨绿跌 A股配色）"""

    plt.rcParams.update({
        "font.family":      "monospace",
        "axes.facecolor":   PANEL_BG,
        "figure.facecolor": BG,
        "text.color":       TEXT_PRI,
        "axes.labelcolor":  TEXT_SEC,
        "xtick.color":      TEXT_SEC,
        "ytick.color":      TEXT_SEC,
        "axes.edgecolor":   GRID,
        "grid.color":       GRID,
        "grid.linewidth":   0.5,
    })

    fig = plt.figure(figsize=(18, 10))
    gs  = gridspec.GridSpec(2, 1, height_ratios=[4, 1], hspace=0.05)
    ax_k = fig.add_subplot(gs[0])
    ax_v = fig.add_subplot(gs[1], sharex=ax_k)

    n_hist    = len(df)
    n_pred    = len(pred_df)
    all_dates = list(df["timestamps"]) + list(pred_df["timestamps"])
    x_hist    = list(range(n_hist))
    x_pred    = list(range(n_hist, n_hist + n_pred))

    # ── 分隔线 ──
    ax_k.axvline(x=n_hist - 0.5, color=GOLD, linewidth=1.5, linestyle="--", alpha=0.8, zorder=3)
    ax_v.axvline(x=n_hist - 0.5, color=GOLD, linewidth=1.5, linestyle="--", alpha=0.8, zorder=3)

    # ── 采样路径阴影 ──
    if sample_paths:
        for path in sample_paths:
            closes = path["close"].values
            ax_k.fill_between(x_pred, closes * 0.995, closes * 1.005,
                               alpha=0.06, color=ACCENT, zorder=1)
            ax_k.plot(x_pred, closes, color=ACCENT, alpha=0.13, linewidth=0.8, zorder=1)

    # ── 绘制 K 线蜡烛 ──
    def draw_candles(ax_target, x_vals, data, up_c, dn_c, alpha=1.0, width=0.6):
        for xi, (_, row) in zip(x_vals, data.iterrows()):
            o, h, l, c = row["open"], row["high"], row["low"], row["close"]
            color  = up_c if c >= o else dn_c
            body_h = max(abs(c - o), 0.001 * o)      # 至少显示一点实体
            ax_target.plot([xi, xi], [l, h], color=color, linewidth=0.9, alpha=alpha, zorder=2)
            rect = plt.Rectangle(
                (xi - width/2, min(o, c)), width, body_h,
                facecolor=color, edgecolor=color,
                linewidth=0.5, alpha=alpha, zorder=2
            )
            ax_target.add_patch(rect)

    draw_candles(ax_k, x_hist, df,      UP_H, DN_H)
    draw_candles(ax_k, x_pred, pred_df, UP_P, DN_P, alpha=0.85, width=0.55)

    # ── 预测区域背景 ──
    ax_k.axvspan(n_hist - 0.5, n_hist + n_pred, color=ACCENT, alpha=0.03, zorder=0)
    ax_v.axvspan(n_hist - 0.5, n_hist + n_pred, color=ACCENT, alpha=0.03, zorder=0)

    # ── 成交量柱 ──
    if "volume" in df.columns:
        vols     = df["volume"].values
        v_colors = [UP_H if df["close"].iloc[i] >= df["open"].iloc[i] else DN_H
                    for i in range(n_hist)]
        ax_v.bar(x_hist, vols, color=v_colors, alpha=0.7, width=0.7, zorder=2)
        ax_v.set_ylabel("Vol", fontsize=8, color=TEXT_SEC)
        ax_v.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M" if x >= 1e6 else f"{x/1e3:.0f}K")
        )

    # ── X 轴标签 ──
    tick_step = max(1, (n_hist + n_pred) // 12)
    tick_pos  = list(range(0, n_hist + n_pred, tick_step))
    tick_lbs  = [all_dates[i].strftime("%m/%d") for i in tick_pos if i < len(all_dates)]
    ax_v.set_xticks(tick_pos[:len(tick_lbs)])
    ax_v.set_xticklabels(tick_lbs, fontsize=8, rotation=30, ha="right")
    ax_k.set_xticklabels([])

    # ── 网格 ──
    ax_k.grid(True, axis="y", alpha=0.4)
    ax_v.grid(True, axis="y", alpha=0.3)

    # ── Y 轴价格范围 ──
    all_prices = pd.concat([df[["open","high","low","close"]],
                             pred_df[["open","high","low","close"]]])
    p_min = all_prices.min().min() * 0.995
    p_max = all_prices.max().max() * 1.005
    ax_k.set_ylim(p_min, p_max)
    ax_k.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.2f}"))

    # ── 统计标注（左上角） ──
    last_close = df["close"].iloc[-1]
    pred_close = pred_df["close"].iloc[-1]
    chg        = (pred_close - last_close) / last_close * 100
    chg_color  = UP_P if chg >= 0 else DN_P

    info_text = (
        f"  当前: {last_close:.2f}   "
        f"预测末日: {pred_close:.2f}   "
        f"预期变化: {chg:+.2f}%  "
    )
    ax_k.text(0.01, 0.97, info_text, transform=ax_k.transAxes,
              fontsize=9, color=chg_color, va="top",
              bbox=dict(boxstyle="round,pad=0.3", facecolor=PANEL_BG,
                        edgecolor=chg_color, alpha=0.85))

    # ── 颜色说明（右上角附近） ──
    color_note = "配色：红涨绿跌（A股风格）"
    ax_k.text(0.99, 0.97, color_note, transform=ax_k.transAxes,
              fontsize=8, color=TEXT_SEC, va="top", ha="right")

    # ── 主标题 ──
    now_str  = datetime.now().strftime("%Y-%m-%d %H:%M")
    mode_str = "演示模式" if is_demo else "Kronos 真实预测"
    fig.suptitle(
        f"  KRONOS  ·  {ticker.upper()}  K线预测  [{mode_str}]",
        fontsize=16, fontweight="bold", color=TEXT_PRI,
        x=0.01, ha="left", y=0.98
    )
    ax_k.text(1.0, 1.02,
              f"Model: Kronos-{('demo' if is_demo else 'small')}  |  {now_str}  |  "
              f"zdoc.app/zh/shiyu-coder/Kronos",
              transform=ax_k.transAxes, fontsize=7, color=TEXT_SEC, ha="right")

    # ── 图例 ──
    legend_elems = [
        mpatches.Patch(color=UP_H, label="历史上涨（红）"),
        mpatches.Patch(color=DN_H, label="历史下跌（绿）"),
        mpatches.Patch(color=UP_P, label="预测上涨", alpha=0.85),
        mpatches.Patch(color=DN_P, label="预测下跌", alpha=0.85),
    ]
    if sample_paths:
        legend_elems.append(
            mpatches.Patch(color=ACCENT, alpha=0.3, label=f"采样路径 ×{len(sample_paths)}")
        )
    ax_k.legend(handles=legend_elems, loc="upper left",
                facecolor=PANEL_BG, edgecolor=GRID,
                labelcolor=TEXT_PRI, fontsize=8, framealpha=0.9,
                bbox_to_anchor=(0.01, 0.88))

    # ── 预测区间标签 ──
    ax_k.text(n_hist + n_pred / 2, p_max * 0.9995,
              "◀  Kronos 预测区间  ▶",
              ha="center", va="top", fontsize=8, color=GOLD,
              bbox=dict(boxstyle="round", facecolor=BG, edgecolor=GOLD, alpha=0.7))

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight",
                    facecolor=BG, edgecolor="none")
        print(f"💾 图表已保存: {output_path}")
    else:
        plt.show()

# ─── 主程序 ──────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    print("\n" + "═" * 60)
    print("  KRONOS K线预测器  |  红涨绿跌 A股配色")
    print(f"  来源: https://www.zdoc.app/zh/shiyu-coder/Kronos")
    print("═" * 60 + "\n")

    # 1. 获取数据
    df = fetch_data(args.ticker, args.period)

    # 2. 预测
    sample_paths = None
    is_demo      = args.demo

    if not is_demo:
        pred_df = kronos_predict(
            df, args.pred_len, args.model,
            args.samples, args.kronos_path, args.device
        )
        if pred_df is None:
            is_demo = True

    if is_demo:
        pred_df, sample_paths = demo_predict(df, args.pred_len, args.samples)

    print(f"\n📊 预测结果 ({args.ticker}):")
    print(pred_df[["timestamps", "open", "high", "low", "close"]].to_string(index=False))

    # 3. 绘图
    print("\n🎨 生成K线图（红涨绿跌）...")
    plot_klines(df, pred_df, args.ticker, sample_paths, args.output, is_demo)
    print("\n✅ 完成！")


if __name__ == "__main__":
    main()
