#!/usr/bin/env python3
"""
Kronos K-Line Predictor
========================
使用 Kronos 基础模型 + Yahoo Finance 数据预测股票K线
配色：红涨绿跌（A股/中国市场风格）

安装依赖:
    pip install yfinance pandas matplotlib torch transformers huggingface_hub

使用方法:
    python kronos_predictor.py --ticker AAPL --demo          # 演示模式（无需GPU）
    python kronos_predictor.py --ticker AAPL                 # 真实模型（默认 base）
    python kronos_predictor.py --ticker TSLA --pred_len 15
    python kronos_predictor.py --ticker 7203.T --model base --device cuda:0

真实模型前提:
    在 Kronos 仓库目录下运行，或用 --kronos_path 指定路径
    git clone https://github.com/shiyu-coder/Kronos && cd Kronos

模型来源: https://www.zdoc.app/zh/shiyu-coder/Kronos
"""

import argparse
import sys
import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from datetime import datetime, timedelta

# ─── 中文字体自动检测 ────────────────────────────────────────────────────────

def setup_chinese_font():
    """
    自动检测系统可用的中文字体，解决口口乱码问题。
    优先顺序：系统内置中文字体 → 英文回退（标签改为英文）
    """
    import matplotlib.font_manager as fm

    # 候选中文字体（覆盖 Windows / macOS / Linux）
    candidates = [
        # Windows
        "Microsoft YaHei", "SimHei", "SimSun", "KaiTi", "FangSong",
        # macOS
        "PingFang SC", "Heiti SC", "STHeiti", "Hiragino Sans GB",
        "Arial Unicode MS",
        # Linux / conda 常见
        "WenQuanYi Micro Hei", "WenQuanYi Zen Hei", "Noto Sans CJK SC",
        "Source Han Sans CN", "Droid Sans Fallback",
    ]

    available = {f.name for f in fm.fontManager.ttflist}
    for font in candidates:
        if font in available:
            matplotlib.rcParams["font.sans-serif"] = [font, "DejaVu Sans"]
            matplotlib.rcParams["axes.unicode_minus"] = False
            print(f"✅ 中文字体: {font}")
            return True

    # 找不到中文字体 → 用英文标签，避免乱码
    matplotlib.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False
    print("⚠️  未找到中文字体，图表标签将使用英文（安装方法见下方提示）")
    print("   Windows: 已内置，重启 Python 环境后生效")
    print("   macOS:   已内置，重启 Python 环境后生效")
    print("   Linux:   sudo apt-get install fonts-wqy-microhei && "
          "python -c \"import matplotlib.font_manager; matplotlib.font_manager._load_fontmanager(try_read_cache=False)\"")
    return False

HAS_CN = setup_chinese_font()

def cn(zh_text: str, en_text: str) -> str:
    """根据是否有中文字体，返回中文或英文标签"""
    return zh_text if HAS_CN else en_text

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
    parser.add_argument("--ticker",      type=str,  default="AAPL",
                        help="股票代码 (e.g. AAPL, TSLA, 7203.T, 600977.SS)")
    parser.add_argument("--period",      type=int,  default=120,
                        help="历史数据天数，默认 120（base 模型上下文 512，建议 120~480）")
    parser.add_argument("--pred_len",    type=int,  default=20,
                        help="预测天数，默认 20")
    parser.add_argument("--model",       type=str,  default="base",
                        choices=["mini", "small", "base"],
                        help="Kronos 模型大小（默认 base=102.3M，最强）")
    parser.add_argument("--samples",     type=int,  default=10,
                        help="采样路径数，默认 10（越多越稳定，越慢）")
    parser.add_argument("--top_p",       type=float, default=0.9,
                        help="Nucleus sampling top_p，默认 0.9")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="采样温度，默认 1.0（越低越保守）")
    parser.add_argument("--output",      type=str,  default=None,
                        help="图表保存路径（默认弹窗显示）")
    parser.add_argument("--demo",        action="store_true",
                        help="演示模式（不需要 GPU，用统计模拟）")
    parser.add_argument("--kronos_path", type=str,  default=None,
                        help="Kronos 仓库路径（非仓库目录运行时指定）")
    parser.add_argument("--device",      type=str,  default="auto",
                        help="推理设备: auto / cpu / cuda / cuda:0 / mps")
    return parser.parse_args()

# ─── 数据获取 ────────────────────────────────────────────────────────────────

def fetch_data(ticker: str, period_days: int) -> pd.DataFrame:
    """从 Yahoo Finance 获取 OHLCV 日线数据"""
    try:
        import yfinance as yf
    except ImportError:
        print("❌ 请先安装 yfinance: pip install yfinance")
        sys.exit(1)

    print(f"📡 正在从 Yahoo Finance 获取 [{ticker}] 数据...")
    end   = datetime.today()
    start = end - timedelta(days=period_days + 40)   # 多拉一些备用

    df = yf.Ticker(ticker).history(
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d")
    )

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
    if device_arg != "auto":
        return device_arg
    try:
        import torch
        if torch.cuda.is_available():
            dev = "cuda"
            props = torch.cuda.get_device_properties(0)
            print(f"🖥️  GPU: {props.name}  |  显存: {props.total_memory/1024**3:.1f} GB")
            return dev
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            print("🖥️  设备: Apple MPS")
            return "mps"
    except ImportError:
        pass
    print("🖥️  设备: CPU（推理较慢）")
    return "cpu"


def kronos_predict(df: pd.DataFrame, pred_len: int, model_size: str,
                   samples: int, top_p: float, temperature: float,
                   kronos_path: str, device_arg: str) -> pd.DataFrame:
    """使用真实 Kronos 模型进行预测"""

    # 搜索 Kronos 仓库
    search_paths = [p for p in [kronos_path,
                                 os.path.dirname(os.path.abspath(__file__)),
                                 os.getcwd()] if p]
    for p in search_paths:
        if os.path.isdir(os.path.join(p, "model")):
            if p not in sys.path:
                sys.path.insert(0, p)
            print(f"✅ 找到 Kronos 仓库: {p}")
            break

    try:
        from model import Kronos, KronosTokenizer, KronosPredictor
    except ImportError:
        print("\n⚠️  未找到 Kronos model 模块！")
        print("   请在 Kronos 仓库目录下运行，或用 --kronos_path 指定路径")
        print("   git clone https://github.com/shiyu-coder/Kronos && cd Kronos")
        print("   → 自动切换到演示模式...\n")
        return None

    # 模型配置
    model_cfg = {
        "mini":  ("NeoQuasar/Kronos-mini",  "NeoQuasar/Kronos-Tokenizer-2k",   2048),
        "small": ("NeoQuasar/Kronos-small", "NeoQuasar/Kronos-Tokenizer-base",  512),
        "base":  ("NeoQuasar/Kronos-base",  "NeoQuasar/Kronos-Tokenizer-base",  512),
    }
    model_name, tok_name, max_ctx = model_cfg[model_size]
    device = resolve_device(device_arg)

    # 如果 period 超过 max_ctx，自动截断并提示
    actual_lookback = min(len(df), max_ctx)
    if len(df) > max_ctx:
        print(f"⚠️  历史数据 {len(df)} 天超过模型上下文 {max_ctx}，自动截断至最近 {max_ctx} 天")
        df = df.tail(max_ctx).reset_index(drop=True)

    print(f"\n🤖 加载 Kronos-{model_size} ({model_name})")
    print(f"   上下文: {actual_lookback} 天  |  预测: {pred_len} 天  |  "
          f"采样: {samples} 路径  |  device: {device}")
    print(f"   temperature={temperature}  top_p={top_p}")
    print("   （首次运行会从 HuggingFace 下载权重，请耐心等待）\n")

    try:
        import torch
        tokenizer = KronosTokenizer.from_pretrained(tok_name)
        model     = Kronos.from_pretrained(model_name)
        model     = model.to(device)
        model.eval()
        predictor = KronosPredictor(model, tokenizer, max_context=max_ctx)

        # 显示参数量
        total_params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"✅ 模型加载完成  |  参数量: {total_params:.1f}M\n")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        print("   → 自动切换到演示模式...")
        return None

    # 准备输入
    x_df        = df[["open", "high", "low", "close", "volume"]].copy()
    x_timestamp = df["timestamps"]

    last_date   = df["timestamps"].iloc[-1]
    freq        = pd.infer_freq(df["timestamps"]) or "B"
    y_timestamp = pd.Series(pd.date_range(
        start=last_date, periods=pred_len + 1, freq=freq
    )[1:])

    print(f"🔮 推理中...")
    try:
        pred_df = predictor.predict(
            df           = x_df,
            x_timestamp  = x_timestamp,
            y_timestamp  = y_timestamp,
            pred_len     = pred_len,
            T            = temperature,
            top_p        = top_p,
            sample_count = samples,
        )
        pred_df["timestamps"] = y_timestamp.values
        print("✅ 推理完成！\n")
        return pred_df
    except Exception as e:
        print(f"❌ 推理失败: {e}")
        print("   → 自动切换到演示模式...")
        return None

# ─── 演示模式预测（统计模拟）─────────────────────────────────────────────────

def demo_predict(df: pd.DataFrame, pred_len: int, samples: int,
                 temperature: float = 1.0) -> tuple:
    print("🎭 演示模式：统计模拟 Kronos 预测风格（真实模型请去掉 --demo）")

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
        path, prev_close = [], closes[-1]
        for i in range(pred_len):
            shock   = np.random.normal(mu * 0.3, recent_sigma * temperature * (1 + i * 0.02))
            open_p  = prev_close * (1 + np.random.normal(0, sigma * 0.3 * temperature))
            close_p = prev_close * (1 + shock)
            high_p  = max(open_p, close_p) * (1 + abs(np.random.normal(0, sigma * 0.5)))
            low_p   = min(open_p, close_p) * (1 - abs(np.random.normal(0, sigma * 0.5)))
            path.append({"timestamps": future_dates[i],
                         "open": open_p, "high": high_p,
                         "low":  low_p,  "close": close_p})
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

# ─── K线图绘制 ───────────────────────────────────────────────────────────────

def plot_klines(df: pd.DataFrame, pred_df: pd.DataFrame,
                ticker: str, model_size: str, samples: int,
                sample_paths=None, output_path=None, is_demo=False):

    plt.rcParams.update({
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

    fig = plt.figure(figsize=(20, 11))
    gs  = gridspec.GridSpec(2, 1, height_ratios=[4, 1], hspace=0.05)
    ax_k = fig.add_subplot(gs[0])
    ax_v = fig.add_subplot(gs[1], sharex=ax_k)

    n_hist    = len(df)
    n_pred    = len(pred_df)
    all_dates = list(df["timestamps"]) + list(pred_df["timestamps"])
    x_hist    = list(range(n_hist))
    x_pred    = list(range(n_hist, n_hist + n_pred))

    # 分隔线
    for ax in [ax_k, ax_v]:
        ax.axvline(x=n_hist - 0.5, color=GOLD, linewidth=1.5,
                   linestyle="--", alpha=0.8, zorder=3)

    # 采样路径
    if sample_paths:
        for path in sample_paths:
            closes = path["close"].values
            ax_k.plot(x_pred, closes, color=ACCENT,
                      alpha=0.12, linewidth=0.7, zorder=1)
        # 置信区间带
        all_closes = np.array([p["close"].values for p in sample_paths])
        p10 = np.percentile(all_closes, 10, axis=0)
        p90 = np.percentile(all_closes, 90, axis=0)
        ax_k.fill_between(x_pred, p10, p90,
                          color=ACCENT, alpha=0.08, zorder=1,
                          label=cn("80% 置信区间", "80% CI"))

    # 绘制蜡烛
    def draw_candles(ax_target, x_vals, data, up_c, dn_c, alpha=1.0, width=0.6):
        for xi, (_, row) in zip(x_vals, data.iterrows()):
            o, h, l, c = row["open"], row["high"], row["low"], row["close"]
            color  = up_c if c >= o else dn_c
            body_h = max(abs(c - o), abs(o) * 0.001)
            ax_target.plot([xi, xi], [l, h], color=color,
                           linewidth=0.9, alpha=alpha, zorder=2)
            ax_target.add_patch(plt.Rectangle(
                (xi - width / 2, min(o, c)), width, body_h,
                facecolor=color, edgecolor=color,
                linewidth=0.4, alpha=alpha, zorder=2
            ))

    draw_candles(ax_k, x_hist, df,      UP_H, DN_H)
    draw_candles(ax_k, x_pred, pred_df, UP_P, DN_P, alpha=0.88, width=0.55)

    # 预测区背景
    ax_k.axvspan(n_hist - 0.5, n_hist + n_pred, color=ACCENT, alpha=0.025, zorder=0)
    ax_v.axvspan(n_hist - 0.5, n_hist + n_pred, color=ACCENT, alpha=0.025, zorder=0)

    # 成交量
    if "volume" in df.columns:
        v_colors = [UP_H if df["close"].iloc[i] >= df["open"].iloc[i] else DN_H
                    for i in range(n_hist)]
        ax_v.bar(x_hist, df["volume"].values,
                 color=v_colors, alpha=0.7, width=0.7, zorder=2)
        ax_v.set_ylabel(cn("成交量", "Volume"), fontsize=8, color=TEXT_SEC)
        ax_v.yaxis.set_major_formatter(plt.FuncFormatter(
            lambda x, _: (f"{x/1e9:.1f}B" if x >= 1e9
                          else f"{x/1e6:.1f}M" if x >= 1e6
                          else f"{x/1e3:.0f}K")
        ))

    # X 轴
    tick_step = max(1, (n_hist + n_pred) // 14)
    tick_pos  = list(range(0, n_hist + n_pred, tick_step))
    tick_lbs  = [all_dates[i].strftime("%m/%d")
                 for i in tick_pos if i < len(all_dates)]
    ax_v.set_xticks(tick_pos[:len(tick_lbs)])
    ax_v.set_xticklabels(tick_lbs, fontsize=8, rotation=30, ha="right")
    ax_k.tick_params(labelbottom=False)

    # 网格
    ax_k.grid(True, axis="y", alpha=0.35)
    ax_v.grid(True, axis="y", alpha=0.25)

    # Y 轴范围
    all_px = pd.concat([df[["open","high","low","close"]],
                        pred_df[["open","high","low","close"]]])
    p_min = all_px.min().min() * 0.994
    p_max = all_px.max().max() * 1.006
    ax_k.set_ylim(p_min, p_max)
    ax_k.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.2f}"))

    # 统计标注
    last_close = df["close"].iloc[-1]
    pred_close = pred_df["close"].iloc[-1]
    chg        = (pred_close - last_close) / last_close * 100
    chg_color  = UP_P if chg >= 0 else DN_P
    chg_arrow  = "▲" if chg >= 0 else "▼"

    info = (f"  {cn('当前','Now')}: {last_close:.2f}   "
            f"{cn('预测末日','Forecast End')}: {pred_close:.2f}   "
            f"{cn('预期','Exp')}: {chg_arrow} {abs(chg):.2f}%  ")
    ax_k.text(0.005, 0.975, info, transform=ax_k.transAxes,
              fontsize=9.5, color=chg_color, va="top",
              bbox=dict(boxstyle="round,pad=0.35", facecolor=PANEL_BG,
                        edgecolor=chg_color, alpha=0.88))

    # MA 线（5日、20日）
    closes_all = list(df["close"].values) + list(pred_df["close"].values)
    x_all      = list(range(len(closes_all)))
    for win, col, lbl in [(5, "#f0c040", "MA5"), (20, "#a0c8ff", "MA20")]:
        ma = pd.Series(closes_all).rolling(win).mean().values
        ax_k.plot(x_all, ma, color=col, linewidth=0.9,
                  alpha=0.7, zorder=1, label=lbl)

    # 图例
    legend_elems = [
        mpatches.Patch(color=UP_H, label=cn("历史上涨", "Hist Up")),
        mpatches.Patch(color=DN_H, label=cn("历史下跌", "Hist Down")),
        mpatches.Patch(color=UP_P, label=cn("预测上涨", "Pred Up"), alpha=0.88),
        mpatches.Patch(color=DN_P, label=cn("预测下跌", "Pred Down"), alpha=0.88),
    ]
    if sample_paths:
        legend_elems.append(
            mpatches.Patch(color=ACCENT, alpha=0.25,
                           label=f"80% CI ({samples} {cn('路径','paths')})"))
    handles, labels = ax_k.get_legend_handles_labels()
    ax_k.legend(handles=legend_elems + handles,
                loc="upper left", facecolor=PANEL_BG, edgecolor=GRID,
                labelcolor=TEXT_PRI, fontsize=8, framealpha=0.9,
                bbox_to_anchor=(0.005, 0.88))

    # 预测区间标签
    ax_k.text(n_hist + n_pred / 2, p_max * 0.9994,
              f"◀  Kronos {cn('预测区间','Forecast')}  ▶",
              ha="center", va="top", fontsize=8, color=GOLD,
              bbox=dict(boxstyle="round", facecolor=BG,
                        edgecolor=GOLD, alpha=0.75))

    # 主标题
    now_str  = datetime.now().strftime("%Y-%m-%d %H:%M")
    mode_lbl = cn("演示模式","Demo") if is_demo else f"Kronos-{model_size}"
    fig.suptitle(
        f"  KRONOS  ·  {ticker.upper()}  "
        f"{cn('K线预测','K-Line Forecast')}  [{mode_lbl}]",
        fontsize=16, fontweight="bold", color=TEXT_PRI,
        x=0.01, ha="left", y=0.985
    )
    ax_k.text(1.0, 1.018,
              f"samples={samples}  |  {now_str}  |  "
              f"zdoc.app/zh/shiyu-coder/Kronos",
              transform=ax_k.transAxes, fontsize=7,
              color=TEXT_SEC, ha="right")

    # 右侧配色说明
    ax_k.text(0.995, 0.975,
              cn("红涨绿跌（A股风格）", "Red=Up  Green=Down"),
              transform=ax_k.transAxes, fontsize=8,
              color=TEXT_SEC, va="top", ha="right")

    plt.tight_layout(rect=[0, 0, 1, 0.975])

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight",
                    facecolor=BG, edgecolor="none")
        print(f"💾 {cn('图表已保存','Saved')}: {output_path}")
    else:
        plt.show()

# ─── 主程序 ──────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    print("\n" + "═" * 62)
    print(f"  KRONOS K线预测器  |  {cn('红涨绿跌 A股配色','Red=Up Green=Down')}")
    print(f"  模型: Kronos-{args.model}  |  "
          f"{cn('历史','Hist')}: {args.period}天  |  "
          f"{cn('预测','Pred')}: {args.pred_len}天  |  "
          f"{cn('采样','Samples')}: {args.samples}")
    print(f"  来源: https://www.zdoc.app/zh/shiyu-coder/Kronos")
    print("═" * 62 + "\n")

    # 1. 获取数据
    df = fetch_data(args.ticker, args.period)

    # 2. 预测
    sample_paths = None
    is_demo      = args.demo

    if not is_demo:
        pred_df = kronos_predict(
            df, args.pred_len, args.model,
            args.samples, args.top_p, args.temperature,
            args.kronos_path, args.device
        )
        if pred_df is None:
            is_demo = True

    if is_demo:
        pred_df, sample_paths = demo_predict(
            df, args.pred_len, args.samples, args.temperature
        )

    print(f"\n{cn('预测结果','Forecast')} ({args.ticker}):")
    print(pred_df[["timestamps","open","high","low","close"]].to_string(index=False))

    # 3. 绘图
    print(f"\n{cn('生成K线图','Rendering chart')}...")
    plot_klines(df, pred_df, args.ticker, args.model, args.samples,
                sample_paths, args.output, is_demo)
    print(f"\n✅ {cn('完成！','Done!')}")


if __name__ == "__main__":
    main()
