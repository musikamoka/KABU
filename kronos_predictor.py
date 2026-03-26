#!/usr/bin/env python3
"""
Kronos K-Line Predictor  v3  —— 增强版
========================================
新增：
  · RSI / MACD / 布林带 技术指标叠加显示
  · 多次采样取中位数（比平均更鲁棒）
  · 自动 CUDA 检测 + 混合精度推理（FP16）提速
  · --samples 默认 20，统计分布更稳定
  · 置信区间改为 P25-P75（更保守的区间）
  · 预测结果附带技术面信号摘要

使用:
    python kronos_predictor.py --ticker MU --pred_len 5
    python kronos_predictor.py --ticker MU --pred_len 5 --device cuda
    python kronos_predictor.py --ticker AAPL --pred_len 10 --samples 30 --demo

模型来源: https://www.zdoc.app/zh/shiyu-coder/Kronos
"""

import argparse, sys, os, warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from datetime import datetime, timedelta

# ─── 中文字体 ────────────────────────────────────────────────────────────────

def setup_chinese_font():
    import matplotlib.font_manager as fm
    candidates = [
        "Microsoft YaHei","SimHei","SimSun","KaiTi",
        "PingFang SC","Heiti SC","STHeiti","Hiragino Sans GB","Arial Unicode MS",
        "WenQuanYi Micro Hei","WenQuanYi Zen Hei","Noto Sans CJK SC",
        "Source Han Sans CN","Droid Sans Fallback",
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    for font in candidates:
        if font in available:
            matplotlib.rcParams["font.sans-serif"]  = [font, "DejaVu Sans"]
            matplotlib.rcParams["axes.unicode_minus"] = False
            print(f"✅ 中文字体: {font}")
            return True
    matplotlib.rcParams["font.sans-serif"]  = ["DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False
    print("⚠️  未找到中文字体，使用英文标签")
    return False

HAS_CN = setup_chinese_font()
def cn(z, e): return z if HAS_CN else e

# ─── 配色 ────────────────────────────────────────────────────────────────────
BG       = "#0d1117"
PANEL_BG = "#161b22"
GRID     = "#21262d"
UP_H     = "#e84040"
DN_H     = "#26a84e"
UP_P     = "#ff6b6b"
DN_P     = "#3fcf70"
ACCENT   = "#58a6ff"
TEXT_PRI = "#e6edf3"
TEXT_SEC = "#8b949e"
GOLD     = "#d29922"
PURPLE   = "#bc8cff"

# ─── 参数 ────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Kronos K线预测器 v3")
    p.add_argument("--ticker",      default="AAPL")
    p.add_argument("--period",      type=int,   default=120,   help="历史天数")
    p.add_argument("--pred_len",    type=int,   default=5,     help="预测天数")
    p.add_argument("--model",       default="base", choices=["mini","small","base"])
    p.add_argument("--samples",     type=int,   default=20,    help="采样路径数（默认20）")
    p.add_argument("--top_p",       type=float, default=0.9)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--output",      default=None)
    p.add_argument("--demo",        action="store_true")
    p.add_argument("--kronos_path", default=None)
    p.add_argument("--device",      default="auto")
    p.add_argument("--fp16",        action="store_true",
                   help="FP16 混合精度推理（GPU显存不足时使用）")
    return p.parse_args()

# ─── 技术指标 ────────────────────────────────────────────────────────────────

def calc_rsi(closes: np.ndarray, period=14) -> np.ndarray:
    delta = np.diff(closes, prepend=closes[0])
    gain  = np.where(delta > 0, delta, 0.0)
    loss  = np.where(delta < 0, -delta, 0.0)
    avg_g = pd.Series(gain).ewm(com=period-1, min_periods=period).mean().values
    avg_l = pd.Series(loss).ewm(com=period-1, min_periods=period).mean().values
    rs    = np.where(avg_l == 0, 100, avg_g / (avg_l + 1e-10))
    return 100 - (100 / (1 + rs))

def calc_macd(closes: np.ndarray, fast=12, slow=26, signal=9):
    s   = pd.Series(closes)
    ema_f = s.ewm(span=fast,   adjust=False).mean()
    ema_s = s.ewm(span=slow,   adjust=False).mean()
    macd  = ema_f - ema_s
    sig   = macd.ewm(span=signal, adjust=False).mean()
    hist  = macd - sig
    return macd.values, sig.values, hist.values

def calc_boll(closes: np.ndarray, period=20, std_mult=2.0):
    s    = pd.Series(closes)
    mid  = s.rolling(period).mean()
    std  = s.rolling(period).std()
    return mid.values, (mid + std_mult*std).values, (mid - std_mult*std).values

def tech_signal_summary(df: pd.DataFrame) -> str:
    """生成简短的技术面信号摘要"""
    closes = df["close"].values
    rsi    = calc_rsi(closes)[-1]
    _, _, macd_hist = calc_macd(closes)
    macd_trend = "↑多" if macd_hist[-1] > macd_hist[-3] else "↓空"
    mid, upper, lower = calc_boll(closes)
    last = closes[-1]
    if last > upper[-1]:   boll_pos = cn("超买(上轨外)","Overbought")
    elif last < lower[-1]: boll_pos = cn("超卖(下轨外)","Oversold")
    else:                  boll_pos = cn("中性","Neutral")

    ma5  = np.mean(closes[-5:])
    ma20 = np.mean(closes[-20:]) if len(closes) >= 20 else np.mean(closes)
    trend = cn("多头排列","Bull") if ma5 > ma20 else cn("空头排列","Bear")

    rsi_lbl = (cn("超买","OB") if rsi > 70 else
               cn("超卖","OS") if rsi < 30 else
               cn("中性","Neutral"))

    return (f"RSI({rsi:.1f} {rsi_lbl})  "
            f"MACD({macd_trend})  "
            f"BOLL({boll_pos})  "
            f"MA({trend})")

# ─── 数据获取 ────────────────────────────────────────────────────────────────

def fetch_data(ticker: str, period_days: int) -> pd.DataFrame:
    try:
        import yfinance as yf
    except ImportError:
        print("❌ pip install yfinance"); sys.exit(1)

    print(f"📡 Yahoo Finance [{ticker}]...")
    end   = datetime.today()
    start = end - timedelta(days=period_days + 40)
    df = yf.Ticker(ticker).history(
        start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"))
    if df.empty:
        print(f"❌ 无数据 [{ticker}]"); sys.exit(1)

    df = df.reset_index()
    df.columns = [c.lower() for c in df.columns]
    df = df.rename(columns={"date":"timestamps"})
    df["timestamps"] = pd.to_datetime(df["timestamps"]).dt.tz_localize(None)
    df = df.tail(period_days).reset_index(drop=True)
    print(f"✅ {len(df)} 个交易日  "
          f"[{df['timestamps'].iloc[0].date()} → {df['timestamps'].iloc[-1].date()}]")
    return df[["timestamps","open","high","low","close","volume"]]

# ─── 设备检测 ────────────────────────────────────────────────────────────────

def resolve_device(device_arg: str) -> str:
    if device_arg != "auto":
        return device_arg
    try:
        import torch
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            print(f"🖥️  GPU: {p.name}  |  显存: {p.total_memory/1024**3:.1f}GB")
            return "cuda"
        if hasattr(torch.backends,"mps") and torch.backends.mps.is_available():
            print("🖥️  设备: Apple MPS"); return "mps"
    except ImportError: pass
    print("🖥️  设备: CPU")
    return "cpu"

# ─── Kronos 真实预测 ─────────────────────────────────────────────────────────

def kronos_predict(df, pred_len, model_size, samples,
                   top_p, temperature, kronos_path, device_arg, use_fp16):

    for p in [p for p in [kronos_path,
                           os.path.dirname(os.path.abspath(__file__)),
                           os.getcwd()] if p]:
        if os.path.isdir(os.path.join(p, "model")):
            if p not in sys.path: sys.path.insert(0, p)
            print(f"✅ Kronos 仓库: {p}"); break

    try:
        from model import Kronos, KronosTokenizer, KronosPredictor
    except ImportError:
        print("⚠️  未找到 Kronos model → 演示模式"); return None

    cfg = {
        "mini":  ("NeoQuasar/Kronos-mini",  "NeoQuasar/Kronos-Tokenizer-2k",  2048),
        "small": ("NeoQuasar/Kronos-small", "NeoQuasar/Kronos-Tokenizer-base", 512),
        "base":  ("NeoQuasar/Kronos-base",  "NeoQuasar/Kronos-Tokenizer-base", 512),
    }
    model_name, tok_name, max_ctx = cfg[model_size]
    device = resolve_device(device_arg)

    if len(df) > max_ctx:
        print(f"⚠️  截断至最近 {max_ctx} 天")
        df = df.tail(max_ctx).reset_index(drop=True)

    print(f"\n🤖 Kronos-{model_size} ({model_name})")
    print(f"   上下文:{len(df)}天  预测:{pred_len}天  "
          f"采样:{samples}  device:{device}  fp16:{use_fp16}")
    print(f"   temperature={temperature}  top_p={top_p}\n")

    try:
        import torch
        tokenizer = KronosTokenizer.from_pretrained(tok_name)
        model     = Kronos.from_pretrained(model_name).to(device)

        # FP16 推理 —— 显著减少显存占用 & 提速（Ampere+ GPU 约 2x）
        if use_fp16 and device.startswith("cuda"):
            model = model.half()
            print("⚡ FP16 混合精度已启用")

        model.eval()
        predictor = KronosPredictor(model, tokenizer, max_context=max_ctx)
        total_p   = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"✅ 模型加载完成  参数量:{total_p:.1f}M\n")
    except Exception as e:
        print(f"❌ 加载失败: {e} → 演示模式"); return None

    x_df        = df[["open","high","low","close","volume"]].copy()
    x_timestamp = df["timestamps"]
    freq        = pd.infer_freq(df["timestamps"]) or "B"
    y_timestamp = pd.Series(pd.date_range(
        start=df["timestamps"].iloc[-1], periods=pred_len+1, freq=freq)[1:])

    print("🔮 推理中...")
    try:
        pred_df = predictor.predict(
            df=x_df, x_timestamp=x_timestamp, y_timestamp=y_timestamp,
            pred_len=pred_len, T=temperature, top_p=top_p,
            sample_count=samples)
        pred_df["timestamps"] = y_timestamp.values

        # ── 影线裁剪：让 high/low 贴近实体，避免全是影线 ──
        # 用历史K线的影线/实体比例来约束预测K线的影线长度
        hist_bodies = (df["close"] - df["open"]).abs()
        hist_ranges = (df["high"]  - df["low"]).abs()
        wick_ratio  = (hist_ranges / hist_bodies.replace(0, np.nan)).median()
        wick_ratio  = float(np.clip(wick_ratio, 1.5, 3.5))  # 限制在合理范围

        for idx in pred_df.index:
            o = pred_df.at[idx, "open"]
            c = pred_df.at[idx, "close"]
            body_h   = max(abs(c - o), abs(o) * 0.002)  # 最小实体
            max_wick = body_h * (wick_ratio - 1) / 2
            pred_df.at[idx, "high"] = max(o, c) + max_wick
            pred_df.at[idx, "low"]  = min(o, c) - max_wick

        print("✅ 推理完成！\n")
        return pred_df
    except Exception as e:
        print(f"❌ 推理失败: {e} → 演示模式"); return None

# ─── 演示模式 ────────────────────────────────────────────────────────────────

def demo_predict(df, pred_len, samples, temperature=1.0):
    print("🎭 演示模式（统计模拟）")
    closes = df["close"].values
    ret    = np.diff(closes) / closes[:-1]
    mu, sigma = ret.mean(), ret.std()
    rec_sig   = ret[-20:].std() if len(ret)>=20 else sigma

    last_date = df["timestamps"].iloc[-1]
    fut = []
    d   = last_date
    while len(fut) < pred_len:
        d += timedelta(days=1)
        if d.weekday() < 5: fut.append(d)

    all_paths = []
    for _ in range(samples):
        path, prev = [], closes[-1]
        for i in range(pred_len):
            shock = np.random.normal(mu*0.3, rec_sig*temperature*(1+i*0.02))
            o = prev*(1+np.random.normal(0, sigma*0.3*temperature))
            c = prev*(1+shock)
            h = max(o,c)*(1+abs(np.random.normal(0,sigma*0.5)))
            l = min(o,c)*(1-abs(np.random.normal(0,sigma*0.5)))
            path.append({"timestamps":fut[i],"open":o,"high":h,"low":l,"close":c})
            prev = c
        all_paths.append(pd.DataFrame(path))

    # 中位数比均值更鲁棒
    avg = pd.DataFrame({
        "timestamps": fut,
        "open":  np.median([p["open"].values  for p in all_paths], axis=0),
        "high":  np.median([p["high"].values  for p in all_paths], axis=0),
        "low":   np.median([p["low"].values   for p in all_paths], axis=0),
        "close": np.median([p["close"].values for p in all_paths], axis=0),
    })
    return avg, all_paths

# ─── 绘图 ────────────────────────────────────────────────────────────────────

def plot_klines(df, pred_df, ticker, model_size, samples,
                sample_paths=None, output_path=None, is_demo=False):

    plt.rcParams.update({
        "axes.facecolor":PANEL_BG, "figure.facecolor":BG,
        "text.color":TEXT_PRI, "axes.labelcolor":TEXT_SEC,
        "xtick.color":TEXT_SEC, "ytick.color":TEXT_SEC,
        "axes.edgecolor":GRID, "grid.color":GRID, "grid.linewidth":0.5,
    })

    # 4行布局：K线 / MACD / RSI / 成交量
    fig = plt.figure(figsize=(22, 13))
    gs  = gridspec.GridSpec(4, 1,
                            height_ratios=[5, 1.5, 1.5, 1],
                            hspace=0.06)
    ax_k    = fig.add_subplot(gs[0])
    ax_macd = fig.add_subplot(gs[1], sharex=ax_k)
    ax_rsi  = fig.add_subplot(gs[2], sharex=ax_k)
    ax_v    = fig.add_subplot(gs[3], sharex=ax_k)

    n_hist    = len(df)
    n_pred    = len(pred_df)
    all_dates = list(df["timestamps"]) + list(pred_df["timestamps"])
    x_hist    = list(range(n_hist))
    x_pred    = list(range(n_hist, n_hist + n_pred))

    for ax in [ax_k, ax_macd, ax_rsi, ax_v]:
        ax.axvline(x=n_hist-0.5, color=GOLD,
                   linewidth=1.2, linestyle="--", alpha=0.8, zorder=3)

    # ── 置信区间（P25-P75，保守） ──
    if sample_paths:
        all_c = np.array([p["close"].values for p in sample_paths])
        p25   = np.percentile(all_c, 25, axis=0)
        p75   = np.percentile(all_c, 75, axis=0)
        p10   = np.percentile(all_c, 10, axis=0)
        p90   = np.percentile(all_c, 90, axis=0)
        ax_k.fill_between(x_pred, p10, p90,
                          color=ACCENT, alpha=0.06, zorder=1,
                          label=cn("P10-P90","P10-P90"))
        ax_k.fill_between(x_pred, p25, p75,
                          color=ACCENT, alpha=0.14, zorder=1,
                          label=cn("P25-P75（核心区间）","P25-P75 Core"))
        for path in sample_paths:
            ax_k.plot(x_pred, path["close"].values,
                      color=ACCENT, alpha=0.08, linewidth=0.6, zorder=1)

    # ── K线蜡烛 ──
    def draw_candles(ax_t, xv, data, uc, dc, alpha=1.0, w=0.6):
        for xi, (_,row) in zip(xv, data.iterrows()):
            o,h,l,c = row["open"],row["high"],row["low"],row["close"]
            col = uc if c>=o else dc
            bh  = max(abs(c-o), abs(o)*0.001)
            ax_t.plot([xi,xi],[l,h], color=col,
                      linewidth=0.9, alpha=alpha, zorder=2)
            ax_t.add_patch(plt.Rectangle(
                (xi-w/2, min(o,c)), w, bh,
                facecolor=col, edgecolor=col,
                linewidth=0.4, alpha=alpha, zorder=2))

    draw_candles(ax_k, x_hist, df,      UP_H, DN_H)
    draw_candles(ax_k, x_pred, pred_df, UP_P, DN_P, alpha=0.9, w=0.55)
    ax_k.axvspan(n_hist-0.5, n_hist+n_pred, color=ACCENT, alpha=0.02, zorder=0)

    # ── MA 线 ──
    closes_all = list(df["close"].values) + list(pred_df["close"].values)
    x_all      = list(range(len(closes_all)))
    for win, col, lbl in [(5,"#f0c040","MA5"),(20,"#a0c8ff","MA20"),(60,"#c080ff","MA60")]:
        if len(closes_all) >= win:
            ma = pd.Series(closes_all).rolling(win).mean().values
            ax_k.plot(x_all, ma, color=col, linewidth=0.85,
                      alpha=0.75, zorder=1, label=lbl)

    # ── 布林带（仅历史部分） ──
    if n_hist >= 20:
        mid, upper, lower = calc_boll(df["close"].values)
        ax_k.plot(x_hist, upper, color=PURPLE, linewidth=0.7,
                  alpha=0.5, linestyle=":", label=cn("布林上轨","BOLL+"))
        ax_k.plot(x_hist, lower, color=PURPLE, linewidth=0.7,
                  alpha=0.5, linestyle=":", label=cn("布林下轨","BOLL-"))
        ax_k.fill_between(x_hist, lower, upper,
                          color=PURPLE, alpha=0.03)

    # ── MACD ──
    macd, sig, hist_macd = calc_macd(df["close"].values)
    ax_macd.plot(x_hist, macd, color="#f0c040", linewidth=0.9, label="MACD")
    ax_macd.plot(x_hist, sig,  color="#a0c8ff", linewidth=0.9, label="Signal")
    colors_macd = [UP_H if v >= 0 else DN_H for v in hist_macd]
    ax_macd.bar(x_hist, hist_macd, color=colors_macd, alpha=0.6, width=0.7)
    ax_macd.axhline(0, color=GRID, linewidth=0.8)
    ax_macd.set_ylabel("MACD", fontsize=8, color=TEXT_SEC)
    ax_macd.legend(fontsize=7, loc="upper left",
                   facecolor=PANEL_BG, edgecolor=GRID, labelcolor=TEXT_PRI)
    ax_macd.axvspan(n_hist-0.5, n_hist+n_pred, color=ACCENT, alpha=0.02)

    # ── RSI ──
    rsi = calc_rsi(df["close"].values)
    ax_rsi.plot(x_hist, rsi, color="#ff9f40", linewidth=1.0, label="RSI(14)")
    ax_rsi.axhline(70, color=UP_H, linewidth=0.7, linestyle="--", alpha=0.6)
    ax_rsi.axhline(30, color=DN_H, linewidth=0.7, linestyle="--", alpha=0.6)
    ax_rsi.axhline(50, color=GRID, linewidth=0.5)
    ax_rsi.fill_between(x_hist, rsi, 70,
                        where=np.array(rsi)>70, color=UP_H, alpha=0.15)
    ax_rsi.fill_between(x_hist, rsi, 30,
                        where=np.array(rsi)<30, color=DN_H, alpha=0.15)
    ax_rsi.set_ylim(0, 100)
    ax_rsi.set_ylabel("RSI", fontsize=8, color=TEXT_SEC)
    ax_rsi.legend(fontsize=7, loc="upper left",
                  facecolor=PANEL_BG, edgecolor=GRID, labelcolor=TEXT_PRI)
    ax_rsi.axvspan(n_hist-0.5, n_hist+n_pred, color=ACCENT, alpha=0.02)

    # ── 成交量 ──
    v_colors = [UP_H if df["close"].iloc[i]>=df["open"].iloc[i] else DN_H
                for i in range(n_hist)]
    ax_v.bar(x_hist, df["volume"].values,
             color=v_colors, alpha=0.7, width=0.7)
    ax_v.set_ylabel(cn("成交量","Vol"), fontsize=8, color=TEXT_SEC)
    ax_v.yaxis.set_major_formatter(plt.FuncFormatter(
        lambda x,_: (f"{x/1e9:.1f}B" if x>=1e9
                     else f"{x/1e6:.1f}M" if x>=1e6
                     else f"{x/1e3:.0f}K")))
    ax_v.axvspan(n_hist-0.5, n_hist+n_pred, color=ACCENT, alpha=0.02)

    # ── X 轴 ──
    tick_step = max(1, (n_hist+n_pred)//14)
    tick_pos  = list(range(0, n_hist+n_pred, tick_step))
    tick_lbs  = [all_dates[i].strftime("%m/%d")
                 for i in tick_pos if i<len(all_dates)]
    ax_v.set_xticks(tick_pos[:len(tick_lbs)])
    ax_v.set_xticklabels(tick_lbs, fontsize=8, rotation=30, ha="right")
    for ax in [ax_k, ax_macd, ax_rsi]:
        ax.tick_params(labelbottom=False)

    for ax in [ax_k, ax_macd, ax_rsi, ax_v]:
        ax.grid(True, axis="y", alpha=0.3)

    # ── Y 轴 ──
    all_px = pd.concat([df[["open","high","low","close"]],
                        pred_df[["open","high","low","close"]]])
    p_min  = all_px.min().min()*0.993
    p_max  = all_px.max().max()*1.007
    ax_k.set_ylim(p_min, p_max)
    ax_k.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x,_: f"{x:.2f}"))

    # ── 统计标注 ──
    last_close = df["close"].iloc[-1]
    pred_close = pred_df["close"].iloc[-1]
    chg        = (pred_close - last_close)/last_close*100
    chg_color  = UP_P if chg>=0 else DN_P
    arrow      = "▲" if chg>=0 else "▼"

    sig_txt = tech_signal_summary(df)
    info = (f"  {cn('当前','Now')}:{last_close:.2f}  "
            f"{cn('预测末日','End')}:{pred_close:.2f}  "
            f"{cn('预期','Exp')}:{arrow}{abs(chg):.2f}%  "
            f"|  {sig_txt}  ")
    ax_k.text(0.005, 0.975, info, transform=ax_k.transAxes,
              fontsize=8.5, color=chg_color, va="top",
              bbox=dict(boxstyle="round,pad=0.35", facecolor=PANEL_BG,
                        edgecolor=chg_color, alpha=0.9))

    # ── 图例 ──
    legend_elems = [
        mpatches.Patch(color=UP_H, label=cn("历史上涨","Hist Up")),
        mpatches.Patch(color=DN_H, label=cn("历史下跌","Hist Down")),
        mpatches.Patch(color=UP_P, label=cn("预测上涨","Pred Up"), alpha=0.9),
        mpatches.Patch(color=DN_P, label=cn("预测下跌","Pred Down"), alpha=0.9),
    ]
    handles, labels = ax_k.get_legend_handles_labels()
    ax_k.legend(handles=legend_elems+handles, loc="upper left",
                facecolor=PANEL_BG, edgecolor=GRID,
                labelcolor=TEXT_PRI, fontsize=7.5, framealpha=0.9,
                bbox_to_anchor=(0.005, 0.87), ncol=2)

    ax_k.text(n_hist+n_pred/2, p_max*0.9994,
              f"◀  Kronos {cn('预测区间','Forecast')}  ▶",
              ha="center", va="top", fontsize=8, color=GOLD,
              bbox=dict(boxstyle="round", facecolor=BG,
                        edgecolor=GOLD, alpha=0.75))

    now_str  = datetime.now().strftime("%Y-%m-%d %H:%M")
    mode_lbl = cn("演示","Demo") if is_demo else f"Kronos-{model_size}"
    fig.suptitle(
        f"  KRONOS  ·  {ticker.upper()}  "
        f"{cn('K线预测','K-Line Forecast')}  [{mode_lbl}]  "
        f"{cn('红涨绿跌','Red=Up Green=Dn')}",
        fontsize=15, fontweight="bold", color=TEXT_PRI,
        x=0.01, ha="left", y=0.988)
    ax_k.text(1.0, 1.015,
              f"samples={samples}  |  {now_str}  |  zdoc.app/zh/shiyu-coder/Kronos",
              transform=ax_k.transAxes, fontsize=7,
              color=TEXT_SEC, ha="right")

    plt.tight_layout(rect=[0,0,1,0.975])

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight",
                    facecolor=BG, edgecolor="none")
        print(f"💾 {cn('已保存','Saved')}: {output_path}")
    else:
        plt.show()

# ─── 主程序 ──────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    print("\n" + "═"*62)
    print(f"  KRONOS K线预测器 v3  |  {cn('红涨绿跌','Red=Up Green=Dn')}")
    print(f"  模型:Kronos-{args.model}  历史:{args.period}天  "
          f"预测:{args.pred_len}天  采样:{args.samples}")
    print(f"  来源: https://www.zdoc.app/zh/shiyu-coder/Kronos")
    print("═"*62 + "\n")

    df = fetch_data(args.ticker, args.period)

    # 技术面摘要
    print(f"📊 {cn('技术面信号','Tech Signal')}: {tech_signal_summary(df)}\n")

    sample_paths = None
    is_demo      = args.demo

    if not is_demo:
        pred_df = kronos_predict(
            df, args.pred_len, args.model,
            args.samples, args.top_p, args.temperature,
            args.kronos_path, args.device, args.fp16)
        if pred_df is None: is_demo = True

    if is_demo:
        pred_df, sample_paths = demo_predict(
            df, args.pred_len, args.samples, args.temperature)

    print(f"\n{cn('预测结果','Forecast')} ({args.ticker}):")
    print(pred_df[["timestamps","open","high","low","close"]].to_string(index=False))

    print(f"\n{cn('生成图表','Rendering')}...")
    plot_klines(df, pred_df, args.ticker, args.model, args.samples,
                sample_paths, args.output, is_demo)
    print(f"\n✅ {cn('完成','Done')}!")

if __name__ == "__main__":
    main()
