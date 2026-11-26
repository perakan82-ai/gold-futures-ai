"""Technical indicators (เขียนเองด้วย pandas ไม่พึ่งไลบรารีภายนอก)
ของเดิมในโปรเจ็กต์คริปโต: EMA, RSI, MACD, ATR/ADX, Bollinger, VWAP, OBV
เพิ่มใหม่สำหรับทอง: Supertrend, Ichimoku, Keltner + Squeeze, Stochastic RSI, Donchian"""
from __future__ import annotations

import numpy as np
import pandas as pd


def ema(s: pd.Series, n: int) -> pd.Series:
    out = s.ewm(span=n, adjust=False).mean()
    out.iloc[: n - 1] = np.nan
    return out


def rma(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(alpha=1 / n, adjust=False).mean()


def true_range(d: pd.DataFrame) -> pd.Series:
    pc = d["close"].shift()
    return pd.concat([d["high"] - d["low"], (d["high"] - pc).abs(), (d["low"] - pc).abs()], axis=1).max(axis=1)


def rsi(c: pd.Series, n: int = 14) -> pd.Series:
    delta = c.diff()
    up, dn = rma(delta.clip(lower=0), n), rma(-delta.clip(upper=0), n)
    return (100 - 100 / (1 + up / dn.replace(0, np.nan))).fillna(50)


def supertrend(d: pd.DataFrame, n: int = 10, mult: float = 3.0) -> tuple[pd.Series, pd.Series]:
    """คืน (เส้น supertrend, ทิศ +1/−1)"""
    atr = rma(true_range(d), n)
    hl2 = (d["high"] + d["low"]) / 2
    ub, lb = (hl2 + mult * atr).values, (hl2 - mult * atr).values
    c = d["close"].values
    fu, fl = ub.copy(), lb.copy()
    dirn = np.ones(len(d))
    line = np.full(len(d), np.nan)
    for i in range(1, len(d)):
        if np.isnan(atr.iloc[i]):
            continue
        fu[i] = ub[i] if (ub[i] < fu[i - 1] or c[i - 1] > fu[i - 1]) else fu[i - 1]
        fl[i] = lb[i] if (lb[i] > fl[i - 1] or c[i - 1] < fl[i - 1]) else fl[i - 1]
        if dirn[i - 1] == 1:
            dirn[i] = -1 if c[i] < fl[i] else 1
        else:
            dirn[i] = 1 if c[i] > fu[i] else -1
        line[i] = fl[i] if dirn[i] == 1 else fu[i]
    return pd.Series(line, d.index), pd.Series(dirn, d.index)


def add_all(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    d = df.copy()
    c, h, l = d["close"], d["high"], d["low"]
    for n in (9, 21, 50, 200):
        d[f"ema{n}"] = ema(c, n)
    d["rsi"] = rsi(c)
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    d["macd"], d["macd_sig"] = macd, macd.ewm(span=9, adjust=False).mean()
    d["macd_h"] = d["macd"] - d["macd_sig"]

    tr = true_range(d)
    d["atr"] = rma(tr, 14)
    upm, dnm = h.diff(), -l.diff()
    pdm = pd.Series(np.where((upm > dnm) & (upm > 0), upm, 0.0), d.index)
    mdm = pd.Series(np.where((dnm > upm) & (dnm > 0), dnm, 0.0), d.index)
    d["pdi"] = 100 * rma(pdm, 14) / d["atr"]
    d["mdi"] = 100 * rma(mdm, 14) / d["atr"]
    d["adx"] = rma(100 * (d["pdi"] - d["mdi"]).abs() / (d["pdi"] + d["mdi"]).replace(0, np.nan), 14)

    m, sd = c.rolling(20).mean(), c.rolling(20).std()
    d["bb_mid"], d["bb_up"], d["bb_lo"] = m, m + 2 * sd, m - 2 * sd
    d["bb_z"] = (c - m) / sd.replace(0, np.nan)
    # Keltner (EMA20 ± 1.5 ATR) + Squeeze: BB อยู่ในกรอบ Keltner = ความผันผวนบีบตัว มักตามด้วยการเลือกทาง
    km = ema(c, 20)
    d["kc_up"], d["kc_lo"] = km + 1.5 * d["atr"], km - 1.5 * d["atr"]
    d["squeeze"] = (d["bb_up"] < d["kc_up"]) & (d["bb_lo"] > d["kc_lo"])

    d["st_line"], d["st_dir"] = supertrend(d)

    # Ichimoku (9, 26, 52) · เมฆ ณ แท่งปัจจุบัน = span ที่คำนวณไว้ 26 แท่งก่อน
    tenkan = (h.rolling(9).max() + l.rolling(9).min()) / 2
    kijun = (h.rolling(26).max() + l.rolling(26).min()) / 2
    span_a = (tenkan + kijun) / 2
    span_b = (h.rolling(52).max() + l.rolling(52).min()) / 2
    d["tenkan"], d["kijun"] = tenkan, kijun
    d["cloud_a"], d["cloud_b"] = span_a.shift(26), span_b.shift(26)
    d["cloud_top"] = d[["cloud_a", "cloud_b"]].max(axis=1)
    d["cloud_bot"] = d[["cloud_a", "cloud_b"]].min(axis=1)

    r = d["rsi"]
    st = (r - r.rolling(14).min()) / (r.rolling(14).max() - r.rolling(14).min()).replace(0, np.nan)
    d["stochrsi_k"] = (st.rolling(3).mean() * 100)
    d["stochrsi_d"] = d["stochrsi_k"].rolling(3).mean()

    d["dc_up"], d["dc_lo"] = h.rolling(20).max().shift(1), l.rolling(20).min().shift(1)

    vol = d["volume"].replace(0, np.nan)
    tp = (h + l + c) / 3
    d["vwap20"] = (tp * vol).rolling(20, min_periods=5).sum() / vol.rolling(20, min_periods=5).sum()
    d["obv"] = (np.sign(c.diff().fillna(0)) * d["volume"]).cumsum()
    return d


def tech_votes(d: pd.DataFrame) -> pd.DataFrame:
    """โหวตทิศทางรายแท่ง (+1 ขึ้น / −1 ลง / 0 กลาง) — ใช้ทั้งแผนปัจจุบันและ backtest
    ADX ต่ำ (ตลาดไซด์เวย์) จะลดน้ำหนักโหวตแนวโน้ม"""
    if d.empty:
        return pd.DataFrame()
    c = d["close"]
    v = pd.DataFrame(index=d.index)
    v["ema_stack"] = np.select([(c > d["ema50"]) & (d["ema50"] > d["ema200"]),
                                (c < d["ema50"]) & (d["ema50"] < d["ema200"])], [1, -1], 0)
    v["ema_fast"] = np.sign(d["ema9"] - d["ema21"]).fillna(0)
    v["supertrend"] = d["st_dir"].fillna(0)
    v["ichimoku"] = np.select([c > d["cloud_top"], c < d["cloud_bot"]], [1, -1], 0)
    v["macd"] = np.sign(d["macd_h"]).fillna(0)
    v["rsi"] = np.select([d["rsi"] > 55, d["rsi"] < 45], [1, -1], 0)
    v["donchian"] = np.select([c > d["dc_up"], c < d["dc_lo"]], [1, -1], 0)
    v["dmi"] = np.sign(d["pdi"] - d["mdi"]).fillna(0)
    w = {"ema_stack": 2.0, "ema_fast": 1.0, "supertrend": 1.5, "ichimoku": 1.5, "macd": 1.0,
         "rsi": 1.0, "donchian": 1.0, "dmi": 1.0}
    raw = sum(v[k] * wt for k, wt in w.items()) / sum(w.values())
    trend_conf = np.clip((d["adx"].fillna(15) - 15) / 15, 0.35, 1.0)  # ADX 15→0.35 … 30+→1.0
    v["score"] = (raw * trend_conf).clip(-1, 1)
    return v
