"""แนวรับ–แนวต้าน: swing clustering + Pivot Points รายวัน + Fibonacci"""
from __future__ import annotations

import numpy as np
import pandas as pd


def swings(d: pd.DataFrame, w: int = 5) -> tuple[pd.Series, pd.Series]:
    h, l = d["high"], d["low"]
    hi = h[(h == h.rolling(2 * w + 1, center=True).max())]
    lo = l[(l == l.rolling(2 * w + 1, center=True).min())]
    return hi.dropna(), lo.dropna()


def cluster_levels(d: pd.DataFrame, atr: float, w: int = 5, lookback: int = 250, tol_atr: float = 0.5) -> list[dict]:
    """รวม swing high/low ที่ใกล้กัน (ภายใน 0.5 ATR) เป็นโซน · strength = จำนวนครั้งที่ราคาแตะ"""
    x = d.tail(lookback)
    if len(x) < 2 * w + 2 or not np.isfinite(atr) or atr <= 0:
        return []
    hi, lo = swings(x, w)
    pts = sorted([(p, "H") for p in hi.values] + [(p, "L") for p in lo.values])
    groups: list[list[float]] = []
    for p, _ in pts:
        if groups and p - np.mean(groups[-1]) <= tol_atr * atr:
            groups[-1].append(p)
        else:
            groups.append([p])
    last_t = len(x)
    out = []
    for g in groups:
        lvl = float(np.mean(g))
        out.append({"price": lvl, "touches": len(g), "src": "swing"})
    return sorted(out, key=lambda r: r["price"]) if last_t else out


def pivots(d1: pd.DataFrame) -> dict:
    """Classic pivots จากแท่งวันก่อนหน้า (แท่งที่ปิดแล้ว)"""
    if len(d1) < 2:
        return {}
    p = d1.iloc[-2]
    P = (p.high + p.low + p.close) / 3
    rng = p.high - p.low
    return {"P": P, "R1": 2 * P - p.low, "S1": 2 * P - p.high, "R2": P + rng, "S2": P - rng,
            "R3": p.high + 2 * (P - p.low), "S3": p.low - 2 * (p.high - P)}


def fib(d1: pd.DataFrame, lookback: int = 120) -> dict:
    """Retracement มาตรฐาน + 0.886 (Gartley/Bat ratio ที่ใช้กันทั่วไปในทองคำ/FX ไม่ใช่เฉพาะคริปโต)

    ไม่ใส่ 0.942 แบบที่ทำใน crypto-scanner — ระดับนั้นสะท้อนการย่อลึกจากภาวะ leverage เกินตัวที่
    พบมากในคริปโต ไม่ใช่พฤติกรรมทั่วไปของทองคำ ใส่ไปโดยไม่มีหลักฐานรองรับจะเป็นแค่เลขลอยๆ
    """
    x = d1.tail(lookback)
    if x.empty:
        return {}
    hi_i, lo_i = x["high"].idxmax(), x["low"].idxmin()
    hi, lo = float(x["high"].max()), float(x["low"].min())
    up = lo_i < hi_i  # ขาขึ้นล่าสุด: วัด retracement ลงจาก high
    rng = hi - lo
    lv = {f"{r:.3f}": (hi - r * rng) if up else (lo + r * rng) for r in (0.236, 0.382, 0.5, 0.618, 0.786, 0.886)}
    return {"swing_high": hi, "swing_low": lo, "direction": "up" if up else "down", "levels": lv}


def build(d1: pd.DataFrame, d4: pd.DataFrame, price: float, atr_d: float, d1h: pd.DataFrame | None = None,
          intraday: bool = False) -> dict:
    """swing = Swing (ATR 1D, แนวจาก 1D+4H) · intraday = ATR 4H, แนวจาก 4H+1H + pivot/fib/EMA รายวัน"""
    lv: list[dict] = []
    if intraday:
        atr_4 = float(d4["atr"].iloc[-1])
        lv += [dict(r, tf="4H") for r in cluster_levels(d4, atr_4, w=4, lookback=300)]
        if d1h is not None and not d1h.empty:
            lv += [dict(r, tf="1H") for r in cluster_levels(d1h, float(d1h["atr"].iloc[-1]), w=5, lookback=300)]
    else:
        lv += [dict(r, tf="1D") for r in cluster_levels(d1, atr_d, w=5, lookback=250)]
        if d4 is not None and not d4.empty and "atr" in d4:
            lv += [dict(r, tf="4H") for r in cluster_levels(d4, float(d4["atr"].iloc[-1]), w=4, lookback=300)]
    pv = pivots(d1)
    lv += [{"price": v, "touches": 1, "src": f"pivot {k}", "tf": "1D"} for k, v in pv.items()]
    fb = fib(d1)
    lv += [{"price": v, "touches": 1, "src": f"fib {k}", "tf": "1D"} for k, v in fb.get("levels", {}).items()]
    for n in (50, 200):
        col = f"ema{n}"
        if col in d1 and np.isfinite(d1[col].iloc[-1]):
            lv.append({"price": float(d1[col].iloc[-1]), "touches": 1, "src": f"EMA{n} 1D", "tf": "1D"})
    # รวมระดับที่ซ้อนกัน (ภายใน 0.3 ATR) → confluence
    lv = sorted(lv, key=lambda r: r["price"])
    merged: list[dict] = []
    for r in lv:
        if merged and abs(r["price"] - merged[-1]["price"]) <= 0.3 * atr_d:
            m = merged[-1]
            n = m["n"] + 1
            m["price"] = (m["price"] * m["n"] + r["price"]) / n
            m["n"], m["touches"] = n, m["touches"] + r["touches"]
            m["src"] = m["src"] + " + " + r["src"] if r["src"] not in m["src"] else m["src"]
        else:
            merged.append(dict(r, n=1))
    for m in merged:
        m["strength"] = min(5, m["touches"] + (m["n"] - 1))
    sup = [m for m in merged if m["price"] < price - 0.1 * atr_d][::-1]
    res = [m for m in merged if m["price"] > price + 0.1 * atr_d]
    return {"supports": sup[:6], "resistances": res[:6], "pivots": pv, "fib": fb, "all": merged}
