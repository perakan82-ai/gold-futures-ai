"""Backtest แบบแยก In-sample / Out-of-sample (กัน overfitting)
- ใช้เฉพาะคะแนน technical รายวัน (ข่าว/มหภาคย้อนหลังไม่ครบ จึงไม่รวม)
- เลือก threshold ที่ดีที่สุดในช่วง In-sample (70% แรก) แล้วล็อกค่า ไปทดสอบช่วง Out-of-sample (30% หลัง)
- เข้าที่ราคาเปิดแท่งถัดไป · SL/TP ตาม ATR · หัก cost ต่อไม้"""
from __future__ import annotations

import numpy as np
import pandas as pd

import indicators as I


def simulate(d: pd.DataFrame, score: pd.Series, th: float, sl_atr: float = 1.5, tp_atr: float = 3.0,
             max_hold: int = 20, cost_atr: float = 0.05) -> list[dict]:
    o, h, l, c, atr = (d[k].values for k in ("open", "high", "low", "close", "atr"))
    s = score.values
    trades, i, n = [], 1, len(d)
    while i < n - 1:
        sig = 1 if s[i] > th else (-1 if s[i] < -th else 0)
        if sig == 0 or not np.isfinite(atr[i]):
            i += 1
            continue
        e = o[i + 1]
        sl, tp = e - sig * sl_atr * atr[i], e + sig * tp_atr * atr[i]
        res, j = None, i + 1
        while j < min(n, i + 1 + max_hold):
            hit_sl = (l[j] <= sl) if sig == 1 else (h[j] >= sl)
            hit_tp = (h[j] >= tp) if sig == 1 else (l[j] <= tp)
            if hit_sl:  # แตะทั้งคู่ในแท่งเดียว → นับว่าโดน SL (มองแง่ร้าย)
                res = -sl_atr
                break
            if hit_tp:
                res = tp_atr
                break
            j += 1
        if res is None:
            j = min(j, n - 1)
            res = sig * (c[j] - e) / atr[i]
        trades.append({"entry_time": d.index[i + 1], "side": sig, "R": (res - cost_atr) / sl_atr, "bars": j - i})
        i = j + 1
    return trades


def metrics(tr: list[dict]) -> dict:
    if not tr:
        return {"trades": 0, "win%": np.nan, "avg_R": np.nan, "total_R": 0.0, "max_dd_R": 0.0, "pf": np.nan}
    r = np.array([t["R"] for t in tr])
    eq = np.cumsum(r)
    dd = float((np.maximum.accumulate(np.r_[0, eq]) - np.r_[0, eq]).max())
    gp, gl = r[r > 0].sum(), -r[r < 0].sum()
    return {"trades": len(r), "win%": float((r > 0).mean() * 100), "avg_R": float(r.mean()),
            "total_R": float(r.sum()), "max_dd_R": dd, "pf": float(gp / gl) if gl > 0 else np.inf}


def run(d1_raw: pd.DataFrame, oos: float = 0.3, grid=(0.2, 0.3, 0.4, 0.5, 0.6)) -> dict:
    d = I.add_all(d1_raw).dropna(subset=["ema200", "atr"])
    if len(d) < 300:
        return {"ok": False, "error": "ข้อมูลไม่พอสำหรับ backtest (ต้อง ≥ 300 แท่งหลัง warm-up)"}
    score = I.tech_votes(d)["score"]
    cut = int(len(d) * (1 - oos))
    ins, outs = d.iloc[:cut], d.iloc[cut:]
    table = []
    for th in grid:
        m = metrics(simulate(ins, score.iloc[:cut], th))
        table.append({"threshold": th, **m})
    valid = [t for t in table if t["trades"] >= 20]
    best = max(valid or table, key=lambda t: t["avg_R"] if np.isfinite(t["avg_R"]) else -9)
    tr_oos = simulate(outs, score.iloc[cut:], best["threshold"])
    m_oos = metrics(tr_oos)
    verdict = ("ผ่าน: Out-of-sample ยังได้ผลบวก" if m_oos["trades"] >= 10 and m_oos["avg_R"] > 0 else
               "ไม่ผ่าน: Out-of-sample ไม่เป็นบวก หรือจำนวนไม้น้อยเกินไป — อย่าใช้สัญญาณ technical นี้เพียงลำพัง")
    return {"ok": True, "grid": pd.DataFrame(table), "best_th": best["threshold"], "in_sample": best,
            "oos": m_oos, "oos_trades": pd.DataFrame(tr_oos), "split": d.index[cut], "verdict": verdict}
