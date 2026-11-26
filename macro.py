"""ปัจจัยมหภาค + ตลาดที่เกี่ยวข้อง + COT
- Real yield (DFII10) ลง = หนุนทอง · DXY ลง = หนุนทอง (ความสัมพันธ์ผกผันระยะสั้น)
- ตรวจว่าความสัมพันธ์ยังใช้ได้ไหมด้วย rolling correlation แล้วปรับน้ำหนักตามนั้น
- GVZ = ความผันผวนคาดการณ์ 30 วัน (ใช้ปรับความเสี่ยง ไม่ใช่ทิศทาง)
- Gold/Silver ratio = มุมมอง sentiment (แสดงเป็นบริบท ไม่นำไปคิดคะแนน)
- COT Managed Money: แน่นเกินไป (percentile สูง/ต่ำมาก) = เสี่ยงกลับตัว"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config as C


def _align(a: pd.Series, b: pd.Series) -> pd.DataFrame:
    if a.empty or b.empty:
        return pd.DataFrame()
    a = a.copy(); b = b.copy()
    a.index = a.index.normalize(); b.index = b.index.normalize()
    return pd.concat([a.groupby(level=0).last(), b.groupby(level=0).last()], axis=1, join="inner").dropna()


def _chg(s: pd.Series, n: int = 20) -> float | None:
    s = s.dropna()
    return float(s.iloc[-1] - s.iloc[-1 - n]) if len(s) > n else None


def _pct(s: pd.Series, n: int = 20) -> float | None:
    s = s.dropna()
    return float(s.iloc[-1] / s.iloc[-1 - n] - 1) * 100 if len(s) > n else None


def analyze(gold_d: pd.DataFrame, side: dict, fred: dict) -> dict:
    gold = gold_d["close"] if not gold_d.empty else pd.Series(dtype=float)
    out: dict = {"components": [], "context": []}

    # ── Real yield ──
    ry = fred.get("real10", pd.Series(dtype=float))
    dry = _chg(ry, 20)
    if dry is not None:
        df = _align(gold.pct_change(), ry.diff())
        corr = float(df.iloc[-120:].corr().iloc[0, 1]) if len(df) > 60 else np.nan
        s = float(np.clip(-dry / 0.25, -1, 1))  # ลง 25bp ใน 20 วัน = +1 เต็ม
        wt = 1.0 if not np.isfinite(corr) else float(np.clip(0.4 + max(0.0, -corr) * 2, 0.4, 1.0))
        out["components"].append({"name": "Real yield 10Y (DFII10)", "value": float(ry.iloc[-1]),
                                  "chg": f"{dry * 100:+.0f} bp / 20 วัน", "score": s, "weight": wt,
                                  "corr120": corr,
                                  "read": "ลดลง → หนุนทอง" if dry < 0 else "สูงขึ้น → กดดันทอง"})
    # ── DXY ──
    dxy = side.get("dxy", pd.Series(dtype=float))
    pdx = _pct(dxy, 20)
    if pdx is not None:
        df = _align(gold.pct_change(), dxy.pct_change())
        corr = float(df.iloc[-120:].corr().iloc[0, 1]) if len(df) > 60 else np.nan
        s = float(np.clip(-pdx / 2.0, -1, 1))  # DXY ลง 2% ใน 20 วัน = +1 เต็ม
        wt = 1.0 if not np.isfinite(corr) else float(np.clip(0.4 + max(0.0, -corr) * 2, 0.4, 1.0))
        out["components"].append({"name": "ดอลลาร์ (DXY)", "value": float(dxy.iloc[-1]),
                                  "chg": f"{pdx:+.2f}% / 20 วัน", "score": s, "weight": wt, "corr120": corr,
                                  "read": "อ่อนค่า → หนุนทอง" if pdx < 0 else "แข็งค่า → กดดันทอง"})
    # ── US10Y nominal (บริบท) ──
    be = fred.get("be5", pd.Series(dtype=float))
    dbe = _chg(be, 20)
    if dbe is not None:
        out["context"].append({"name": "Breakeven เงินเฟ้อ 5Y (T5YIE)", "value": float(be.iloc[-1]),
                               "chg": f"{dbe * 100:+.0f} bp / 20 วัน",
                               "read": "ตลาดคาดเงินเฟ้อสูงขึ้น" if dbe > 0 else "ตลาดคาดเงินเฟ้อลดลง"})
    # ── GVZ ──
    gvz = side.get("gvz", pd.Series(dtype=float)).dropna()
    if not gvz.empty:
        g = float(gvz.iloc[-1])
        regime = "ต่ำ" if g < C.GVZ_LOW else ("ปกติ" if g <= C.GVZ_NORMAL else "สูง")
        out["gvz"] = {"value": g, "regime": regime,
                      "pct_1y": float((gvz.tail(252) <= g).mean() * 100),
                      "implied_daily_move_pct": g / np.sqrt(252)}
    # ── Gold/Silver ratio ──
    sil = side.get("silver", pd.Series(dtype=float))
    df = _align(gold, sil)
    if len(df) > 260:
        ratio = df.iloc[:, 0] / df.iloc[:, 1]
        z = float((ratio.iloc[-1] - ratio.tail(252).mean()) / ratio.tail(252).std())
        out["context"].append({"name": "Gold/Silver ratio", "value": float(ratio.iloc[-1]),
                               "chg": f"z-score 1 ปี {z:+.2f}",
                               "read": "ทองแพงกว่าเงินเทียบอดีต (มักสะท้อนความกังวล)" if z > 1 else
                               ("เงินแรงกว่าทอง (มักสะท้อน risk-on)" if z < -1 else "อยู่ในช่วงปกติ")})
    # ── คะแนนรวมมหภาค ──
    comps = out["components"]
    if comps:
        tw = sum(c["weight"] for c in comps)
        out["score"] = float(sum(c["score"] * c["weight"] for c in comps) / tw)
    else:
        out["score"] = 0.0
    return out


def cot_view(cot: pd.DataFrame) -> dict:
    if cot is None or cot.empty or len(cot) < 20:
        return {"score": 0.0, "available": False}
    net = cot["mm_net"]
    win = net.tail(156)  # ~3 ปี
    pct = float((win <= net.iloc[-1]).mean() * 100)
    wk = float(net.iloc[-1] - net.iloc[-2])
    # แน่นเกินไป → เสี่ยงกลับตัว (คะแนนสวน) · ไม่แน่น → ใช้การเปลี่ยนแปลงรายสัปดาห์เป็นแรงส่ง
    if pct >= 85:
        s, read = -0.6, "Managed Money ถือ Long แน่นมาก → เสี่ยงเทขายถ้าหลุดแนวรับ"
    elif pct <= 15:
        s, read = 0.6, "Managed Money ถือ Long น้อยมาก → มีเงินที่อาจกลับเข้าซื้อได้"
    else:
        s = float(np.clip(wk / max(win.diff().abs().mean(), 1), -1, 1)) * 0.4
        read = "สถานะอยู่ช่วงกลาง — ดูทิศการเปลี่ยนแปลงรายสัปดาห์"
    return {"available": True, "date": cot.index[-1], "net": float(net.iloc[-1]), "wk_chg": wk,
            "pct_3y": pct, "pct_oi": float(cot["mm_net_pct_oi"].iloc[-1]), "score": s, "read": read,
            "series": cot}
