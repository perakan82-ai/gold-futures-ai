"""รวมคะแนนทุกส่วน → Bias → แผนเทรด (จุดเข้า / SL / TP / ขนาดสัญญา) + คำเตือน"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

import config as C
import indicators as I
import levels as L
import macro as M
import news as N


def round_lot(x: float, step: float) -> float:
    return math.floor(x / step + 1e-9) * step


def shift_frames(frames: dict, offset: float) -> dict:
    """เลื่อนราคาอ้างอิง (COMEX GC=F) ให้ตรงกับราคา GOLD ใน XM · indicators ส่วนใหญ่ไม่เปลี่ยนเมื่อเลื่อนทั้งชุด"""
    if not offset:
        return frames
    out = {}
    for tf, df in frames.items():
        d = df.copy()
        if not d.empty:
            d[["open", "high", "low", "close"]] += offset
        out[tf] = d
    return out


def analyze(data: dict, equity: float, risk_pct: float = C.RISK_PCT, stop_atr: float = C.STOP_ATR,
            gemini_key: str = "", xm: dict | None = None, xm_price: float = 0.0, intraday: bool = False) -> dict:
    """xm = สเปกบัญชี XM (ดู config.XM) · xm_price = ราคา GOLD ใน XM ตอนนี้ (0 = ไม่ปรับ)"""
    xm = {**C.XM, **(xm or {})}
    ref = data["frames"]["1h"] if not data["frames"]["1h"].empty else data["frames"]["1d"]
    ref_px = float(ref["close"].iloc[-1]) if not ref.empty else 0.0
    offset = (xm_price - ref_px) if (xm_price and ref_px) else 0.0
    fr = {tf: I.add_all(df) for tf, df in shift_frames(data["frames"], offset).items()}
    d1 = fr["1d"]
    if d1.empty or len(d1) < 60:
        return {"ok": False, "error": "ข้อมูลราคารายวันไม่พอ (ต้อง ≥ 60 แท่ง)"}
    price = float(fr["1h"]["close"].iloc[-1]) if not fr["1h"].empty else float(d1["close"].iloc[-1])
    atr_d = float(d1["atr"].iloc[-1])

    tech = {}
    for tf in ("1d", "4h", "1h", "1w"):
        v = I.tech_votes(fr[tf]) if not fr[tf].empty and len(fr[tf]) > 60 else pd.DataFrame()
        tech[tf] = {"score": float(v["score"].iloc[-1]) if not v.empty else 0.0,
                    "votes": v.iloc[-1].drop("score").to_dict() if not v.empty else {}}

    mac = M.analyze(data["frames"]["1d"], data["side"], data["fred"])
    cot = M.cot_view(data["cot"])
    gvz = (mac.get("gvz") or {}).get("value")
    scen = N.event_scenarios(data["calendar"], tech["1d"]["score"], gvz, price)
    sent = N.headline_sentiment(data["news"])
    nsc = N.news_score(scen, sent)

    parts = {"tech_1d": tech["1d"]["score"], "tech_4h": tech["4h"]["score"], "tech_1h": tech["1h"]["score"],
             "macro": mac["score"], "cot": cot["score"], "news": nsc}
    total = 100 * sum(parts[k] * w for k, w in C.WEIGHTS.items()) / sum(C.WEIGHTS.values())
    signs = [np.sign(v) for v in parts.values() if abs(v) > 0.1]
    agree = abs(sum(signs)) / len(signs) if signs else 0.0
    bias = "LONG" if total > C.BIAS_TH else ("SHORT" if total < -C.BIAS_TH else "WAIT")

    use_4h = intraday and not fr["4h"].empty and len(fr["4h"]) > 60
    atr_plan = float(fr["4h"]["atr"].iloc[-1]) if use_4h else atr_d
    lv = L.build(d1, fr["4h"], price, atr_plan, d1h=fr["1h"], intraday=use_4h)
    plans = {side: make_plan(side, price, atr_plan, lv, stop_atr, equity, risk_pct, xm) for side in ("LONG", "SHORT")}

    warn = []
    soon = [s for s in scen if s["impact"] == "High" and s["hours_left"] <= C.EVENT_WARN_H]
    if soon:
        warn.append("มีข่าวแรงภายใน 24 ชม.: " + ", ".join(s["event"] for s in soon) +
                    " — ควรรอหลังข่าว หรือลดขนาดไม้ (ขาดทุนจริงอาจเกิน SL ช่วงข่าว)")
    if gvz and gvz > C.GVZ_NORMAL:
        warn.append(f"GVZ = {gvz:.1f} (ความผันผวนสูง) — ลดขนาดไม้ และระวังมาร์จินถูกปรับขึ้น")
    if float(d1["adx"].iloc[-1]) < 20:
        warn.append("ADX(1D) < 20 → ตลาดไม่มีแนวโน้มชัด สัญญาณตามเทรนด์เชื่อถือได้น้อยลง")
    if bool(d1["squeeze"].iloc[-1]):
        warn.append("BB Squeeze (1D) — ความผันผวนบีบตัว มักตามด้วยการเลือกทางแรง รอ breakout ยืนยัน")
    if cot.get("available") and (cot["pct_3y"] >= 85 or cot["pct_3y"] <= 15):
        warn.append("COT: " + cot["read"])
    if agree < 0.5:
        warn.append("ปัจจัยแต่ละส่วนขัดแย้งกัน (ความเห็นตรงกันต่ำ) — ความมั่นใจต่ำ")
    if not offset:
        warn.append("ยังไม่ได้ใส่ราคา GOLD จาก XM — ระดับราคาอิง COMEX GC=F ซึ่งต่างจากราคาสปอตของ XM ได้หลายสิบดอลลาร์")

    return {"ok": True, "price": price, "atr_d": atr_d, "atr_plan": atr_plan, "intraday": use_4h, "frames": fr, "tech": tech, "macro": mac, "cot": cot,
            "scenarios": scen, "sentiment": sent, "news_score": nsc, "parts": parts, "total": float(total),
            "agree": agree, "bias": bias, "levels": lv, "plans": plans, "warnings": warn,
            "xm": xm, "offset": offset, "ref_price": ref_px,
            "gemini": N.gemini_outlook(gemini_key, {"price": round(price, 1), "total_score": round(total),
                                                    "parts": {k: round(v, 2) for k, v in parts.items()},
                                                    "macro": [(c["name"], c["chg"]) for c in mac["components"]],
                                                    "gvz": gvz}, scen, data["news"]) if gemini_key else None}


def make_plan(side: str, price: float, atr: float, lv: dict, stop_atr: float, equity: float,
              risk_pct: float, xm: dict) -> dict:
    sgn = 1 if side == "LONG" else -1
    near = lv["supports"] if side == "LONG" else lv["resistances"]
    far = lv["resistances"] if side == "LONG" else lv["supports"]
    # จุดเข้า: รอย่อ/เด้งไปที่แนวที่แข็งแรงที่สุดในระยะ 1.5 ATR ถ้าไม่มี ใช้ราคา ∓ 0.5 ATR
    cand = [z for z in near if abs(price - z["price"]) <= 1.5 * atr]
    if cand:
        z = max(cand, key=lambda r: (r["strength"], -abs(price - r["price"])))
        level, why = z["price"], z["src"]
    else:
        level, why = price - sgn * 0.5 * atr, "ราคา ∓ 0.5 ATR (ไม่มีแนวใกล้)"
    entry_hi = level + 0.25 * atr if side == "LONG" else level
    entry_lo = level if side == "LONG" else level - 0.25 * atr
    entry = (entry_hi + entry_lo) / 2
    sl = level - sgn * stop_atr * atr
    risk_pt = abs(entry - sl)
    tgts = [z["price"] for z in far if sgn * (z["price"] - entry) > 0.5 * risk_pt]
    tp1 = tgts[0] if tgts else entry + sgn * 1.5 * risk_pt
    if abs(tp1 - entry) < C.MIN_RR * risk_pt:
        tp1 = entry + sgn * C.MIN_RR * risk_pt
    tp2 = next((t for t in tgts if sgn * (t - tp1) > 0.5 * risk_pt), entry + sgn * 3 * risk_pt)
    rr1, rr2 = abs(tp1 - entry) / risk_pt, abs(tp2 - entry) / risk_pt

    # ── ขนาด lot สำหรับ XM ── (ระยะเสี่ยงรวม spread)
    oz = float(xm["oz_per_lot"])
    risk_money = equity * risk_pct / 100
    loss_per_lot = (risk_pt + float(xm["spread_usd"])) * oz
    lots = round_lot(risk_money / loss_per_lot, float(xm["lot_step"]))
    if lots < float(xm["min_lot"]):
        lots = 0.0
    margin = lots * oz * entry / max(float(xm["leverage"]), 1)
    return {"side": side, "entry_lo": float(entry_lo), "entry_hi": float(entry_hi), "entry": float(entry),
            "entry_why": why, "sl": float(sl), "tp1": float(tp1), "tp2": float(tp2), "rr1": float(rr1),
            "rr2": float(rr2), "risk_pts": float(risk_pt), "risk_money": risk_money, "lots": lots,
            "loss_per_lot": loss_per_lot, "actual_risk": lots * loss_per_lot, "margin": margin,
            "min_lot_risk": float(xm["min_lot"]) * loss_per_lot,
            "invalidation": f"ปิดแท่ง 4H {'ต่ำกว่า' if side == 'LONG' else 'สูงกว่า'} {sl:,.2f}"}
