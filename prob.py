"""ความน่าจะเป็นจากสถิติย้อนหลัง (ไม่ใช่การเดา)

วิธีคิด: หาแท่งในอดีตที่ "คะแนนเทคนิค" อยู่ช่วงเดียวกับตอนนี้ แล้วนับว่าหลังจากนั้น H แท่ง
  - ราคาปิดสูงกว่า/ต่ำกว่าเดิมกี่ %            → โอกาส Long / Short
  - ราคาแตะระดับที่ห่าง X ATR กี่ %             → โอกาสถึงแต่ละจุด (แนวรับ/แนวต้าน/จุดเข้า)
  - แตะ TP ก่อน SL กี่ %                        → โอกาสชนะของแผน
ระยะทุกจุดวัดเป็น "จำนวน ATR" เพื่อเทียบข้ามช่วงราคาได้
ตัวอย่างที่ซ้อนทับกันถูกนับเป็นจำนวนอิสระ (n_eff = n ÷ H) แล้วดึงเข้าหาค่าเฉลี่ยรวมถ้าตัวอย่างน้อย"""
from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

BINS = (-1.01, -0.5, -0.2, 0.2, 0.5, 1.01)
PRIOR_N = 20  # ตัวอย่างอิสระสมมติสำหรับดึงเข้าหาค่าฐาน


def _windows(d: pd.DataFrame, H: int):
    n = len(d)
    if n <= H + 10:
        return None
    hi = sliding_window_view(d["high"].values[1:], H)[: n - H]   # แท่ง i+1 … i+H
    lo = sliding_window_view(d["low"].values[1:], H)[: n - H]
    c0 = d["close"].values[: n - H]
    atr = d["atr"].values[: n - H]
    cH = d["close"].values[H:]
    return hi, lo, c0, atr, cH


def _shrink(k: float, n: float, base: float, H: int) -> tuple[float, float]:
    n_eff = n / max(H, 1)
    k_eff = k / max(H, 1)
    return float((k_eff + PRIOR_N * base) / (n_eff + PRIOR_N)), n_eff


class Stats:
    """คำนวณครั้งเดียวต่อ timeframe แล้วถามได้หลายคำถาม"""

    def __init__(self, d: pd.DataFrame, score: pd.Series, H: int):
        self.H = H
        self.ok = False
        w = _windows(d, H)
        if w is None:
            return
        hi, lo, c0, atr, cH = w
        sc = score.reindex(d.index).values[: len(c0)]
        good = np.isfinite(atr) & (atr > 0) & np.isfinite(sc)
        if good.sum() < 100:
            return
        self.up = (hi.max(axis=1) - c0)[good] / atr[good]       # แกว่งขึ้นสูงสุด (ATR)
        self.dn = (c0 - lo.min(axis=1))[good] / atr[good]       # แกว่งลงสูงสุด (ATR)
        self.hi = (hi[good] - c0[good, None]) / atr[good, None]  # เส้นทางรายแท่ง (ATR)
        self.lo = (c0[good, None] - lo[good]) / atr[good, None]
        self.ret = (cH - c0)[good]
        self.score = sc[good]
        self.ok = True

    def subset(self, cur_score: float) -> np.ndarray:
        b = np.digitize([cur_score], BINS)[0]
        lo_b, hi_b = BINS[max(b - 1, 0)], BINS[min(b, len(BINS) - 1)]
        return (self.score >= lo_b) & (self.score < hi_b)

    def direction(self, cur_score: float) -> dict:
        base = float((self.ret > 0).mean())
        m = self.subset(cur_score)
        n = int(m.sum())
        k = float((self.ret[m] > 0).sum())
        p, n_eff = _shrink(k, n, base, self.H)
        return {"p_up": p, "p_down": 1 - p, "n": n, "n_eff": n_eff, "base_up": base}

    def touch(self, cur_score: float, dist_atr: float, up: bool) -> float:
        """โอกาสแตะระดับที่ห่าง dist_atr ATR ภายใน H แท่ง"""
        exc = self.up if up else self.dn
        base = float((exc >= dist_atr).mean())
        m = self.subset(cur_score)
        p, _ = _shrink(float((exc[m] >= dist_atr).sum()), int(m.sum()), base, self.H)
        return p

    def tp_before_sl(self, cur_score: float, tp_atr: float, sl_atr: float, long: bool) -> dict:
        """โอกาสแตะ TP ก่อน SL ภายใน H แท่ง (แตะทั้งคู่ในแท่งเดียว = นับว่าโดน SL)"""
        fav, adv = (self.hi, self.lo) if long else (self.lo, self.hi)
        H = fav.shape[1]
        t_tp = np.where((fav >= tp_atr).any(axis=1), (fav >= tp_atr).argmax(axis=1), H + 1)
        t_sl = np.where((adv >= sl_atr).any(axis=1), (adv >= sl_atr).argmax(axis=1), H + 1)
        win = (t_tp < t_sl) & (t_tp <= H)
        loss = (t_sl <= t_tp) & (t_sl <= H)
        base_w, base_l = float(win.mean()), float(loss.mean())
        m = self.subset(cur_score)
        pw, _ = _shrink(float(win[m].sum()), int(m.sum()), base_w, self.H)
        pl, _ = _shrink(float(loss[m].sum()), int(m.sum()), base_l, self.H)
        return {"p_tp": pw, "p_sl": pl, "p_open": max(0.0, 1 - pw - pl)}


def plan_probs(st: Stats, cur_score: float, price: float, atr: float, plan: dict, long: bool) -> dict:
    """ใส่ % ให้แผน: โอกาสได้ราคาเข้า, TP ก่อน SL (นับจากจุดเข้า), โอกาสแตะ TP1/TP2"""
    entry, sl, tp1, tp2 = plan["entry"], plan["sl"], plan["tp1"], plan["tp2"]
    d_entry = abs(price - entry) / atr
    entry_is_pullback = (entry < price) if long else (entry > price)
    p_fill = 1.0 if d_entry < 0.1 else st.touch(cur_score, d_entry, up=not entry_is_pullback)
    ft = st.tp_before_sl(cur_score, abs(tp1 - entry) / atr, abs(entry - sl) / atr, long)
    p_tp2 = st.tp_before_sl(cur_score, abs(tp2 - entry) / atr, abs(entry - sl) / atr, long)["p_tp"]
    ev = ft["p_tp"] * plan["rr1"] - ft["p_sl"]  # ผลคาดหวังต่อไม้ (หน่วย R) ไม่รวมไม้ที่ค้าง
    return {"p_fill": p_fill, "p_tp1": ft["p_tp"], "p_sl": ft["p_sl"], "p_open": ft["p_open"],
            "p_tp2": p_tp2, "ev_R": ev}
