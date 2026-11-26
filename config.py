"""ค่าตั้งค่าทั้งหมดของ Gold Futures AI"""
from __future__ import annotations

VERSION = "1.3.0"

HTTP_TIMEOUT = 20
UA = {"User-Agent": "Mozilla/5.0 (GoldFuturesAI; personal research dashboard)"}

# ───────── แหล่งข้อมูลราคา (Yahoo Finance chart API — ฟรี ไม่ต้องใช้ key) ─────────
# GC=F = COMEX Gold ต่อเนื่อง (สัญญาใกล้สุด) · มีช่องว่างราคาตอนโรลสัญญาได้
YF = {
    "gold": "GC=F",
    "silver": "SI=F",
    "dxy": "DX-Y.NYB",
    "gvz": "^GVZ",
    "us10y": "^TNX",
}

# ───────── FRED (ฟรี ไม่ต้องใช้ key ผ่าน fredgraph.csv) ─────────
FRED = {
    "real10": "DFII10",   # 10Y TIPS real yield
    "be5": "T5YIE",       # 5Y breakeven inflation (ตลาดคาดเงินเฟ้อ)
    "gvz_fred": "GVZCLS",  # สำรองของ GVZ
}

# ───────── CFTC COT (Disaggregated Futures-only) ─────────
COT_URL = "https://publicreporting.cftc.gov/resource/72hh-3qpy.json"
COT_GOLD_CODE = "088691"  # GOLD - COMMODITY EXCHANGE INC.

# ───────── ปฏิทินเศรษฐกิจ (ForexFactory JSON ฟรี) ─────────
CALENDAR_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"

# ───────── ข่าว (RSS) ─────────
NEWS_QUERIES = [
    "gold price",
    "Federal Reserve rate",
    "US dollar treasury yields",
    "central bank gold buying",
]

# ───────── สัญญา: XM GOLD (CFD อิงราคาทองสปอต XAU/USD) ─────────
# หน้าเว็บ XM ไม่ได้แสดงสเปกสัญญา → ค่าด้านล่างเป็นค่ามาตรฐานของตลาด ปรับได้ในแอป
# ตรวจค่าจริงใน MT4/MT5: คลิกขวาที่ GOLD → Specification
XM = {
    "symbol": "GOLD",
    "oz_per_lot": 100.0,   # 1 lot = 100 oz (มาตรฐาน XAUUSD) · 1 USD/oz = 100 USD ต่อ 1 lot
    "min_lot": 0.01,
    "lot_step": 0.01,
    "leverage": 500,       # XM ให้ 1:1 ถึง 1000:1 · ใส่ตามบัญชีจริง
    "spread_usd": 0.30,    # spread โดยประมาณ (USD/oz) — ดูค่าจริงในแพลตฟอร์ม
}
# Margin XM = lots × contract size × ราคา ÷ min(leverage บัญชี, leverage ของสัญลักษณ์)

# ───────── ความเสี่ยง (ค่าเริ่มต้น ปรับได้ในแอป) ─────────
RISK_PCT = 1.0        # % ของพอร์ตต่อไม้ (ช่วงที่ใช้กันทั่วไป 0.5–2%)
STOP_ATR = 1.5        # SL = ระดับ ± 1.5 × ATR(1D)
MIN_RR = 1.5
EVENT_WARN_H = 24     # เตือนถ้ามีข่าวแรงภายใน 24 ชม.

# ───────── น้ำหนักคะแนน ─────────
WEIGHTS = {"tech_1d": 0.25, "tech_4h": 0.20, "tech_1h": 0.10, "macro": 0.25, "cot": 0.05, "news": 0.15}
BIAS_TH = 20  # คะแนนรวม > +20 = Long, < −20 = Short, ระหว่างนั้น = รอ

# ───────── GVZ (ความผันผวนคาดการณ์ 30 วัน) ─────────
GVZ_LOW, GVZ_NORMAL = 12.0, 20.0

# ───────── Gemini (ไม่บังคับ) ─────────
GEMINI_PREF = ("flash-lite", "flash", "pro")
GEMINI_FALLBACK = ["gemini-flash-lite-latest", "gemini-flash-latest", "gemini-2.5-flash-lite", "gemini-2.5-flash"]
