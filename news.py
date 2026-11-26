"""วิเคราะห์ข่าว: ปฏิทินเศรษฐกิจ → Scenario ล่วงหน้า + Sentiment พาดหัวข่าว + (ไม่บังคับ) Gemini

หลักคิด: แอป "ทายตัวเลขข่าว" ไม่ได้ แต่บอกได้ว่า
  1) ตลาดคาด (forecast) ร้อนขึ้นหรือเย็นลงจากครั้งก่อน
  2) ถ้าออกมาสูง/ต่ำกว่าคาด ทองมีแนวโน้มไปทางไหน (ผ่านช่องทาง Fed → real yield/ดอลลาร์)
  3) ความผันผวนที่ควรเผื่อ (จาก GVZ)"""
from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import config as C

log = logging.getLogger(__name__)

# (คำค้น, หมวด, ขั้ว) ขั้ว = ทิศทองถ้าตัวเลขออกมา "สูงกว่าคาด" (+1 ทองขึ้น, −1 ทองลง)
EVENT_RULES = [
    (r"\bcpi\b|consumer price", "เงินเฟ้อ", -1),
    (r"\bpce\b", "เงินเฟ้อ", -1),
    (r"\bppi\b|producer price", "เงินเฟ้อ", -1),
    (r"average hourly earnings|wage", "เงินเฟ้อ", -1),
    (r"non-?farm|nfp|adp", "ตลาดแรงงาน", -1),
    (r"unemployment rate", "ตลาดแรงงาน", +1),
    (r"jobless claims|unemployment claims", "ตลาดแรงงาน", +1),
    (r"jolts|job openings", "ตลาดแรงงาน", -1),
    (r"federal funds|fomc|fed chair|powell|interest rate", "Fed", -1),
    (r"\bgdp\b", "การเติบโต", -1),
    (r"retail sales", "การเติบโต", -1),
    (r"ism|pmi", "การเติบโต", -1),
    (r"consumer confidence|consumer sentiment", "การเติบโต", -1),
]

BULL_WORDS = ["rate cut", "dovish", "safe haven", "safe-haven", "geopolitic", "tension", "war", "weaker dollar",
              "dollar falls", "dollar slips", "central bank buying", "central bank gold", "etf inflow", "recession",
              "yields fall", "yields drop", "record high", "rally", "lifts gold", "gold rises", "gold climbs",
              "uncertainty", "tariff", "sanction", "debt ceiling", "stagflation"]
BEAR_WORDS = ["rate hike", "hawkish", "stronger dollar", "dollar rises", "dollar strengthens", "dollar gains",
              "yields rise", "yields climb", "outflow", "profit-taking", "profit taking", "risk-on", "ceasefire",
              "gold falls", "gold slips", "gold drops", "slides", "tumbles", "higher for longer", "sell-off"]


def num(s: str | None) -> float | None:
    """'0.3%' → 0.3 · '120K' → 120000 · '-1.2B' → -1.2e9"""
    if not s:
        return None
    m = re.match(r"^\s*([<>]?)(-?\d+(?:\.\d+)?)\s*([KMBT%]?)", str(s).replace(",", ""))
    if not m:
        return None
    v = float(m.group(2))
    return v * {"K": 1e3, "M": 1e6, "B": 1e9, "T": 1e12}.get(m.group(3), 1)


def classify(title: str) -> tuple[str, int] | None:
    t = title.lower()
    if "speaks" in t or "member" in t:
        return ("Fed (คำพูด)", -1) if ("fomc" in t or "fed" in t or "powell" in t) else None
    for pat, cat, pol in EVENT_RULES:
        if re.search(pat, t):
            return cat, pol
    return None


def event_scenarios(cal: list[dict], gold_trend: float, gvz: float | None, price: float,
                    now: datetime | None = None) -> list[dict]:
    """สร้าง Scenario ของข่าวที่ยังไม่ออก (Medium/High)"""
    now = now or datetime.now(timezone.utc)
    out = []
    for e in cal:
        if e.get("impact") not in ("High", "Medium"):
            continue
        try:
            t = pd.Timestamp(e["date"]).tz_convert("UTC").to_pydatetime()
        except Exception:  # noqa: BLE001
            continue
        if t < now:
            continue
        cl = classify(e.get("title", ""))
        if not cl:
            continue
        cat, pol = cl
        f, p = num(e.get("forecast")), num(e.get("previous"))
        if f is not None and p is not None and f != p:
            hotter = f > p
            pre = pol if hotter else -pol
            exp_txt = f"ตลาดคาด {'สูงขึ้น' if hotter else 'ต่ำลง'} จากครั้งก่อน ({e.get('previous')} → {e.get('forecast')})"
        else:
            pre = 0
            exp_txt = "ตลาดคาดใกล้เคียงครั้งก่อน" if f is not None else "ไม่มีตัวเลขคาดการณ์ (ดูถ้อยแถลง/น้ำเสียง)"
        hi_dir = "ขึ้น" if pol > 0 else "ลง"
        lo_dir = "ลง" if pol > 0 else "ขึ้น"
        if cat.startswith("Fed"):
            hi_txt, lo_txt = "ถ้อยแถลงเข้มงวด (hawkish) → ทองมีแนวโน้มลง", "ถ้อยแถลงผ่อนคลาย (dovish) → ทองมีแนวโน้มขึ้น"
        else:
            hi_txt = f"ออกมาสูงกว่าคาด → ทองมีแนวโน้ม{hi_dir}"
            lo_txt = f"ออกมาต่ำกว่าคาด → ทองมีแนวโน้ม{lo_dir}"
        inline = "ตรงคาด → มักกลับไปตามแนวโน้มเดิม (" + ("ขาขึ้น" if gold_trend > 0.15 else
                                                    "ขาลง" if gold_trend < -0.15 else "ไซด์เวย์") + ")"
        move = None
        if gvz and price:
            # GVZ = vol ต่อปีที่ตลาดคาด → กรอบเคลื่อนไหว 1 วันโดยประมาณ (1 SD)
            move = price * gvz / 100 / np.sqrt(252)
        out.append({"time_utc": t, "hours_left": (t - now).total_seconds() / 3600, "event": e.get("title"),
                    "impact": e.get("impact"), "category": cat, "forecast": e.get("forecast"),
                    "previous": e.get("previous"), "expectation": exp_txt, "if_high": hi_txt, "if_low": lo_txt,
                    "if_inline": inline, "pre_bias": pre, "exp_move_1sd": move})
    return sorted(out, key=lambda r: r["time_utc"])


def headline_sentiment(items: list[dict]) -> dict:
    rows = []
    for it in items:
        t = it["title"].lower()
        b = sum(w in t for w in BULL_WORDS)
        s = sum(w in t for w in BEAR_WORDS)
        sc = 0 if b == s else (1 if b > s else -1)
        rows.append({**it, "score": sc})
    n = len(rows)
    score = float(np.mean([r["score"] for r in rows])) if n else 0.0
    return {"score": score, "n": n, "bull": sum(r["score"] > 0 for r in rows),
            "bear": sum(r["score"] < 0 for r in rows), "rows": rows}


def news_score(scen: list[dict], sent: dict) -> float:
    """รวม bias ก่อนข่าว (ข่าวใน 7 วัน, High หนักกว่า) + sentiment พาดหัว → −1..+1"""
    pre = [r["pre_bias"] * (1.0 if r["impact"] == "High" else 0.5) for r in scen if r["hours_left"] <= 168]
    p = float(np.clip(np.mean(pre), -1, 1)) if pre else 0.0
    return float(np.clip(0.5 * p + 0.5 * sent["score"], -1, 1))


# ════════════ Gemini (ไม่บังคับ) ════════════
def gemini_outlook(api_key: str, snapshot: dict, scen: list[dict], heads: list[dict]) -> dict | None:
    """ให้ Gemini สรุปทิศทางข่าวเป็น JSON · คืน None ถ้าไม่มี key/ใช้ไม่ได้"""
    if not api_key:
        return None
    try:
        from google import genai
        from google.genai import types as gt
    except ImportError:
        return {"error": "ยังไม่ได้ติดตั้ง google-genai"}
    client = genai.Client(api_key=api_key)
    models = []
    try:
        for m in client.models.list():
            n = (getattr(m, "name", "") or "").split("/")[-1]
            if "gemini" in n and not any(x in n for x in ("image", "tts", "live", "embedding", "audio", "gemma")):
                models.append(n)
    except Exception as e:  # noqa: BLE001
        log.warning("list models: %s", e)
    rank = lambda n: (("preview" in n) or ("exp" in n), next((i for i, k in enumerate(C.GEMINI_PREF) if k in n), 9))
    chain = list(dict.fromkeys(sorted(models, key=rank)[:3] + C.GEMINI_FALLBACK))
    prompt = (
        "คุณคือนักวิเคราะห์ตลาดทองคำ (Gold futures) ตอบเป็นภาษาไทย\n"
        "ใช้เฉพาะข้อมูลที่ให้ด้านล่าง ถ้าข้อมูลไม่พอให้ตอบว่า 'ไม่รู้' ห้ามเดาตัวเลข\n"
        "ประเมินทิศทางข่าวช่วง 1–7 วันข้างหน้าต่อราคาทอง\n\n"
        f"สภาพตลาด: {json.dumps(snapshot, ensure_ascii=False, default=str)}\n\n"
        f"ปฏิทินข่าว: {json.dumps([{k: r[k] for k in ('event', 'time_utc', 'forecast', 'previous', 'expectation')} for r in scen[:10]], ensure_ascii=False, default=str)}\n\n"
        f"พาดหัวข่าวล่าสุด: {json.dumps([h['title'] for h in heads[:25]], ensure_ascii=False)}\n\n"
        'ตอบ JSON เท่านั้น: {"bias":"ขึ้น|ลง|กลาง","prob_up":0-100,"confidence":"ต่ำ|กลาง|สูง",'
        '"drivers":["..."],"event_views":[{"event":"...","view":"..."}],"risks":["..."],"summary":"..."}'
    )
    errs = []
    for m in chain:
        try:
            r = client.models.generate_content(model=m, contents=prompt, config=gt.GenerateContentConfig(
                temperature=0.2, response_mime_type="application/json"))
            txt = re.sub(r"```(?:json)?", "", r.text or "")
            js = json.loads(txt[txt.find("{"): txt.rfind("}") + 1])
            js["model"] = m
            return js
        except Exception as e:  # noqa: BLE001
            errs.append(f"{m}: {str(e)[:100]}")
    return {"error": " | ".join(errs[:3])}
