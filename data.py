"""ดึงข้อมูลทั้งหมด: ราคา (Yahoo), FRED, CFTC COT, ปฏิทินเศรษฐกิจ, ข่าว RSS
ทุกฟังก์ชันคืนค่าได้แม้แหล่งข้อมูลล่ม (คืน DataFrame ว่าง/ลิสต์ว่าง) — แอปจะแจ้งว่าแหล่งไหนใช้ไม่ได้"""
from __future__ import annotations

import io
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from urllib.parse import quote_plus

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

import config as C

log = logging.getLogger(__name__)


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update(C.UA)
    r = Retry(total=3, backoff_factor=1.0, status_forcelist=(429, 500, 502, 503, 504), allowed_methods=("GET",))
    s.mount("https://", HTTPAdapter(max_retries=r))
    return s


SESSION = _session()


FAST = requests.Session()  # ไม่ retry · ใช้กับแหล่งที่อาจบล็อก cloud (เช่น FRED)
FAST.headers.update(C.UA)


def _get(url: str, fast: bool = False, **kw) -> requests.Response:
    r = (FAST if fast else SESSION).get(url, timeout=8 if fast else C.HTTP_TIMEOUT, **kw)
    r.raise_for_status()
    return r


# ════════════ ราคา ════════════
def parse_yahoo(js: dict) -> pd.DataFrame:
    res = (js.get("chart") or {}).get("result") or []
    if not res:
        return pd.DataFrame()
    r = res[0]
    ts = r.get("timestamp") or []
    q = ((r.get("indicators") or {}).get("quote") or [{}])[0]
    if not ts or not q:
        return pd.DataFrame()
    df = pd.DataFrame({k: q.get(k) for k in ("open", "high", "low", "close", "volume")},
                      index=pd.to_datetime(ts, unit="s", utc=True))
    df = df.apply(pd.to_numeric, errors="coerce").dropna(subset=["close"])
    df["volume"] = df["volume"].fillna(0.0)
    for k in ("open", "high", "low"):
        df[k] = df[k].fillna(df["close"])
    return df[~df.index.duplicated(keep="last")].sort_index()


def yahoo(symbol: str, interval: str = "1d", rng: str = "2y") -> pd.DataFrame:
    """interval: 1h/1d/1wk · range: 1h ใช้ได้สูงสุด ~730d"""
    for host in ("query1", "query2"):
        try:
            url = f"https://{host}.finance.yahoo.com/v8/finance/chart/{quote_plus(symbol)}"
            df = parse_yahoo(_get(url, params={"interval": interval, "range": rng}).json())
            if not df.empty:
                return df
        except Exception as e:  # noqa: BLE001
            log.warning("yahoo %s %s: %s", host, symbol, e)
    try:  # สำรอง: yfinance (ถ้าติดตั้งไว้)
        import yfinance as yf
        df = yf.Ticker(symbol).history(interval=interval, period=rng, auto_adjust=False)
        if not df.empty:
            df = df.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
            df.index = pd.to_datetime(df.index, utc=True)
            return df
    except Exception as e:  # noqa: BLE001
        log.warning("yfinance %s: %s", symbol, e)
    return pd.DataFrame()


def resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.resample(rule, label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
    return out.dropna(subset=["close"])


# ════════════ FRED ════════════
def parse_fred_csv(text: str) -> pd.Series:
    df = pd.read_csv(io.StringIO(text))
    if df.shape[1] < 2:
        return pd.Series(dtype=float)
    s = pd.Series(pd.to_numeric(df.iloc[:, 1], errors="coerce").values,
                  index=pd.to_datetime(df.iloc[:, 0], utc=True)).dropna()
    return s


def fred(series_id: str) -> pd.Series:
    try:
        return parse_fred_csv(_get("https://fred.stlouisfed.org/graph/fredgraph.csv", fast=True,
                                   params={"id": series_id}).text)
    except Exception as e:  # noqa: BLE001
        log.warning("fred %s: %s", series_id, e)
        return pd.Series(dtype=float)


# ════════════ CFTC COT ════════════
def parse_cot(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    need = ["report_date_as_yyyy_mm_dd", "open_interest_all", "m_money_positions_long_all", "m_money_positions_short_all"]
    if any(c not in df for c in need):
        return pd.DataFrame()
    out = pd.DataFrame({
        "oi": pd.to_numeric(df["open_interest_all"], errors="coerce"),
        "mm_long": pd.to_numeric(df["m_money_positions_long_all"], errors="coerce"),
        "mm_short": pd.to_numeric(df["m_money_positions_short_all"], errors="coerce"),
    })
    out.index = pd.to_datetime(df["report_date_as_yyyy_mm_dd"], utc=True)
    out = out.dropna().sort_index()
    out["mm_net"] = out["mm_long"] - out["mm_short"]
    out["mm_net_pct_oi"] = out["mm_net"] / out["oi"] * 100
    return out[~out.index.duplicated(keep="last")]


def cot_gold(weeks: int = 260) -> pd.DataFrame:
    try:
        rows = _get(C.COT_URL, params={"cftc_contract_market_code": C.COT_GOLD_CODE,
                                       "$order": "report_date_as_yyyy_mm_dd DESC", "$limit": weeks}).json()
        return parse_cot(rows)
    except Exception as e:  # noqa: BLE001
        log.warning("cot: %s", e)
        return pd.DataFrame()


# ════════════ ปฏิทินเศรษฐกิจ ════════════
def calendar() -> list[dict]:
    try:
        rows = _get(C.CALENDAR_URL).json()
        return [r for r in rows if isinstance(r, dict) and r.get("country") == "USD"]
    except Exception as e:  # noqa: BLE001
        log.warning("calendar: %s", e)
        return []


# ════════════ ข่าว RSS (Google News) ════════════
def headlines(per_query: int = 8) -> list[dict]:
    try:
        import feedparser
    except ImportError:
        return []
    out, seen = [], set()
    for q in C.NEWS_QUERIES:
        url = f"https://news.google.com/rss/search?q={quote_plus(q + ' when:3d')}&hl=en-US&gl=US&ceid=US:en"
        try:
            feed = feedparser.parse(_get(url).content)
        except Exception as e:  # noqa: BLE001
            log.warning("rss %s: %s", q, e)
            continue
        for e in feed.entries[:per_query]:
            t = (e.get("title") or "").strip()
            if not t or t.lower() in seen:
                continue
            seen.add(t.lower())
            pub = None
            if e.get("published_parsed"):
                pub = datetime(*e.published_parsed[:6], tzinfo=timezone.utc)
            out.append({"title": t, "link": e.get("link", ""), "published": pub, "query": q})
    out.sort(key=lambda x: x["published"] or datetime.min.replace(tzinfo=timezone.utc), reverse=True)
    return out


# ════════════ รวมทั้งหมด ════════════
def load_all() -> dict:
    """ดึงทุกแหล่งพร้อมกัน (ขนาน) · errors = รายชื่อแหล่งที่ดึงไม่ได้"""
    errors: list[str] = []
    jobs = {
        "g1h": (yahoo, (C.YF["gold"], "1h", "730d")),
        "g1d": (yahoo, (C.YF["gold"], "1d", "10y")),
        **{f"y_{k}": (yahoo, (C.YF[k], "1d", "5y")) for k in ("silver", "dxy", "gvz", "us10y")},
        **{f"f_{k}": (fred, (v,)) for k, v in C.FRED.items()},
        "cot": (cot_gold, ()), "cal": (calendar, ()), "news": (headlines, ()),
    }
    with ThreadPoolExecutor(max_workers=len(jobs)) as ex:
        fut = {k: ex.submit(fn, *args) for k, (fn, args) in jobs.items()}
        res = {}
        for k, f in fut.items():
            try:
                res[k] = f.result(timeout=90)
            except Exception as e:  # noqa: BLE001
                log.warning("job %s: %s", k, e)
                res[k] = None
    empty = pd.DataFrame()
    g1h = res["g1h"] if res["g1h"] is not None else empty
    g1d = res["g1d"] if res["g1d"] is not None else empty
    if g1d.empty:
        errors.append("ราคาทอง (Yahoo GC=F) — วิเคราะห์ไม่ได้")
    frames = {"1h": g1h, "4h": resample(g1h, "4h"), "1d": g1d, "1w": resample(g1d, "W-FRI")}
    side = {}
    for k in ("silver", "dxy", "gvz", "us10y"):
        df = res[f"y_{k}"]
        side[k] = df["close"] if df is not None and not df.empty else pd.Series(dtype=float)
        if side[k].empty:
            errors.append(f"Yahoo {C.YF[k]}")
    fr = {k: (res[f"f_{k}"] if res[f"f_{k}"] is not None else pd.Series(dtype=float)) for k in C.FRED}
    if any(v.empty for v in fr.values()):
        errors.append("FRED (ใช้ดอกเบี้ย 10 ปีจาก Yahoo แทน)")
    if side["gvz"].empty and not fr["gvz_fred"].empty:
        side["gvz"] = fr["gvz_fred"]
    cot = res["cot"] if res["cot"] is not None else empty
    if cot.empty:
        errors.append("CFTC COT")
    cal = res["cal"] or []
    if not cal:
        errors.append("ปฏิทินเศรษฐกิจ")
    news = res["news"] or []
    if not news:
        errors.append("ข่าว RSS")
    return {"frames": frames, "side": side, "fred": fr, "cot": cot, "calendar": cal, "news": news,
            "errors": errors, "fetched": datetime.now(timezone.utc)}


def synthetic(n: int = 1500, seed: int = 7) -> dict:
    """ข้อมูลจำลองสำหรับทดสอบ/สาธิต (ใช้ตอนแหล่งข้อมูลจริงใช้ไม่ได้)"""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(end=pd.Timestamp.now(tz="UTC").floor("h"), periods=n * 6, freq="h")
    ret = rng.normal(0.00005, 0.0025, len(idx))
    close = 3800 * np.exp(np.cumsum(ret))
    hi = close * (1 + np.abs(rng.normal(0, 0.0015, len(idx))))
    lo = close * (1 - np.abs(rng.normal(0, 0.0015, len(idx))))
    op = np.r_[close[0], close[:-1]]
    g1h = pd.DataFrame({"open": op, "high": np.maximum.reduce([hi, op, close]),
                        "low": np.minimum.reduce([lo, op, close]), "close": close,
                        "volume": rng.integers(500, 5000, len(idx)).astype(float)}, index=idx)
    g1d = resample(g1h, "1D")
    d = g1d.index
    side = {
        "silver": pd.Series(g1d["close"].values / (80 + np.cumsum(rng.normal(0, 0.3, len(d)))), d),
        "dxy": pd.Series(100 + np.cumsum(rng.normal(0, 0.3, len(d))), d),
        "gvz": pd.Series(np.clip(18 + np.cumsum(rng.normal(0, 0.5, len(d))), 9, 60), d),
        "us10y": pd.Series(4.3 + np.cumsum(rng.normal(0, 0.03, len(d))), d),
    }
    fr = {"real10": pd.Series(2.0 + np.cumsum(rng.normal(0, 0.03, len(d))), d),
          "be5": pd.Series(2.3 + np.cumsum(rng.normal(0, 0.01, len(d))), d),
          "gvz_fred": side["gvz"]}
    w = pd.date_range(end=d[-1], periods=200, freq="W-TUE")
    longs = 150000 + np.cumsum(rng.normal(0, 4000, len(w)))
    shorts = np.clip(40000 + np.cumsum(rng.normal(0, 2000, len(w))), 5000, None)
    cot = pd.DataFrame({"oi": 450000.0, "mm_long": longs, "mm_short": shorts}, index=w)
    cot["mm_net"] = cot["mm_long"] - cot["mm_short"]
    cot["mm_net_pct_oi"] = cot["mm_net"] / cot["oi"] * 100
    now = pd.Timestamp.now(tz="UTC")
    cal = [
        {"title": "CPI m/m", "country": "USD", "date": (now + pd.Timedelta(hours=20)).isoformat(),
         "impact": "High", "forecast": "0.3%", "previous": "0.2%"},
        {"title": "Non-Farm Employment Change", "country": "USD", "date": (now + pd.Timedelta(days=2)).isoformat(),
         "impact": "High", "forecast": "120K", "previous": "150K"},
        {"title": "Unemployment Rate", "country": "USD", "date": (now + pd.Timedelta(days=2)).isoformat(),
         "impact": "High", "forecast": "4.3%", "previous": "4.2%"},
        {"title": "FOMC Statement", "country": "USD", "date": (now + pd.Timedelta(days=4)).isoformat(),
         "impact": "High", "forecast": "", "previous": ""},
    ]
    news = [{"title": t, "link": "", "published": now, "query": "demo"} for t in (
        "Gold slips as dollar strengthens after hawkish Fed comments",
        "Central bank gold buying stays strong, supports prices",
        "Treasury yields rise ahead of CPI data",
        "Safe-haven demand lifts gold amid geopolitical tension",
    )]
    return {"frames": {"1h": g1h, "4h": resample(g1h, "4h"), "1d": g1d, "1w": resample(g1d, "W-FRI")},
            "side": side, "fred": fr, "cot": cot, "calendar": cal, "news": news,
            "errors": ["โหมดสาธิต: ใช้ข้อมูลจำลอง ไม่ใช่ราคาจริง"], "fetched": now.to_pydatetime()}
