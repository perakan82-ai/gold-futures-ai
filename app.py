"""Gold Futures AI — แดชบอร์ดวิเคราะห์ทองคำล่วงหน้า (Streamlit)
รัน: streamlit run app.py"""
from __future__ import annotations

import os
from datetime import timedelta, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import backtest as BT
import config as C
import data as D
import engine as E

TH = timezone(timedelta(hours=7))
st.set_page_config(page_title="Gold Futures AI", page_icon="🥇", layout="wide")


@st.cache_data(ttl=900, show_spinner="กำลังดึงข้อมูลตลาด…")
def load(demo: bool) -> dict:
    return D.synthetic() if demo else D.load_all()


@st.cache_data(ttl=3600, show_spinner="กำลัง backtest…")
def run_bt(d1: pd.DataFrame) -> dict:
    return BT.run(d1)


def fmt(x, n=1):
    return "—" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:,.{n}f}"


# ═════════════ Sidebar ═════════════
with st.sidebar:
    st.header("⚙️ ตั้งค่า")
    contract = st.selectbox("สัญญาที่เทรด", list(C.CONTRACTS))
    spec = C.CONTRACTS[contract]
    ccy = "THB" if spec["kind"] == "gf10" else "USD"
    equity = st.number_input(f"เงินในพอร์ต ({ccy})", min_value=100.0,
                             value=300000.0 if ccy == "THB" else 10000.0, step=1000.0)
    risk_pct = st.slider("ความเสี่ยงต่อไม้ (% ของพอร์ต)", 0.25, 3.0, C.RISK_PCT, 0.25)
    stop_atr = st.slider("ระยะ SL (× ATR รายวัน จากแนว)", 0.5, 3.0, C.STOP_ATR, 0.25)
    chart_tf = st.radio("Timeframe กราฟ", ["1h", "4h", "1d", "1w"], index=2, horizontal=True)
    use_ai = st.toggle("ให้ Gemini สรุปทิศทางข่าว", value=False,
                       help="ต้องใส่ GEMINI_API_KEY ใน Streamlit secrets")
    demo = st.toggle("โหมดสาธิต (ข้อมูลจำลอง)", value=os.environ.get("GOLD_DEMO") == "1")
    if st.button("🔄 ดึงข้อมูลใหม่"):
        load.clear()
    st.caption(f"v{C.VERSION} · ข้อมูลเก็บไว้ 15 นาที")

key = st.secrets.get("GEMINI_API_KEY", "") if use_ai else ""
data = load(demo)
if data["errors"]:
    st.warning("แหล่งข้อมูลที่ใช้ไม่ได้รอบนี้: " + " · ".join(data["errors"]))
if data["frames"]["1d"].empty:
    st.error("ดึงราคาทองไม่ได้ — เปิด 'โหมดสาธิต' เพื่อดูหน้าตาแอป หรือลองใหม่ภายหลัง")
    st.stop()

R = E.analyze(data, contract, equity, risk_pct, stop_atr, key)
if not R["ok"]:
    st.error(R["error"])
    st.stop()

# ═════════════ Header ═════════════
st.title("🥇 Gold Futures AI")
fetched = pd.Timestamp(data["fetched"]).tz_convert(TH).strftime("%d/%m/%Y %H:%M")
st.caption(f"ราคาอ้างอิง COMEX GC=F (สัญญาต่อเนื่อง) · อัปเดต {fetched} น. (เวลาไทย) · "
           "เครื่องมือช่วยวิเคราะห์ ไม่ใช่คำแนะนำการลงทุน")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("ราคา (USD/oz)", fmt(R["price"]))
if R["usdthb"] and spec["kind"] == "gf10":
    c1.caption(f"≈ GF10 {fmt(E.to_gf10(R['price'], R['usdthb']), 0)} บาท/บาททองคำ")
bias_txt = {"LONG": "🟢 LONG", "SHORT": "🔴 SHORT", "WAIT": "⚪ รอ"}[R["bias"]]
c2.metric("มุมมองรวม", bias_txt, f"{R['total']:+.0f} / 100")
c3.metric("ความเห็นตรงกัน", f"{R['agree'] * 100:.0f}%")
c4.metric("ATR รายวัน", fmt(R["atr_d"]))
g = R["macro"].get("gvz")
c5.metric("GVZ (ความผันผวน)", fmt(g["value"]) if g else "—", g["regime"] if g else None, delta_color="off")

for w in R["warnings"]:
    st.warning(w)

tabs = st.tabs(["🎯 แผนเทรด", "📈 กราฟ & Indicator", "🌍 มหภาค & COT", "📰 ข่าว & Scenario", "🧪 Backtest", "📘 วิธีใช้"])

# ═════════════ 1) แผนเทรด ═════════════
with tabs[0]:
    st.subheader("คะแนนแต่ละส่วน (−1 ลง … +1 ขึ้น)")
    names = {"tech_1d": "เทคนิค 1D", "tech_4h": "เทคนิค 4H", "tech_1h": "เทคนิค 1H", "macro": "มหภาค",
             "cot": "COT", "news": "ข่าว"}
    pdf = pd.DataFrame([{"ส่วน": names[k], "คะแนน": round(v, 2), "น้ำหนัก": C.WEIGHTS[k]} for k, v in R["parts"].items()])
    fig = go.Figure(go.Bar(x=pdf["คะแนน"], y=pdf["ส่วน"], orientation="h",
                           marker_color=["#2e9e5b" if v > 0 else "#d64545" for v in pdf["คะแนน"]]))
    fig.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10), xaxis_range=[-1, 1])
    st.plotly_chart(fig, width="stretch")
    st.caption(f"Weekly (ไม่รวมในคะแนน ใช้ดูภาพใหญ่): {R['tech']['1w']['score']:+.2f}")

    order = ["LONG", "SHORT"] if R["bias"] != "SHORT" else ["SHORT", "LONG"]
    cols = st.columns(2)
    for col, side in zip(cols, order):
        p = R["plans"][side]
        main = side == R["bias"]
        with col:
            st.markdown(f"### {'⭐ ' if main else ''}แผน {side}" + ("" if main else " (สำรอง / ถ้าสถานการณ์เปลี่ยน)"))
            if R["bias"] == "WAIT":
                st.info("คะแนนรวมยังไม่ชัด — แผนนี้ใช้เมื่อราคามาถึงโซนและมีสัญญาณยืนยัน")
            rows = [("โซนเข้า", f"{fmt(p['entry_lo'])} – {fmt(p['entry_hi'])}"), ("เหตุผลของโซน", p["entry_why"]),
                    ("Stop loss", fmt(p["sl"])), ("TP1", f"{fmt(p['tp1'])}  (R:R {p['rr1']:.1f})"),
                    ("TP2", f"{fmt(p['tp2'])}  (R:R {p['rr2']:.1f})"), ("ยกเลิกแผนเมื่อ", p["invalidation"]),
                    ("ระยะเสี่ยง", f"{fmt(p['risk_pts'])} USD/oz")]
            if R["usdthb"] and spec["kind"] == "gf10":
                cv = lambda x: fmt(E.to_gf10(x, R["usdthb"]), 0)
                rows += [("โซนเข้า (GF10)", f"{cv(p['entry_lo'])} – {cv(p['entry_hi'])}"),
                         ("SL / TP1 / TP2 (GF10)", f"{cv(p['sl'])} / {cv(p['tp1'])} / {cv(p['tp2'])}")]
            if p["contracts"] is not None:
                rows += [("เสียต่อ 1 สัญญาถ้าโดน SL", f"{fmt(p['loss_1c'], 0)} {R['ccy']}"),
                         ("ขนาดไม้ที่แนะนำ", f"{p['contracts']} สัญญา (เสี่ยง {fmt(p['risk_money'], 0)} {R['ccy']})")]
            st.table(pd.DataFrame(rows, columns=["รายการ", "ค่า"]).set_index("รายการ"))
            if p["contracts"] == 0:
                st.error("พอร์ต/ความเสี่ยงที่ตั้งไว้ไม่พอสำหรับ 1 สัญญานี้ — เลือกสัญญาที่เล็กกว่า หรือข้ามไม้นี้ "
                         "(อย่าเพิ่มความเสี่ยงเพื่อให้ได้ 1 สัญญา)")

    st.subheader("แนวรับ / แนวต้านสำคัญ")
    a, b = st.columns(2)
    lvl = lambda xs: pd.DataFrame([{"ราคา": fmt(z["price"]), "ความแข็งแรง": "★" * z["strength"], "ที่มา": z["src"]}
                                   for z in xs])
    a.markdown("**แนวต้าน (ใกล้ → ไกล)**"); a.dataframe(lvl(R["levels"]["resistances"]), hide_index=True)
    b.markdown("**แนวรับ (ใกล้ → ไกล)**"); b.dataframe(lvl(R["levels"]["supports"]), hide_index=True)

# ═════════════ 2) กราฟ ═════════════
with tabs[1]:
    d = R["frames"][chart_tf].tail({"1h": 300, "4h": 300, "1d": 250, "1w": 200}[chart_tf])
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.6, 0.2, 0.2], vertical_spacing=0.03)
    fig.add_trace(go.Candlestick(x=d.index, open=d.open, high=d.high, low=d.low, close=d.close, name="ราคา"), 1, 1)
    for n, colr in ((21, "#f2a900"), (50, "#3b82f6"), (200, "#a855f7")):
        fig.add_trace(go.Scatter(x=d.index, y=d[f"ema{n}"], name=f"EMA{n}", line=dict(width=1, color=colr)), 1, 1)
    fig.add_trace(go.Scatter(x=d.index, y=d["st_line"], name="Supertrend", mode="markers",
                             marker=dict(size=3, color=np.where(d["st_dir"] > 0, "#2e9e5b", "#d64545"))), 1, 1)
    fig.add_trace(go.Scatter(x=d.index, y=d["cloud_top"], name="Ichimoku cloud", line=dict(width=0),
                             showlegend=False), 1, 1)
    fig.add_trace(go.Scatter(x=d.index, y=d["cloud_bot"], name="Ichimoku cloud", line=dict(width=0),
                             fill="tonexty", fillcolor="rgba(150,150,150,0.15)"), 1, 1)
    for z in R["levels"]["supports"][:4]:
        fig.add_hline(y=z["price"], line=dict(color="#2e9e5b", dash="dot", width=1), row=1, col=1)
    for z in R["levels"]["resistances"][:4]:
        fig.add_hline(y=z["price"], line=dict(color="#d64545", dash="dot", width=1), row=1, col=1)
    fig.add_trace(go.Scatter(x=d.index, y=d["rsi"], name="RSI", line=dict(color="#f2a900")), 2, 1)
    fig.add_trace(go.Scatter(x=d.index, y=d["stochrsi_k"], name="StochRSI %K", line=dict(color="#94a3b8", width=1)), 2, 1)
    for y in (30, 70):
        fig.add_hline(y=y, line=dict(color="gray", dash="dash", width=1), row=2, col=1)
    fig.add_trace(go.Bar(x=d.index, y=d["macd_h"], name="MACD hist",
                         marker_color=np.where(d["macd_h"] > 0, "#2e9e5b", "#d64545")), 3, 1)
    fig.update_layout(height=760, xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=10, b=10),
                      legend=dict(orientation="h"))
    st.plotly_chart(fig, width="stretch")

    st.subheader("สรุป Indicator รายตัว")
    lab = {1: "🟢 ขึ้น", -1: "🔴 ลง", 0: "⚪ กลาง"}
    vt = pd.DataFrame({tf.upper(): {k: lab.get(int(np.sign(v)), "⚪") for k, v in R["tech"][tf]["votes"].items()}
                       for tf in ("1h", "4h", "1d", "1w")})
    st.dataframe(vt, width="stretch")
    last = R["frames"]["1d"].iloc[-1]
    st.caption(f"1D: RSI {last.rsi:.0f} · ADX {last.adx:.0f} · StochRSI {last.stochrsi_k:.0f} · "
               f"BB z {last.bb_z:+.2f} · Squeeze {'ใช่' if last.squeeze else 'ไม่'}")
    pv = R["levels"]["pivots"]
    if pv:
        st.markdown("**Pivot รายวัน:** " + " · ".join(f"{k} {fmt(v)}" for k, v in pv.items()))
    fb = R["levels"]["fib"]
    if fb:
        st.markdown(f"**Fibonacci** (swing {fmt(fb['swing_low'])} → {fmt(fb['swing_high'])}, ขา{('ขึ้น' if fb['direction'] == 'up' else 'ลง')}): "
                    + " · ".join(f"{k} {fmt(v)}" for k, v in fb["levels"].items()))

# ═════════════ 3) มหภาค & COT ═════════════
with tabs[2]:
    m = R["macro"]
    st.subheader(f"ปัจจัยมหภาค · คะแนน {m['score']:+.2f}")
    if m["components"]:
        st.dataframe(pd.DataFrame([{"ปัจจัย": c["name"], "ล่าสุด": fmt(c["value"], 2), "เปลี่ยนแปลง": c["chg"],
                                    "อ่านว่า": c["read"], "คะแนน": round(c["score"], 2),
                                    "Corr กับทอง 120 วัน": fmt(c["corr120"], 2), "น้ำหนักที่ใช้": round(c["weight"], 2)}
                                   for c in m["components"]]), hide_index=True, width="stretch")
        st.caption("ถ้า correlation ไม่เป็นลบตามทฤษฎี (ความสัมพันธ์อ่อนลง) แอปจะลดน้ำหนักปัจจัยนั้นเอง")
    if m["context"]:
        st.markdown("**บริบทเพิ่มเติม (ไม่นำไปคิดคะแนน)**")
        st.dataframe(pd.DataFrame([{"ตัวชี้วัด": c["name"], "ล่าสุด": fmt(c["value"], 2), "รายละเอียด": c["chg"],
                                    "อ่านว่า": c["read"]} for c in m["context"]]), hide_index=True, width="stretch")
    if g:
        st.markdown(f"**GVZ {g['value']:.1f}** ({g['regime']}, สูงกว่า {g['pct_1y']:.0f}% ของ 1 ปี) → "
                    f"กรอบเคลื่อนไหวคาดการณ์ ~±{g['implied_daily_move_pct']:.2f}% ต่อวัน "
                    f"(≈ ±{R['price'] * g['implied_daily_move_pct'] / 100:,.0f} USD)")
    cot = R["cot"]
    st.subheader("COT — Managed Money (กองทุน) ในทอง COMEX")
    if cot.get("available"):
        k1, k2, k3 = st.columns(3)
        k1.metric("Net long", f"{cot['net']:,.0f}", f"{cot['wk_chg']:+,.0f} สัปดาห์นี้")
        k2.metric("Percentile 3 ปี", f"{cot['pct_3y']:.0f}%")
        k3.metric("% ของ Open interest", f"{cot['pct_oi']:.1f}%")
        st.info(cot["read"] + f" · ข้อมูล ณ {cot['date']:%d/%m/%Y} (CFTC ออกทุกศุกร์ ใช้ข้อมูลวันอังคาร)")
        s = cot["series"].tail(156)
        st.plotly_chart(go.Figure(go.Scatter(x=s.index, y=s["mm_net"], fill="tozeroy", name="MM net"))
                        .update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10)), width="stretch")
    else:
        st.write("ไม่มีข้อมูล COT รอบนี้")

# ═════════════ 4) ข่าว ═════════════
with tabs[3]:
    st.subheader(f"Scenario ข่าวที่กำลังจะออก · คะแนนข่าว {R['news_score']:+.2f}")
    st.caption("แอปไม่ทายตัวเลขจริง — บอกว่าตลาดคาดอะไร และถ้าออกมาแต่ละแบบ ทองมักไปทางไหน (ผ่านช่องทาง Fed → ดอกเบี้ยแท้จริง/ดอลลาร์)")
    if R["scenarios"]:
        for s in R["scenarios"]:
            t = pd.Timestamp(s["time_utc"]).tz_convert(TH).strftime("%a %d/%m %H:%M")
            icon = "🔴" if s["impact"] == "High" else "🟠"
            with st.expander(f"{icon} {t} น. · {s['event']} · อีก {s['hours_left']:.0f} ชม.", expanded=s["hours_left"] < 48):
                st.markdown(f"- **หมวด:** {s['category']} · คาด {s['forecast'] or '—'} / ครั้งก่อน {s['previous'] or '—'}\n"
                            f"- **ตลาดคาด:** {s['expectation']}\n- {s['if_high']}\n- {s['if_low']}\n- {s['if_inline']}"
                            + (f"\n- **กรอบแกว่ง 1 วันที่ควรเผื่อ (จาก GVZ, 1 SD):** ±{s['exp_move_1sd']:,.0f} USD" if s["exp_move_1sd"] else ""))
    else:
        st.write("ไม่มีข่าว USD ระดับ Medium/High ที่ยังไม่ออกในสัปดาห์นี้ (หรือดึงปฏิทินไม่ได้)")

    if R["gemini"]:
        st.subheader("🤖 มุมมองจาก Gemini")
        gm = R["gemini"]
        if gm.get("error"):
            st.error("Gemini ใช้ไม่ได้: " + gm["error"])
        else:
            st.markdown(f"**ทิศทาง:** {gm.get('bias')} · โอกาสขึ้น ~{gm.get('prob_up')}% · ความมั่นใจ {gm.get('confidence')} "
                        f"· โมเดล {gm.get('model')}")
            st.write(gm.get("summary", ""))
            for x in gm.get("drivers", []):
                st.markdown(f"- {x}")
            for ev in gm.get("event_views", []):
                st.markdown(f"- **{ev.get('event')}**: {ev.get('view')}")
            if gm.get("risks"):
                st.markdown("**ความเสี่ยง:** " + " · ".join(gm["risks"]))
            st.caption("ความเห็น AI อาจผิดได้ ใช้ประกอบเท่านั้น")

    sent = R["sentiment"]
    st.subheader(f"พาดหัวข่าว 3 วันล่าสุด · บวก {sent['bull']} / ลบ {sent['bear']} จาก {sent['n']}")
    for r in sent["rows"][:30]:
        ic = "🟢" if r["score"] > 0 else ("🔴" if r["score"] < 0 else "⚪")
        st.markdown(f"{ic} [{r['title']}]({r['link']})" if r["link"] else f"{ic} {r['title']}")
    st.caption("คะแนนพาดหัวใช้คำสำคัญ (keyword) — หยาบ ใช้ดูภาพรวมอารมณ์ข่าวเท่านั้น")

# ═════════════ 5) Backtest ═════════════
with tabs[4]:
    st.subheader("ทดสอบสัญญาณ technical รายวันย้อนหลัง (In-sample 70% / Out-of-sample 30%)")
    st.caption("SL 1.5 ATR / TP 3 ATR · เข้าที่ราคาเปิดแท่งถัดไป · หักต้นทุน 0.05 ATR ต่อไม้ · ถ้าแตะ SL และ TP ในแท่งเดียวนับว่าโดน SL")
    bt = run_bt(data["frames"]["1d"])
    if not bt["ok"]:
        st.write(bt["error"])
    else:
        st.markdown(f"**ผลช่วง In-sample (เลือก threshold):** ตัดข้อมูลที่ {bt['split']:%d/%m/%Y}")
        st.dataframe(bt["grid"].round(2), hide_index=True)
        o = bt["oos"]
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("ไม้ OOS", o["trades"]); k2.metric("Win %", fmt(o["win%"]))
        k3.metric("เฉลี่ย R/ไม้", fmt(o["avg_R"], 2)); k4.metric("Max DD (R)", fmt(o["max_dd_R"]))
        (st.success if "ผ่าน:" in bt["verdict"] and "ไม่ผ่าน" not in bt["verdict"] else st.error)(bt["verdict"])
        if not bt["oos_trades"].empty:
            eq = bt["oos_trades"].set_index("entry_time")["R"].cumsum()
            st.plotly_chart(go.Figure(go.Scatter(x=eq.index, y=eq, name="Equity (R)"))
                            .update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10)), width="stretch")

# ═════════════ 6) วิธีใช้ ═════════════
with tabs[5]:
    st.markdown("""
**ลำดับการใช้งานที่แนะนำ**
1. ดู **มุมมองรวม** และ **ความเห็นตรงกัน** — ถ้าต่ำกว่า 50% ให้ถือว่ายังไม่มีจังหวะ
2. เปิดแท็บ **ข่าว** — ถ้ามีข่าวแรงใน 24 ชม. ให้รอหลังข่าว
3. ใช้ **โซนเข้า / SL / TP** จากแท็บแผนเทรด · ตั้ง SL ทุกครั้ง · ขนาดไม้ตามที่แอปคำนวณ
4. ดูแท็บ **Backtest** — ถ้า Out-of-sample ไม่ผ่าน อย่าเชื่อสัญญาณเทคนิคเพียงลำพัง
5. ทดลองแบบ paper trade ก่อนใช้เงินจริง

**ข้อจำกัดที่ควรรู้**
- ราคาจาก GC=F เป็นสัญญาต่อเนื่อง มีช่องว่างตอนโรลสัญญา · ราคา TFEX จริงอาจต่างจากที่แปลงไว้เล็กน้อย (premium/ค่าเงิน)
- Yahoo / ปฏิทินข่าว / RSS เป็นแหล่งฟรี อาจล่าช้าหรือดึงไม่ได้บางครั้ง
- Futures ใช้เลเวอเรจ ขาดทุนได้เกินเงินที่วางไว้ · มาร์จินถูกปรับขึ้นได้ตอนตลาดผันผวน
- แอปนี้เป็นเครื่องมือช่วยวิเคราะห์ ไม่ใช่คำแนะนำการลงทุน
""")
