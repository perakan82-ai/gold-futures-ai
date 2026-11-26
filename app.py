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
st.set_page_config(page_title="Gold AI — XM GOLD", page_icon="🥇", layout="wide")


@st.cache_data(ttl=900, show_spinner="กำลังดึงข้อมูลตลาด…")
def load(demo: bool) -> dict:
    return D.synthetic() if demo else D.load_all()


@st.cache_data(ttl=3600, show_spinner="กำลัง backtest…")
def run_bt(d1: pd.DataFrame) -> dict:
    return BT.run(d1)


def fmt(x, n=1):
    return "—" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:,.{n}f}"


# ═════════════ Sidebar (ใส่แค่ 3 อย่างก็ใช้ได้) ═════════════
with st.sidebar:
    st.header("① ใส่ข้อมูล 3 อย่าง")
    xm_price = st.number_input("ราคา GOLD ใน XM ตอนนี้", min_value=0.0, value=0.0, step=0.01, format="%.2f",
                               help="ดูจาก MT4/MT5 แล้วพิมพ์ใส่ ทุกราคาในแอปจะตรงกับ XM (ใส่ 0 = ใช้ราคา COMEX)")
    equity = st.number_input("เงินในพอร์ต XM (USD)", min_value=10.0, value=1000.0, step=100.0)
    style = st.radio("เทรดแบบไหน", ["ในวัน (Intraday)", "ถือหลายวัน (Swing)"], index=0,
                     help="ในวัน = SL แคบกว่า เหมาะกับพอร์ตเล็ก · ถือหลายวัน = SL กว้าง มีค่า swap ข้ามคืน")
    if st.button("🔄 อัปเดตข้อมูลตลาด", width="stretch"):
        load.clear()
    with st.expander("ตั้งค่าเพิ่มเติม (ไม่ต้องแก้ก็ได้)"):
        risk_pct = st.slider("ความเสี่ยงต่อไม้ (% ของพอร์ต)", 0.25, 3.0, C.RISK_PCT, 0.25)
        stop_atr = st.slider("ระยะ SL (× ATR จากแนว)", 0.5, 3.0, C.STOP_ATR, 0.25)
        st.caption("สเปก XM — ตรวจใน MT5: คลิกขวา GOLD → Specification")
        oz = st.number_input("oz ต่อ 1 lot", min_value=1.0, value=C.XM["oz_per_lot"], step=1.0)
        lev = st.number_input("Leverage (1:x)", min_value=1, max_value=1000, value=C.XM["leverage"], step=50)
        spread = st.number_input("Spread (USD)", min_value=0.0, value=C.XM["spread_usd"], step=0.05)
        min_lot = st.number_input("Lot ต่ำสุด", min_value=0.01, value=C.XM["min_lot"], step=0.01)
        use_ai = st.toggle("ให้ Gemini สรุปข่าว", value=False, help="ต้องใส่ GEMINI_API_KEY ใน Streamlit secrets")
        demo = st.toggle("โหมดสาธิต (ข้อมูลจำลอง)", value=os.environ.get("GOLD_DEMO") == "1")
    xm = {"oz_per_lot": oz, "leverage": lev, "spread_usd": spread, "min_lot": min_lot, "lot_step": C.XM["lot_step"]}
    st.caption(f"v{C.VERSION} · ข้อมูลตลาดอัปเดตทุก 15 นาที")

key = st.secrets.get("GEMINI_API_KEY", "") if use_ai else ""
data = load(demo)
if data["errors"]:
    st.caption("⚠️ แหล่งข้อมูลที่ใช้ไม่ได้รอบนี้: " + " · ".join(data["errors"]))
if data["frames"]["1d"].empty:
    st.error("ดึงราคาทองไม่ได้ — เปิด 'โหมดสาธิต' เพื่อดูหน้าตาแอป หรือลองใหม่ภายหลัง")
    st.stop()

R = E.analyze(data, equity, risk_pct, stop_atr, key, xm=xm, xm_price=xm_price, intraday=style.startswith("ในวัน"))
if not R["ok"]:
    st.error(R["error"])
    st.stop()

# ═════════════ Header ═════════════
st.title("🥇 Gold AI — XM GOLD")
fetched = pd.Timestamp(data["fetched"]).tz_convert(TH).strftime("%d/%m/%Y %H:%M")
off_txt = (f"เลื่อนให้ตรง XM แล้ว ({R['offset']:+.2f} USD จาก COMEX {R['ref_price']:,.2f})" if R["offset"]
           else "ยังไม่ได้ปรับให้ตรงราคา XM")
st.caption(f"วิเคราะห์จากข้อมูล COMEX GC=F · {off_txt} · อัปเดต {fetched} น. (เวลาไทย) · "
           "ทองใน XM เป็น CFD (มี swap ข้ามคืน) · เครื่องมือช่วยวิเคราะห์ ไม่ใช่คำแนะนำการลงทุน")

S = E.summary(R)
p2 = lambda x: f"{x:,.2f}"

# ═════════════ ② สรุปตรงๆ ═════════════
with st.container(border=True):
    st.markdown(f"## {S['headline']}")
    dirp = S.get("direction")
    if dirp:
        u, dn_ = dirp["p_up"] * 100, dirp["p_down"] * 100
        q = st.columns([1, 1, 2])
        q[0].metric("โอกาสขึ้น (Long)", f"{u:.0f}%")
        q[1].metric("โอกาสลง (Short)", f"{dn_:.0f}%")
        q[2].progress(int(round(u)), text=f"ขึ้น {u:.0f}% · ลง {dn_:.0f}% ใน {R['probs']['horizon']}")
        st.caption(f"จากสถิติย้อนหลัง: แท่ง {R['probs']['tf']} ที่สัญญาณกราฟคล้ายตอนนี้ {dirp['n']:,} ครั้ง "
                   f"(ค่าเฉลี่ยทุกช่วง ขึ้น {dirp['base_up'] * 100:.0f}%) · นับเฉพาะสัญญาณกราฟ ไม่รวมข่าว/ดอลลาร์")
    st.caption(f"GOLD {p2(R['price'])} · คะแนนรวม {R['total']:+.0f} (เกิน +20 = ซื้อ, ต่ำกว่า −20 = ขาย)")
    for n in S["news_soon"]:
        t = pd.Timestamp(n["time_utc"]).tz_convert(TH).strftime("%H:%M น. %d/%m")
        st.error(f"⏰ ข่าว **{n['event']}** ออก {t} (อีก {n['hours_left']:.0f} ชม.) — อย่าเปิดไม้ก่อนข่าว "
                 f"ราคาอาจแกว่ง ±{n['exp_move_1sd'] or 0:,.0f} USD และกระโดดข้าม SL ได้")
    if S["order"] and S["action"] in ("LONG", "SHORT"):
        o = S["order"]
        st.markdown("**ตั้งออเดอร์ใน MT5 แบบนี้**")
        k = st.columns(5)
        k[0].metric("ออเดอร์", o["type"])
        k[1].metric("ราคาเข้า", p2(o["entry"]))
        k[2].metric("Stop Loss", p2(o["sl"]))
        k[3].metric("Take Profit", p2(o["tp1"]))
        k[4].metric("Lot", f"{o['lots']:.2f}")
        pr = o.get("prob")
        if pr:
            z = st.columns(4)
            z[0].metric("โอกาสได้ราคาเข้า", f"{pr['p_fill'] * 100:.0f}%")
            z[1].metric("โอกาสถึง TP ก่อน SL", f"{pr['p_tp1'] * 100:.0f}%")
            z[2].metric("โอกาสโดน SL", f"{pr['p_sl'] * 100:.0f}%")
            z[3].metric("โอกาสถึง TP2", f"{pr['p_tp2'] * 100:.0f}%")
            st.caption(f"ภายใน {R['probs']['horizon']} · ที่เหลือ {pr['p_open'] * 100:.0f}% = ยังไม่ถึงทั้ง TP และ SL")
        if S.get("stat_warn"):
            st.warning(S["stat_warn"])
        st.markdown(f"- ถ้าโดน SL จะเสียประมาณ **{o['risk']:,.2f} USD** · ถ้าถึง TP ได้ประมาณ **{o['risk'] * o['rr1']:,.2f} USD**\n"
                    f"- ถึง TP แล้วไปต่อได้: TP2 {p2(o['tp2'])} (ปิดครึ่งที่ TP แล้วเลื่อน SL มาที่ราคาเข้า)\n"
                    f"- ยกเลิกออเดอร์ถ้า: {o['cancel']}")
    elif S["action"] == "SKIP" and S["order"]:
        o = S["order"]
        st.markdown(f"ไม้นี้ต้องวาง SL ห่าง {abs(o['entry'] - o['sl']):,.2f} USD — แม้ 0.01 lot ก็เสี่ยงเกินที่ตั้งไว้ "
                    f"({risk_pct:.2f}% ของ {equity:,.0f} USD) **รอจังหวะที่ SL แคบกว่า**" + ("" if R["intraday"] else " หรือเปลี่ยนเป็นเทรดในวัน"))
    else:
        for w in S.get("watch", []):
            st.markdown(f"- {w}")
    st.markdown("**ทำไม**")
    for r in S["reasons"]:
        st.markdown(f"- {r}")

# ═════════════ ③ เครื่องคิด lot ═════════════
with st.container(border=True):
    st.markdown("### 🧮 คำนวณ lot จากทุน")
    cside = R["bias"] if R["bias"] in ("LONG", "SHORT") else "LONG"
    cp = R["plans"][cside]
    kk = f"{cside}_{cp['entry']:.2f}"
    st.caption("ค่าเริ่มต้นมาจากแผนด้านบน" + ("" if R["bias"] != "WAIT" else " (ฝั่งซื้อ เพราะตอนนี้ยังรอ)") +
               " — แก้ตัวเลขได้ตามที่ตั้งจริงใน MT5")
    a1, a2 = st.columns(2)
    cap = a1.number_input("ทุนในพอร์ต (USD)", min_value=10.0, value=float(equity), step=100.0, key=f"cap{equity}")
    rp = a2.number_input("ยอมเสียต่อไม้ (% ของทุน)", min_value=0.1, max_value=10.0, value=float(risk_pct), step=0.25, key=f"rp{risk_pct}")
    b1, b2, b3, b4 = st.columns(4)
    en = b1.number_input("ราคาเข้า", value=round(cp["entry"], 2), step=0.1, format="%.2f", key="en" + kk)
    sl_ = b2.number_input("Stop Loss", value=round(cp["sl"], 2), step=0.1, format="%.2f", key="sl" + kk)
    t1 = b3.number_input("Take Profit 1", value=round(cp["tp1"], 2), step=0.1, format="%.2f", key="t1" + kk)
    t2 = b4.number_input("Take Profit 2 (ไม่ใช้ใส่ 0)", value=round(cp["tp2"], 2), step=0.1, format="%.2f", key="t2" + kk)
    K = E.calc_position(cap, rp, en, sl_, t1, t2 or None, xm)
    if not K["ok"]:
        st.error(K["error"])
    elif K["lots"] == 0:
        st.error(f"เปิดไม่ได้: 0.01 lot ก็เสีย {K['min_lot_risk']:,.2f} USD ถ้าโดน SL "
                 f"(= {K['min_lot_risk'] / cap * 100:.1f}% ของทุน) เกินที่ยอมเสีย {K['risk_target']:,.2f} USD "
                 f"→ ต้องมีทุนอย่างน้อย {K['min_lot_risk'] / (rp / 100):,.0f} USD หรือวาง SL ให้แคบลง")
    else:
        m = st.columns(5)
        m[0].metric("เปิด", f"{K['lots']:.2f} lot")
        m[1].metric("ถ้าโดน SL เสีย", f"−{K['loss']:,.2f} $", f"{K['loss_pct']:.2f}% ของทุน", delta_color="off")
        m[2].metric("ถ้าถึง TP1 ได้", f"+{K['gain_tp1']:,.2f} $", f"R:R 1:{K['rr1']:.2f}", delta_color="off")
        m[3].metric("Margin ที่ใช้", f"{K['margin']:,.2f} $", f"{K['margin_pct']:.1f}% ของทุน", delta_color="off")
        m[4].metric("ระยะ SL / TP1", f"{K['sl_dist']:,.2f} / {K['tp1_dist']:,.2f}")
        otype = E.order_type("LONG" if K["long"] else "SHORT", en, R["price"], R["atr_plan"])
        st.success(f"**{otype} {K['lots']:.2f} lot @ {en:,.2f} · SL {sl_:,.2f} · TP {t1:,.2f}**")
        sp = K["split"]
        if sp:
            st.markdown(f"- แบ่งปิด: **{sp['lots_tp1']:.2f} lot ที่ TP1 {t1:,.2f}** และ **{sp['lots_tp2']:.2f} lot ที่ TP2 {K['tp2']:,.2f}** "
                        f"→ ถ้าถึงทั้งคู่ได้ ~{sp['gain_total']:,.2f} $ · พอถึง TP1 ให้เลื่อน SL ของส่วนที่เหลือมาที่ {en:,.2f} (ไม่ขาดทุนแล้ว)")
        elif K["tp2"]:
            st.markdown("- lot น้อยเกินกว่าจะแบ่งปิด 2 จุด → ปิดทั้งหมดที่ TP1")
        st.markdown(f"- ทุก 0.01 lot: ราคาขยับ 1 USD = ได้/เสีย {K['usd_per_1_move_per_001']:.2f} USD · "
                    f"ไม้นี้ 0.01 lot เสีย {K['per_001']:,.2f} USD ถ้าโดน SL (รวม spread)")
        if K["capped_by_margin"]:
            st.warning("ลด lot ลงเพื่อไม่ให้ margin เกินครึ่งของทุน")
        if K["loss_pct"] > 2:
            st.warning("เสี่ยงเกิน 2% ของทุนต่อไม้ — สูงกว่าที่แนะนำทั่วไป")
        with st.expander("เทียบขนาด lot อื่น"):
            st.dataframe(pd.DataFrame([{"lot": f"{t['lot']:.2f}", "เสียถ้าโดน SL ($)": f"{t['loss']:,.2f}",
                                        "% ของทุน": f"{t['loss_pct']:.2f}%", "ได้ถ้าถึง TP1 ($)": f"{t['gain_tp1']:,.2f}",
                                        "Margin ($)": f"{t['margin']:,.2f}"} for t in K["table"]]), hide_index=True)

other = [w for w in R["warnings"] if not w.startswith("มีข่าวแรง") and not any(w.endswith(r[-25:]) for r in S["reasons"])]
if other:
    with st.expander(f"⚠️ ข้อควรระวังอื่น ({len(other)})"):
        for w in other:
            st.markdown(f"- {w}")

st.divider()
st.caption("รายละเอียดด้านล่างสำหรับคนที่อยากดูลึก — ใช้แค่กล่องด้านบนก็พอ")
chart_tf = "1d"
tabs = st.tabs(["📋 แผนทั้ง 2 ฝั่ง", "📈 กราฟ", "🌍 ดอลลาร์/ดอกเบี้ย", "📰 ข่าว", "🧪 ทดสอบย้อนหลัง", "📘 วิธีใช้"])

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
            rows = [("โซนเข้า", f"{fmt(p['entry_lo'], 2)} – {fmt(p['entry_hi'], 2)}"), ("เหตุผลของโซน", p["entry_why"]),
                    ("Stop loss", fmt(p["sl"], 2)), ("TP1", f"{fmt(p['tp1'], 2)}  (R:R {p['rr1']:.1f})"),
                    ("TP2", f"{fmt(p['tp2'], 2)}  (R:R {p['rr2']:.1f})"), ("ยกเลิกแผนเมื่อ", p["invalidation"]),
                    ("ระยะเสี่ยง", f"{fmt(p['risk_pts'], 2)} USD/oz (+ spread {R['xm']['spread_usd']:.2f})"),
                    ("ขาดทุนต่อ 1.00 lot ถ้าโดน SL", f"{fmt(p['loss_per_lot'], 0)} USD"),
                    ("Lot ที่แนะนำ", f"{p['lots']:.2f} lot"),
                    ("ความเสี่ยงจริง", f"{fmt(p['actual_risk'], 2)} USD (เป้า {fmt(p['risk_money'], 2)})"),
                    ("Margin ที่ใช้ (1:" + str(int(R['xm']['leverage'])) + ")", f"{fmt(p['margin'], 2)} USD")]
            if p.get("prob"):
                pp = p["prob"]
                rows += [("โอกาสได้ราคาเข้า", f"{pp['p_fill'] * 100:.0f}%"),
                         ("โอกาส TP1 ก่อน SL / โดน SL", f"{pp['p_tp1'] * 100:.0f}% / {pp['p_sl'] * 100:.0f}%"),
                         ("โอกาสถึง TP2", f"{pp['p_tp2'] * 100:.0f}%"),
                         ("ผลคาดหวังต่อไม้", f"{pp['ev_R']:+.2f} R")]
            st.table(pd.DataFrame(rows, columns=["รายการ", "ค่า"]).set_index("รายการ"))
            if p["lots"] == 0:
                st.error(f"เงิน/ความเสี่ยงที่ตั้งไว้ไม่พอแม้แต่ lot ต่ำสุด ({R['xm']['min_lot']:.2f} lot ต้องเสี่ยง "
                         f"{fmt(p['min_lot_risk'], 2)} USD) — ข้ามไม้นี้ หรือรอจุดเข้าที่ SL แคบกว่า "
                         "(อย่าเพิ่มความเสี่ยงเพื่อให้เข้าได้)")
            elif p["margin"] > 0.5 * equity:
                st.warning("Margin ที่ใช้เกินครึ่งของพอร์ต — เสี่ยงโดน margin call ถ้าราคาแกว่งแรง")

    st.subheader("แนวรับ / แนวต้านสำคัญ")
    a, b = st.columns(2)
    lvl = lambda xs: pd.DataFrame([{"ราคา": fmt(z["price"], 2),
                                    "โอกาสแตะ": f"{z['p_touch'] * 100:.0f}%" if "p_touch" in z else "—",
                                    "ความแข็งแรง": "★" * z["strength"], "ที่มา": z["src"]} for z in xs])
    st.caption(f"โอกาสแตะ = % ที่ราคาเคยไปถึงระยะเท่านี้ภายใน {R['probs']['horizon']} (สถิติย้อนหลัง)")
    a.markdown("**แนวต้าน (ใกล้ → ไกล)**"); a.dataframe(lvl(R["levels"]["resistances"]), hide_index=True)
    b.markdown("**แนวรับ (ใกล้ → ไกล)**"); b.dataframe(lvl(R["levels"]["supports"]), hide_index=True)

# ═════════════ 2) กราฟ ═════════════
with tabs[1]:
    chart_tf = st.radio("Timeframe", ["1h", "4h", "1d", "1w"], index=1, horizontal=True)
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
    g = m.get("gvz")
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
2. ดู **% โอกาส** — ถ้าโอกาสถึง TP ต่ำกว่าโอกาสโดน SL มาก หรือผลคาดหวังติดลบ ให้ข้าม
3. เปิดแท็บ **ข่าว** — ถ้ามีข่าวแรงใน 24 ชม. ให้รอหลังข่าว
3. ใส่ **ราคา GOLD จาก XM** ที่แถบซ้าย แล้วใช้ **โซนเข้า / SL / TP / lot** จากแท็บแผนเทรด · ตั้ง SL ใน MT4/MT5 ทุกครั้ง
4. ดูแท็บ **Backtest** — ถ้า Out-of-sample ไม่ผ่าน อย่าเชื่อสัญญาณเทคนิคเพียงลำพัง
5. ทดลองแบบ paper trade ก่อนใช้เงินจริง

**ข้อจำกัดที่ควรรู้**
- แอปวิเคราะห์จาก COMEX GC=F แล้วเลื่อนให้ตรงราคา XM ที่คุณใส่ · ส่วนต่าง futures กับสปอตเปลี่ยนช้าๆ ควรอัปเดตราคา XM ทุกครั้งที่ใช้
- GOLD ใน XM เป็น CFD: มีค่า swap เมื่อถือข้ามคืน และ spread กว้างขึ้นช่วงข่าว/ตลาดบาง · สเปก lot ให้ตรวจใน MT5 → Specification
- Yahoo / ปฏิทินข่าว / RSS เป็นแหล่งฟรี อาจล่าช้าหรือดึงไม่ได้บางครั้ง
- CFD ใช้เลเวอเรจสูง ขาดทุนได้เร็วมาก · ราคากระโดดข้าม SL ได้ช่วงข่าวหรือตอนเปิดตลาด
- แอปนี้เป็นเครื่องมือช่วยวิเคราะห์ ไม่ใช่คำแนะนำการลงทุน
""")
