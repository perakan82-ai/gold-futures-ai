import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
import backtest, data, engine, indicators, news


def test_parsers():
    js = {"chart": {"result": [{"timestamp": [1700000000, 1700086400],
          "indicators": {"quote": [{"open": [1, 2], "high": [2, 3], "low": [0.5, 1.5], "close": [1.5, None], "volume": [10, 20]}]}}]}}
    assert len(data.parse_yahoo(js)) == 1
    assert data.parse_fred_csv("observation_date,DFII10\n2026-11-09,2.66\n2026-11-10,.\n").iloc[-1] == 2.66
    cot = data.parse_cot([{"report_date_as_yyyy_mm_dd": "2026-11-17T00:00:00.000", "open_interest_all": "407834",
                           "m_money_positions_long_all": "121994", "m_money_positions_short_all": "19052"}])
    assert cot["mm_net"].iloc[0] == 121994 - 19052


def test_news_rules():
    assert news.num("0.3%") == 0.3 and news.num("120K") == 120000 and news.num("-1.2B") == -1.2e9
    assert news.classify("CPI m/m") == ("เงินเฟ้อ", -1)
    assert news.classify("Unemployment Rate")[1] == 1
    assert news.classify("Non-Farm Employment Change")[1] == -1


def test_engine_and_levels_consistent():
    d = data.synthetic()
    r = engine.analyze(d, 10000)
    assert r["ok"] and -100 <= r["total"] <= 100
    L, S = r["plans"]["LONG"], r["plans"]["SHORT"]
    assert L["sl"] < L["entry"] < L["tp1"] <= L["tp2"] and L["rr1"] >= 1.5 - 1e-9
    assert S["sl"] > S["entry"] > S["tp1"] >= S["tp2"] and S["rr1"] >= 1.5 - 1e-9
    assert all(z["price"] < r["price"] for z in r["levels"]["supports"])
    assert all(z["price"] > r["price"] for z in r["levels"]["resistances"])


def test_xm_lots_and_offset():
    d = data.synthetic()
    ref = float(d["frames"]["1h"]["close"].iloc[-1])
    r = engine.analyze(d, 1000, risk_pct=1.0, xm={"oz_per_lot": 100, "leverage": 500, "spread_usd": 0.3},
                       xm_price=ref + 25.0)
    assert abs(r["price"] - (ref + 25.0)) < 1e-6 and abs(r["offset"] - 25.0) < 1e-6
    p = r["plans"]["LONG"]
    assert abs(p["loss_per_lot"] - (p["risk_pts"] + 0.3) * 100) < 1e-6
    assert p["actual_risk"] <= p["risk_money"] + 1e-9          # ไม่เสี่ยงเกินที่ตั้ง
    assert abs(round(p["lots"] / 0.01) - p["lots"] / 0.01) < 1e-6  # lot เป็นขั้น 0.01
    assert abs(p["margin"] - p["lots"] * 100 * p["entry"] / 500) < 1e-6


def test_no_lookahead_in_votes():
    d = indicators.add_all(data.synthetic()["frames"]["1d"])
    full = indicators.tech_votes(d)["score"]
    cut = indicators.tech_votes(indicators.add_all(data.synthetic()["frames"]["1d"].iloc[:-30]))["score"]
    assert np.allclose(full.iloc[:len(cut)].values[-50:], cut.values[-50:])


def test_backtest_random_walk_has_no_edge():
    rng = np.random.default_rng(3); n = 2500
    c = 1500 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    d = pd.DataFrame({"open": np.r_[c[0], c[:-1]], "close": c}, index=pd.date_range("2015-01-01", periods=n, freq="B", tz="UTC"))
    d["high"], d["low"], d["volume"] = d.max(axis=1) * 1.004, d.min(axis=1) * 0.996, 1.0
    b = backtest.run(d)
    assert b["ok"] and b["oos"]["trades"] > 0


def test_probabilities_sane():
    import prob
    d = indicators.add_all(data.synthetic()["frames"]["4h"])
    st_ = prob.Stats(d, indicators.tech_votes(d)["score"], 12)
    assert st_.ok
    dr = st_.direction(0.0)
    assert 0 < dr["p_up"] < 1 and abs(dr["p_up"] + dr["p_down"] - 1) < 1e-9
    ps = [st_.touch(0.0, x, up=True) for x in (0.5, 1, 2, 3)]
    assert all(a >= b for a, b in zip(ps, ps[1:]))            # ยิ่งไกลยิ่งแตะยาก
    ft = st_.tp_before_sl(0.0, 1.5, 1.0, True)
    assert abs(ft["p_tp"] + ft["p_sl"] + ft["p_open"] - 1) < 0.05


def test_lot_calculator():
    k = engine.calc_position(5000, 1, 4000, 3970, 4045, 4075, {"spread_usd": 0.3})
    assert k["ok"] and k["side"] == "Buy" and k["lots"] == 0.01        # 50$ / 30.3$ ต่อ 0.01 lot
    assert abs(k["loss"] - 30.3) < 1e-9 and abs(k["gain_tp1"] - 44.7) < 1e-9
    k = engine.calc_position(20000, 1, 4000, 3970, 4045, 4075, {"spread_usd": 0.3})
    assert k["lots"] == 0.06 and k["loss"] <= 200 and k["split"]["lots_tp1"] + k["split"]["lots_tp2"] == 0.06
    assert engine.calc_position(5000, 1, 4000, 4030, 4050, None)["ok"] is False   # Sell แต่ TP สูงกว่า
    assert engine.calc_position(300, 1, 4000, 3970, 4045, None)["lots"] == 0
