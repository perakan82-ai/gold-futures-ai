# 🥇 Gold AI — XM GOLD

แดชบอร์ดวิเคราะห์ทองคำสำหรับเทรด **GOLD (XAU/USD CFD) ใน XM** บน Streamlit
วิเคราะห์จากข้อมูล COMEX GC=F แล้วเลื่อนระดับราคาให้ตรงกับราคา GOLD ใน XM ที่ผู้ใช้ใส่
รวมการวิเคราะห์ทางเทคนิคหลาย timeframe, ปัจจัยมหภาค, สถานะกองทุน (COT), Scenario ข่าว และแผนเทรดพร้อมขนาดไม้

> เครื่องมือช่วยวิเคราะห์ ไม่ใช่คำแนะนำการลงทุน · CFD ใช้เลเวอเรจสูง ขาดทุนได้เร็ว · มีค่า swap เมื่อถือข้ามคืน

## สิ่งที่แอปทำ
| ส่วน | รายละเอียด |
|---|---|
| เทคนิค (1H/4H/1D/1W) | EMA 9/21/50/200, RSI, StochRSI, MACD, ATR, ADX/DMI, Bollinger, **Keltner + Squeeze**, **Supertrend**, **Ichimoku**, **Donchian**, VWAP, OBV |
| แนวรับ–แนวต้าน | swing clustering (1D+4H), Pivot รายวัน, Fibonacci, EMA50/200 → รวมเป็นโซนพร้อมระดับความแข็งแรง |
| มหภาค | Real yield 10Y (FRED DFII10), ดอลลาร์ DXY พร้อมตรวจ correlation 120 วัน (ถ้าความสัมพันธ์อ่อน จะลดน้ำหนักเอง), Breakeven 5Y, Gold/Silver ratio, **GVZ** |
| COT | Managed Money net long, percentile 3 ปี (CFTC API) |
| ข่าว | ปฏิทินเศรษฐกิจ USD → Scenario "ถ้าออกมาสูง/ต่ำ/ตรงคาด ทองไปทางไหน" + กรอบแกว่งจาก GVZ + sentiment พาดหัวข่าว + (ไม่บังคับ) Gemini สรุปทิศทาง |
| แผนเทรด XM | Bias รวม −100…+100, โซนเข้า, SL (ATR), TP1/TP2 (R:R ≥ 1.5), **lot ขั้น 0.01** ตาม % ความเสี่ยง (รวม spread), **margin ตาม leverage**, โหมด Swing (ATR 1D) / Intraday (ATR 4H) |
| Backtest | In-sample 70% / Out-of-sample 30% เพื่อตรวจว่าสัญญาณเทคนิคไม่ได้ overfit |

## แหล่งข้อมูล (ฟรีทั้งหมด ไม่ต้องใช้ key)
Yahoo Finance (GC=F, SI=F, DX-Y.NYB, ^GVZ) · FRED (DFII10, T5YIE, GVZCLS) · CFTC Public Reporting (รหัสทอง 088691) · ForexFactory calendar JSON · Google News RSS
Gemini ใช้เฉพาะเมื่อใส่ `GEMINI_API_KEY`

## วิธีรัน
```bash
pip install -r requirements.txt
streamlit run app.py
# ดูหน้าตาแอปด้วยข้อมูลจำลอง: GOLD_DEMO=1 streamlit run app.py
pytest -q tests
```

## Deploy บน Streamlit Cloud
1. push โฟลเดอร์นี้ขึ้น GitHub repo ใหม่
2. Streamlit Cloud → New app → เลือก repo, main file = `app.py`
3. (ไม่บังคับ) Settings → Secrets ใส่ `GEMINI_API_KEY = "..."`

## โครงสร้างไฟล์
```
config.py      ค่าตั้งค่า สัญญา น้ำหนักคะแนน
data.py        ดึงข้อมูลทุกแหล่ง + ข้อมูลจำลอง
indicators.py  indicators + โหวตทิศทาง
levels.py      แนวรับ/แนวต้าน pivot fib
macro.py       มหภาค GVZ GSR COT
news.py        ปฏิทิน → scenario, sentiment, Gemini
engine.py      รวมคะแนน → แผนเทรด ขนาดไม้ คำเตือน
backtest.py    ทดสอบ IS/OOS
app.py         หน้าจอ Streamlit
```

## ข้อจำกัด
- XM ไม่เปิด API ราคาสาธารณะ → ต้องใส่ราคา GOLD จาก MT4/MT5 เอง แอปจะเลื่อนทุกระดับตามส่วนต่าง (futures − spot เปลี่ยนช้าๆ)
- สเปกค่าเริ่มต้น (1 lot = 100 oz, lot ต่ำสุด 0.01, leverage 1:500, spread 0.30) เป็นค่ามาตรฐานตลาด — **ตรวจค่าจริงใน MT5 → GOLD → Specification** แล้วแก้ในแถบซ้าย
- Margin = lots × contract size × ราคา ÷ leverage
- Scenario ข่าวเป็นกฎทั่วไป (ตัวเลขร้อน → Fed เข้มงวด → ทองถูกกดดัน) ซึ่งเปลี่ยนได้ตามช่วงตลาด
- Backtest ทดสอบเฉพาะส่วนเทคนิค ข่าวและมหภาคย้อนหลังไม่ได้ทดสอบ
