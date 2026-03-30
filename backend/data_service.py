import pandas as pd
import numpy as np
import yfinance as yf
import math, random
from datetime import datetime, timedelta

COMPANIES = [
    # Tech
    {"symbol":"AAPL","name":"Apple Inc.","sector":"Tech","mcap":"$3.12T"},
    {"symbol":"MSFT","name":"Microsoft Corp.","sector":"Tech","mcap":"$3.08T"},
    {"symbol":"GOOGL","name":"Alphabet Inc.","sector":"Tech","mcap":"$2.19T"},
    {"symbol":"NVDA","name":"NVIDIA Corp.","sector":"Tech","mcap":"$2.16T"},
    {"symbol":"META","name":"Meta Platforms","sector":"Tech","mcap":"$1.38T"},
    {"symbol":"AMZN","name":"Amazon.com Inc.","sector":"Tech","mcap":"$2.06T"},
    {"symbol":"TSLA","name":"Tesla Inc.","sector":"Tech","mcap":"$780B"},
    {"symbol":"ORCL","name":"Oracle Corp.","sector":"Tech","mcap":"$442B"},
    {"symbol":"CRM","name":"Salesforce Inc.","sector":"Tech","mcap":"$295B"},
    {"symbol":"AMD","name":"Advanced Micro Devices","sector":"Tech","mcap":"$262B"},
    {"symbol":"INTC","name":"Intel Corp.","sector":"Tech","mcap":"$95B"},
    {"symbol":"ADBE","name":"Adobe Inc.","sector":"Tech","mcap":"$178B"},
    {"symbol":"QCOM","name":"Qualcomm Inc.","sector":"Tech","mcap":"$168B"},
    {"symbol":"NFLX","name":"Netflix Inc.","sector":"Tech","mcap":"$412B"},
    {"symbol":"SNOW","name":"Snowflake Inc.","sector":"Tech","mcap":"$52B"},
    # Finance
    {"symbol":"JPM","name":"JPMorgan Chase","sector":"Finance","mcap":"$698B"},
    {"symbol":"BAC","name":"Bank of America","sector":"Finance","mcap":"$318B"},
    {"symbol":"GS","name":"Goldman Sachs","sector":"Finance","mcap":"$178B"},
    {"symbol":"MS","name":"Morgan Stanley","sector":"Finance","mcap":"$148B"},
    {"symbol":"V","name":"Visa Inc.","sector":"Finance","mcap":"$548B"},
    {"symbol":"MA","name":"Mastercard Inc.","sector":"Finance","mcap":"$478B"},
    {"symbol":"BRK-B","name":"Berkshire Hathaway","sector":"Finance","mcap":"$972B"},
    {"symbol":"AXP","name":"American Express","sector":"Finance","mcap":"$198B"},
    {"symbol":"C","name":"Citigroup Inc.","sector":"Finance","mcap":"$122B"},
    {"symbol":"WFC","name":"Wells Fargo","sector":"Finance","mcap":"$228B"},
    # Healthcare
    {"symbol":"JNJ","name":"Johnson & Johnson","sector":"Health","mcap":"$382B"},
    {"symbol":"UNH","name":"UnitedHealth Group","sector":"Health","mcap":"$412B"},
    {"symbol":"LLY","name":"Eli Lilly","sector":"Health","mcap":"$748B"},
    {"symbol":"PFE","name":"Pfizer Inc.","sector":"Health","mcap":"$148B"},
    {"symbol":"ABBV","name":"AbbVie Inc.","sector":"Health","mcap":"$312B"},
    {"symbol":"MRK","name":"Merck & Co.","sector":"Health","mcap":"$268B"},
    {"symbol":"TMO","name":"Thermo Fisher","sector":"Health","mcap":"$198B"},
    {"symbol":"ISRG","name":"Intuitive Surgical","sector":"Health","mcap":"$178B"},
    # Energy
    {"symbol":"XOM","name":"Exxon Mobil","sector":"Energy","mcap":"$488B"},
    {"symbol":"CVX","name":"Chevron Corp.","sector":"Energy","mcap":"$272B"},
    {"symbol":"COP","name":"ConocoPhillips","sector":"Energy","mcap":"$128B"},
    {"symbol":"SLB","name":"Schlumberger","sector":"Energy","mcap":"$62B"},
    {"symbol":"EOG","name":"EOG Resources","sector":"Energy","mcap":"$78B"},
    {"symbol":"NEE","name":"NextEra Energy","sector":"Energy","mcap":"$148B"},
    # Consumer
    {"symbol":"WMT","name":"Walmart Inc.","sector":"Consumer","mcap":"$748B"},
    {"symbol":"PG","name":"Procter & Gamble","sector":"Consumer","mcap":"$368B"},
    {"symbol":"KO","name":"Coca-Cola Co.","sector":"Consumer","mcap":"$268B"},
    {"symbol":"PEP","name":"PepsiCo Inc.","sector":"Consumer","mcap":"$198B"},
    {"symbol":"COST","name":"Costco Wholesale","sector":"Consumer","mcap":"$398B"},
    {"symbol":"MCD","name":"McDonald's Corp.","sector":"Consumer","mcap":"$212B"},
    {"symbol":"NKE","name":"Nike Inc.","sector":"Consumer","mcap":"$118B"},
    # Industrial
    {"symbol":"BA","name":"Boeing Co.","sector":"Industrial","mcap":"$108B"},
    {"symbol":"CAT","name":"Caterpillar Inc.","sector":"Industrial","mcap":"$178B"},
    {"symbol":"GE","name":"GE Aerospace","sector":"Industrial","mcap":"$228B"},
    {"symbol":"HON","name":"Honeywell Intl.","sector":"Industrial","mcap":"$128B"},
    {"symbol":"RTX","name":"RTX Corp.","sector":"Industrial","mcap":"$158B"},
    {"symbol":"UPS","name":"United Parcel Service","sector":"Industrial","mcap":"$98B"},
    # Media
    {"symbol":"DIS","name":"Walt Disney Co.","sector":"Media","mcap":"$188B"},
    {"symbol":"CMCSA","name":"Comcast Corp.","sector":"Media","mcap":"$148B"},
    {"symbol":"SPOT","name":"Spotify Technology","sector":"Media","mcap":"$78B"},
    {"symbol":"WBD","name":"Warner Bros. Discovery","sector":"Media","mcap":"$22B"},
    # Crypto-adjacent
    {"symbol":"COIN","name":"Coinbase Global","sector":"Crypto","mcap":"$62B"},
    {"symbol":"MSTR","name":"MicroStrategy","sector":"Crypto","mcap":"$42B"},
    {"symbol":"RIOT","name":"Riot Platforms","sector":"Crypto","mcap":"$4B"},
    {"symbol":"MARA","name":"Marathon Digital","sector":"Crypto","mcap":"$5B"},
]

COMPANY_MAP = {c["symbol"]: c for c in COMPANIES}

def _synthetic(symbol: str, days: int) -> pd.DataFrame:
    random.seed(abs(hash(symbol)) % 99999)
    base = random.uniform(50, 900)
    dates = pd.date_range(end=datetime.today(), periods=days, freq="B")
    prices = [base]
    for _ in range(len(dates) - 1):
        drift = random.gauss(0.0004, 0.018)
        prices.append(max(1.0, prices[-1] * (1 + drift)))
    df = pd.DataFrame({
        "Open":   [p * random.uniform(0.985, 1.000) for p in prices],
        "High":   [p * random.uniform(1.000, 1.025) for p in prices],
        "Low":    [p * random.uniform(0.975, 1.000) for p in prices],
        "Close":  prices,
        "Volume": [int(random.uniform(5e6, 80e6)) for _ in prices],
    }, index=pd.DatetimeIndex(dates.normalize()))
    return df

def fetch_ohlcv(symbol: str, days: int = 180) -> pd.DataFrame:
    fetch = max(days + 100, 200)
    try:
        end   = datetime.today()
        start = end - timedelta(days=fetch + 30)
        df = yf.Ticker(symbol).history(start=start, end=end)
        if df.empty or len(df) < 20:
            raise ValueError
        df = df[["Open","High","Low","Close","Volume"]].copy()
        df.index = pd.DatetimeIndex(pd.to_datetime(df.index).normalize())
    except Exception:
        df = _synthetic(symbol, fetch)

    df = df.dropna(subset=["Close"])

    # ── Indicators ──────────────────────────────────────
    df["Return"]     = df["Close"].pct_change()
    df["MA7"]        = df["Close"].rolling(7).mean()
    df["MA20"]       = df["Close"].rolling(20).mean()
    df["MA50"]       = df["Close"].rolling(50).mean()
    df["MA200"]      = df["Close"].rolling(200).mean()
    df["Volatility"] = df["Return"].rolling(20).std() * math.sqrt(252)
    df["Momentum"]   = df["Close"] / df["Close"].shift(10) - 1

    d  = df["Close"].diff()
    g  = d.clip(lower=0).rolling(14).mean()
    l  = (-d.clip(upper=0)).rolling(14).mean()
    df["RSI"] = 100 - 100 / (1 + g / (l + 1e-9))

    e12 = df["Close"].ewm(span=12).mean()
    e26 = df["Close"].ewm(span=26).mean()
    df["MACD"]      = e12 - e26
    df["MACD_Sig"]  = df["MACD"].ewm(span=9).mean()
    df["MACD_Hist"] = df["MACD"] - df["MACD_Sig"]

    df["BB_Mid"]  = df["Close"].rolling(20).mean()
    df["BB_Up"]   = df["BB_Mid"] + 2 * df["Close"].rolling(20).std()
    df["BB_Lo"]   = df["BB_Mid"] - 2 * df["Close"].rolling(20).std()

    w52 = df["Close"].rolling(252)
    df["52W_High"] = w52.max()
    df["52W_Low"]  = w52.min()

    df = df.dropna(subset=["MA20","RSI"])
    return df.tail(days)

def _safe(v):
    try:
        f = float(v)
        return None if (math.isnan(f) or math.isinf(f)) else round(f, 4)
    except Exception:
        return None

def compute_metrics(df: pd.DataFrame, symbol: str) -> dict:
    if df.empty or len(df) < 5:
        return {}
    last  = df.iloc[-1]
    prev  = df.iloc[-2]
    prev5 = df.iloc[max(-5, -len(df))]

    close      = float(last["Close"])
    prev_close = float(prev["Close"])
    change     = (close - prev_close) / prev_close * 100
    trend5     = (close - float(prev5["Close"])) / float(prev5["Close"]) * 100

    rsi        = _safe(last.get("RSI")) or 50
    vol        = _safe(last.get("Volatility")) or 0
    macd_hist  = _safe(last.get("MACD_Hist")) or 0
    momentum   = _safe(last.get("Momentum")) or 0
    above_ma20 = close > float(last["MA20"]) if _safe(last["MA20"]) else False
    above_ma50 = close > float(last["MA50"]) if _safe(last.get("MA50")) else False
    w52h       = _safe(last.get("52W_High"))
    w52l       = _safe(last.get("52W_Low"))
    bb_pos     = (close - float(last["BB_Lo"])) / (float(last["BB_Up"]) - float(last["BB_Lo"]) + 1e-9) if _safe(last.get("BB_Up")) else 0.5

    # Decision Score (0–100)
    score = 50
    if above_ma20:  score += 8
    if above_ma50:  score += 8
    if rsi < 30:    score += 12
    elif rsi < 50:  score += 6
    elif rsi > 70:  score -= 10
    elif rsi > 80:  score -= 15
    if macd_hist > 0: score += 8
    if momentum > 0:  score += 6
    if trend5 > 2:    score += 6
    elif trend5 < -2: score -= 6
    if vol > 0.5:   score -= 8
    if vol > 0.35:  score -= 4
    score = max(0, min(100, int(score)))

    if score >= 65:   signal, sig_color = "BUY", "#22c55e"
    elif score >= 45: signal, sig_color = "HOLD", "#f59e0b"
    else:             signal, sig_color = "RISKY", "#ef4444"

    # Trend
    if trend5 > 3 and above_ma20:   trend_label = "Strong Uptrend"
    elif trend5 > 0.5:               trend_label = "Uptrend"
    elif trend5 < -3 and not above_ma20: trend_label = "Strong Downtrend"
    elif trend5 < -0.5:              trend_label = "Downtrend"
    else:                            trend_label = "Sideways"

    meta = COMPANY_MAP.get(symbol, {})
    return {
        "symbol":      symbol,
        "name":        meta.get("name", symbol),
        "sector":      meta.get("sector",""),
        "mcap":        meta.get("mcap",""),
        "price":       round(close, 2),
        "change_pct":  round(change, 2),
        "trend5d_pct": round(trend5, 2),
        "decision_score": score,
        "signal":      signal,
        "signal_color":sig_color,
        "trend_label": trend_label,
        "rsi":         round(rsi, 1),
        "volatility":  round(vol * 100, 1),
        "macd_hist":   round(macd_hist, 4),
        "momentum":    round(momentum * 100, 2),
        "above_ma20":  bool(above_ma20),
        "above_ma50":  bool(above_ma50),
        "bb_position": round(bb_pos, 3),
        "w52_high":    round(w52h, 2) if w52h else None,
        "w52_low":     round(w52l, 2) if w52l else None,
        "ma20":        round(float(last["MA20"]), 2) if _safe(last["MA20"]) else None,
        "ma50":        round(float(last["MA50"]), 2) if _safe(last.get("MA50")) else None,
        "volume":      int(last["Volume"]),
    }

def detect_anomalies(df: pd.DataFrame) -> list:
    ret = df["Return"].dropna()
    mu, sigma = float(ret.mean()), float(ret.std())
    out = []
    for date, row in df.iterrows():
        r = _safe(row.get("Return"))
        if r is None: continue
        z = (r - mu) / (sigma + 1e-9)
        if abs(z) > 2.5:
            out.append({
                "date": str(date.date()),
                "return_pct": round(r * 100, 2),
                "z_score": round(z, 2),
                "type": "spike" if z > 0 else "crash",
                "severity": "extreme" if abs(z) > 3.5 else "high",
            })
    return out[-8:]

def simulate_investment(symbol: str, amount: float, days_ago: int) -> dict:
    df = fetch_ohlcv(symbol, max(days_ago + 10, 60))
    if len(df) < 2:
        return {"error": "insufficient data"}
    idx      = max(0, len(df) - days_ago)
    buy_row  = df.iloc[idx]
    sell_row = df.iloc[-1]
    buy_price  = float(buy_row["Close"])
    sell_price = float(sell_row["Close"])
    shares     = amount / buy_price
    current    = shares * sell_price
    profit     = current - amount
    ret_pct    = profit / amount * 100
    return {
        "symbol":      symbol,
        "amount_invested": round(amount, 2),
        "buy_price":   round(buy_price, 2),
        "buy_date":    str(buy_row.name.date()),
        "current_price": round(sell_price, 2),
        "current_value": round(current, 2),
        "profit_loss": round(profit, 2),
        "return_pct":  round(ret_pct, 2),
        "shares":      round(shares, 4),
        "outcome":     "profit" if profit > 0 else "loss",
    }

def get_alerts(symbol: str) -> list:
    df  = fetch_ohlcv(symbol, 90)
    if df.empty or len(df) < 5: return []
    met = compute_metrics(df, symbol)
    alerts = []
    rsi  = met.get("rsi", 50)
    vol  = met.get("volatility", 0)
    price = met.get("price", 0)
    w52h = met.get("w52_high")
    w52l = met.get("w52_low")

    if rsi > 75:
        alerts.append({"type":"overbought","level":"warning","msg":f"RSI at {rsi} — stock may be overbought. Consider taking profits."})
    if rsi < 30:
        alerts.append({"type":"oversold","level":"info","msg":f"RSI at {rsi} — stock appears oversold. Potential buying opportunity."})
    if vol > 40:
        alerts.append({"type":"volatility","level":"danger","msg":f"Annualized volatility at {vol}% — extremely high risk environment."})
    if w52h and price >= w52h * 0.98:
        alerts.append({"type":"52w_high","level":"info","msg":f"Trading near 52-week high of ${w52h:.2f} — strong momentum signal."})
    if w52l and price <= w52l * 1.03:
        alerts.append({"type":"52w_low","level":"danger","msg":f"Trading near 52-week low of ${w52l:.2f} — significant downside pressure."})
    return alerts
