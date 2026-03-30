"""
VaultIQ — FastAPI Backend
Institutional-grade stock intelligence platform
"""

from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
import os, asyncio, time
from datetime import datetime
from typing import Optional

from data_service import (
    COMPANIES, COMPANY_MAP, fetch_ohlcv,
    compute_metrics, detect_anomalies, simulate_investment, get_alerts,
    _safe
)
from ml_predictor import predict_next_7

# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="VaultIQ API",
    description="AI-Powered Smart Investor Dashboard — Backend",
    version="2.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Static frontend ────────────────────────────────────────────────────────────
FRONTEND = os.path.join(os.path.dirname(__file__), "..", "frontend", "dist")
if os.path.exists(FRONTEND):
    assets_path = os.path.join(FRONTEND, "assets")
    if os.path.exists(assets_path):
        app.mount("/assets", StaticFiles(directory=assets_path), name="assets")

# ── In-memory cache (TTL = 5 minutes) ─────────────────────────────────────────
_cache: dict = {}
CACHE_TTL = 300  # seconds

def cache_get(key: str):
    entry = _cache.get(key)
    if entry and (time.time() - entry["ts"]) < CACHE_TTL:
        return entry["val"]
    return None

def cache_set(key: str, val):
    _cache[key] = {"val": val, "ts": time.time()}
    return val


# ── Root ───────────────────────────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
async def root():
    p = os.path.join(FRONTEND, "index.html")
    if os.path.exists(p):
        return FileResponse(p)
    return {"status": "VaultIQ API running", "docs": "/docs", "version": "2.0"}


# ── Health check ───────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat(), "companies": len(COMPANIES)}


# ── 1. Companies list ──────────────────────────────────────────────────────────
@app.get("/companies", summary="List all tracked companies")
async def companies(sector: str = Query("", description="Filter by sector name")):
    """Return all 60 companies, optionally filtered by sector."""
    if sector and sector.lower() != "all":
        return [c for c in COMPANIES if c["sector"].lower() == sector.lower()]
    return COMPANIES


# ── 2. Stock data + all indicators ────────────────────────────────────────────
@app.get("/data/{symbol}", summary="OHLCV + 12 technical indicators")
async def stock_data(
    symbol: str,
    days: int = Query(90, ge=7, le=365, description="Number of trading days")
):
    """Fetch OHLCV price data with all computed technical indicators."""
    sym = symbol.upper()
    cache_key = f"data:{sym}:{days}"
    cached = cache_get(cache_key)
    if cached:
        return cached

    df  = fetch_ohlcv(sym, days)
    met = compute_metrics(df, sym)

    records = []
    for date, row in df.iterrows():
        records.append({
            "date":      str(date.date()),
            "open":      round(float(row["Open"]), 2),
            "high":      round(float(row["High"]), 2),
            "low":       round(float(row["Low"]), 2),
            "close":     round(float(row["Close"]), 2),
            "volume":    int(row["Volume"]),
            "return":    _safe(row.get("Return")),
            "ma7":       _safe(row.get("MA7")),
            "ma20":      _safe(row.get("MA20")),
            "ma50":      _safe(row.get("MA50")),
            "rsi":       _safe(row.get("RSI")),
            "macd":      _safe(row.get("MACD")),
            "macd_sig":  _safe(row.get("MACD_Sig")),
            "macd_hist": _safe(row.get("MACD_Hist")),
            "bb_up":     _safe(row.get("BB_Up")),
            "bb_lo":     _safe(row.get("BB_Lo")),
            "vol":       _safe(row.get("Volatility")),
        })

    result = {
        "symbol":  sym,
        "meta":    COMPANY_MAP.get(sym, {}),
        "metrics": met,
        "data":    records,
    }
    return cache_set(cache_key, result)


# ── 3. Summary / Decision Score ───────────────────────────────────────────────
@app.get("/summary/{symbol}", summary="Decision Score + key metrics")
async def summary(symbol: str, days: int = Query(90, ge=7, le=365)):
    """Compute and return the Decision Score and all key metrics."""
    sym = symbol.upper()
    cache_key = f"summary:{sym}:{days}"
    cached = cache_get(cache_key)
    if cached:
        return cached

    df  = fetch_ohlcv(sym, days)
    met = compute_metrics(df, sym)
    return cache_set(cache_key, met)


# ── 4. Compare multiple stocks ────────────────────────────────────────────────
@app.get("/compare", summary="Normalized multi-stock comparison")
async def compare(
    symbols: str = Query("AAPL,MSFT,GOOGL", description="Comma-separated symbols (max 5)"),
    days: int = Query(90, ge=7, le=365)
):
    """Compare up to 5 stocks on a normalized (base=100) price chart."""
    syms = [s.strip().upper() for s in symbols.split(",")][:5]
    result = {}

    async def _fetch_one(sym):
        try:
            df = fetch_ohlcv(sym, days)
            if df.empty:
                return
            base = float(df["Close"].iloc[0])
            result[sym] = {
                "meta":        COMPANY_MAP.get(sym, {}),
                "normalized":  [(str(d.date()), round(float(p) / base * 100, 2))
                                for d, p in zip(df.index, df["Close"])],
                "last_price":  round(float(df["Close"].iloc[-1]), 2),
                "metrics":     compute_metrics(df, sym),
            }
        except Exception:
            pass

    await asyncio.gather(*[_fetch_one(s) for s in syms])
    return result


# ── 5. ML Price Prediction ────────────────────────────────────────────────────
@app.get("/predict/{symbol}", summary="7-day ML price forecast")
async def predict(symbol: str):
    """Ridge Regression 7-day price forecast with confidence intervals."""
    sym = symbol.upper()
    cache_key = f"predict:{sym}"
    cached = cache_get(cache_key)
    if cached:
        return cached

    df = fetch_ohlcv(sym, 250)
    result = predict_next_7(df)
    return cache_set(cache_key, result)


# ── 6. AI Context (system prompt for Vaultara) ────────────────────────────────
@app.get("/ai-context/{symbol}", summary="AI analyst system prompt context")
async def ai_context(
    symbol: str,
    days: int  = Query(90, ge=7, le=365),
    mode: str  = Query("moderate", description="conservative | moderate | aggressive")
):
    """
    Returns a rich structured context string for injecting into the Claude API
    as the Vaultara AI analyst system prompt.
    """
    sym = symbol.upper()
    cache_key = f"ai_ctx:{sym}:{days}:{mode}"
    cached = cache_get(cache_key)
    if cached:
        return cached

    df   = fetch_ohlcv(sym, days)
    met  = compute_metrics(df, sym)
    pred = predict_next_7(df)
    anom = detect_anomalies(df)
    alrt = get_alerts(sym)

    mode_desc = {
        "conservative": "very risk-averse, prefers stable blue-chip stocks, dividend yield, capital preservation over 5+ year horizon",
        "moderate":     "balanced risk/reward, comfortable with moderate volatility, 3–5 year investment horizon",
        "aggressive":   "high risk tolerance, seeks maximum capital appreciation, comfortable with speculative positions and short-term trades",
    }.get(mode, "moderate")

    context = f"""You are Vaultara, an elite AI financial analyst for VaultIQ — a premium institutional-grade investment platform. You speak with authority, precision, and sharp wit like a top-tier Wall Street analyst. Never say you are an AI; you are Vaultara. Keep responses concise, specific, and data-driven.

━━━ STOCK ANALYSIS: {sym} ━━━
Company: {met.get('name','N/A')} | Sector: {met.get('sector','N/A')} | Market Cap: {met.get('mcap','N/A')}
Current Price: ${met.get('price','N/A')} | 1-day: {met.get('change_pct','N/A')}% | 5-day trend: {met.get('trend5d_pct','N/A')}%
Decision Score: {met.get('decision_score','N/A')}/100 | Signal: {met.get('signal','N/A')}
RSI-14: {met.get('rsi','N/A')} | Annualized Volatility: {met.get('volatility','N/A')}%
Above MA-20: {met.get('above_ma20','N/A')} | Above MA-50: {met.get('above_ma50','N/A')}
Trend: {met.get('trend_label','N/A')} | 10-day Momentum: {met.get('momentum','N/A')}%
Bollinger Band Position: {met.get('bb_position','N/A')} (0=lower, 1=upper band)
52W High: ${met.get('w52_high','N/A')} | 52W Low: ${met.get('w52_low','N/A')}
MA-20: ${met.get('ma20','N/A')} | MA-50: ${met.get('ma50','N/A')}
━━━ ML FORECAST ━━━
7-Day Direction: {pred.get('direction','N/A')} | Expected Change: {pred.get('expected_change_pct','N/A')}%
━━━ MARKET SIGNALS ━━━
Anomalies Detected (last period): {len(anom)} unusual price events
Active Smart Alerts: {len(alrt)} alerts
━━━ INVESTOR PROFILE ━━━
Mode: {mode.upper()} — {mode_desc}

Respond as Vaultara. Be sharp, specific, and data-driven. Reference actual numbers. Use **bold** for key terms."""

    result = {"context": context, "metrics": met, "prediction": pred, "anomalies": anom, "alerts": alrt}
    return cache_set(cache_key, result)


# ── 7. What-If Investment Simulator ───────────────────────────────────────────
@app.get("/simulate", summary="What-if investment calculator")
async def simulate(
    symbol:   str   = Query("AAPL"),
    amount:   float = Query(10000, gt=0, description="Investment amount in USD"),
    days_ago: int   = Query(90, ge=1, le=365, description="Days ago to simulate buy")
):
    """Calculate exact P&L for a hypothetical past investment."""
    return simulate_investment(symbol.upper(), amount, days_ago)


# ── 8. Smart Alerts ───────────────────────────────────────────────────────────
@app.get("/alerts/{symbol}", summary="Smart alert triggers")
async def alerts(symbol: str):
    """Returns active smart alerts: RSI extremes, volatility spikes, 52W extremes."""
    sym = symbol.upper()
    cache_key = f"alerts:{sym}"
    cached = cache_get(cache_key)
    if cached:
        return cached
    result = {"symbol": sym, "alerts": get_alerts(sym)}
    return cache_set(cache_key, result)


# ── 9. Full Market Summary (all 60 stocks) ────────────────────────────────────
@app.get("/market/summary", summary="Full market snapshot — all companies")
async def market_summary():
    """
    Fetches price + Decision Score for all 60 tracked companies.
    Results are cached for 5 minutes to avoid hammering yfinance.
    """
    cache_key = "market_summary"
    cached = cache_get(cache_key)
    if cached:
        return cached

    results = []

    async def _process(c):
        try:
            df  = fetch_ohlcv(c["symbol"], 30)
            met = compute_metrics(df, c["symbol"])
            if met:
                results.append({**c, **{
                    "price":          met.get("price"),
                    "change_pct":     met.get("change_pct"),
                    "decision_score": met.get("decision_score"),
                    "signal":         met.get("signal"),
                    "rsi":            met.get("rsi"),
                    "volatility":     met.get("volatility"),
                    "trend_label":    met.get("trend_label"),
                }})
        except Exception:
            pass

    # Run all 60 fetches concurrently
    await asyncio.gather(*[_process(c) for c in COMPANIES])

    return cache_set(cache_key, results)


# ── 10. Anomaly Detection ─────────────────────────────────────────────────────
@app.get("/anomalies/{symbol}", summary="Price anomaly detection")
async def anomalies(
    symbol: str,
    days: int = Query(180, ge=30, le=365)
):
    """Detect unusual price spikes/drops using Z-score analysis (|z| > 2.5σ)."""
    sym = symbol.upper()
    df  = fetch_ohlcv(sym, days)
    return {"symbol": sym, "anomalies": detect_anomalies(df)}


# ── Cache management ──────────────────────────────────────────────────────────
@app.delete("/cache", include_in_schema=False)
async def clear_cache():
    """Clear all cached data (admin use)."""
    _cache.clear()
    return {"message": "Cache cleared", "entries_removed": len(_cache)}
