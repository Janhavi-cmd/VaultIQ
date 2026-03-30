# VaultIQ — AI-Powered Smart Investor Dashboard
### Internship Project · Built to Senior Engineer Standard

> An institutional-grade stock intelligence platform with AI-powered insights, 60 companies across 8 sectors, multi-page UX, and a luxury terminal aesthetic.

---

## 🚀 Quick Start (2 minutes)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start the backend
cd backend
uvicorn main:app --reload --port 8000

# 3. Open the app
# Visit: http://localhost:8000
# OR simply open frontend/dist/index.html in a browser (works offline too)
```

> **Works offline too!** All 60 companies with prices, signals, and scores are embedded in the frontend. The grid loads instantly — no backend required to browse companies.

---

## 🏗 Architecture

```
VaultIQ/
├── backend/
│   ├── main.py            ← FastAPI — 10 async REST endpoints, caching
│   ├── data_service.py    ← yfinance + 60 companies + 12 indicators
│   └── ml_predictor.py    ← Ridge Regression 7-day price forecast
├── frontend/
│   └── dist/
│       └── index.html     ← Self-contained SPA (Home → Loading → Dashboard)
├── requirements.txt
└── README.md
```

---

## ✅ Feature Checklist

| Feature | Status | Implementation Detail |
|---|---|---|
| **Decision Intelligence System** | ✅ | Custom 0–100 score from 7 weighted factors: MA crossover, RSI zone, MACD histogram, momentum, volatility, 5-day trend |
| **Investor Personality Mode** | ✅ | Conservative / Moderate / Aggressive — adjusts every Vaultara AI response |
| **AI Analyst Persona (Vaultara)** | ✅ | Named AI analyst with full stock context injected, institutional language, fallback responses |
| **7-Day ML Prediction** | ✅ | Ridge Regression with MinMaxScaler, confidence bands (±2.5%), business-day forecast |
| **What-If Investment Simulator** | ✅ | Exact P&L, shares, buy/current price for any amount × any past date |
| **Anomaly Detection** | ✅ | Z-score > 2.5σ flags spikes/crashes, severity levels (high/extreme) |
| **ELI5 Mode** | ✅ | Ask Vaultara "explain like I'm 5" — switches to lemonade stand metaphors |
| **Smart Alerts** | ✅ | RSI overbought/oversold, volatility >40%, near 52-week extremes |
| **60 Companies, 8 Sectors** | ✅ | Tech, Finance, Health, Energy, Consumer, Industrial, Media, Crypto |
| **Sector Filter + Search** | ✅ | Instant client-side filtering, search by ticker/name/sector |
| **Live Ticker Tape** | ✅ | Scrolling prices across the top, live from API or static fallback |
| **Luxury Terminal UI** | ✅ | IBM Plex Mono + DM Serif Display, deep navy, gold accents, grain overlay |
| **Interactive Charts** | ✅ | Price + MA, MACD, Bollinger Bands, AI Forecast (Chart.js) |
| **RSI + Volume Charts** | ✅ | Side-by-side with overbought/oversold reference lines |
| **Multi-Page Flow** | ✅ | Home → Cinematic loading sequence → Full dashboard |
| **Offline Mode** | ✅ | Static fallback for all 60 companies — works without backend |

---

## 🎯 API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/companies?sector=Tech` | GET | 60 companies, filterable by sector |
| `/data/{symbol}?days=90` | GET | OHLCV + 12 computed indicators |
| `/summary/{symbol}` | GET | Decision Score + all metrics |
| `/compare?symbols=AAPL,MSFT,GOOGL` | GET | Normalized multi-stock comparison |
| `/predict/{symbol}` | GET | 7-day ML price forecast |
| `/ai-context/{symbol}?mode=moderate` | GET | Structured AI system prompt context |
| `/simulate?symbol=AAPL&amount=10000&days_ago=90` | GET | What-if investment simulator |
| `/alerts/{symbol}` | GET | Smart alert triggers |
| `/market/summary` | GET | Full market snapshot (all 60 stocks) |
| `/anomalies/{symbol}` | GET | Z-score anomaly detection |

---

## 📊 Technical Indicators Computed

| Indicator | Window | Use |
|---|---|---|
| Daily Return | — | Anomaly detection, momentum |
| MA-7 / MA-20 / MA-50 / MA-200 | Rolling | Trend detection, Decision Score |
| RSI-14 | 14-day | Overbought/oversold signals |
| MACD (12/26/9) | EMA-based | Momentum direction |
| Bollinger Bands | 20-day ±2σ | Volatility bands, position |
| Volatility | 20-day annualized | Risk scoring |
| 52-Week High/Low | Rolling 252 | Extreme alerts |

---

## 🧠 AI Chat — Vaultara

Every chat response uses the Anthropic Claude API with a rich system prompt containing:

- Full stock metrics (price, RSI, MACD, volatility, Decision Score)
- Investor mode preference (conservative/moderate/aggressive)
- 7-day ML forecast direction + magnitude
- Anomaly count and active alerts
- Conversation memory (last 8 turns)

**Built-in Quick Chips:**
- Buy or Sell? · ELI5 Mode 🧸 · Key Risks · Technical View · AI Forecast · Sector Compare

**Fallback Mode:** If the Anthropic API key is not configured, Vaultara uses rule-based response generation that still gives meaningful, data-driven answers from the local metrics.

---

## 💡 Design Philosophy

VaultIQ was designed to feel like a product from a real fintech startup, not a school project. Key choices:

- **Offline-first**: 60 companies work immediately without any backend. Recruiters can open `index.html` directly.
- **Progressive enhancement**: Static data → API enrichment → AI analysis, each layer adding value
- **Institutional aesthetic**: The "luxury terminal" look (gold + deep navy + monospace) signals financial professionalism
- **Named AI persona**: Vaultara isn't "AI Assistant #3" — she's a character with personality that makes demos memorable

---

## 🔧 Extending the Project

**Add more companies:** Edit `COMPANIES` list in `data_service.py` and `STATIC_COMPANIES` in `index.html`

**Improve the ML model:** Replace Ridge Regression in `ml_predictor.py` with LSTM or Prophet for better accuracy

**Add authentication:** Wrap FastAPI endpoints with OAuth2, add a user portfolio persistence layer (SQLite already in requirements)

**Real-time prices:** Replace yfinance polling with a WebSocket feed from Alpaca or Polygon.io

---

## 📦 Dependencies

```
fastapi==0.115.0        ← Async REST API framework
uvicorn[standard]       ← ASGI server
yfinance==0.2.40        ← Free stock data (Yahoo Finance)
pandas==2.2.2           ← Data processing
numpy==1.26.4           ← Numerical computing
scikit-learn==1.5.1     ← ML (Ridge Regression, MinMaxScaler)
python-multipart        ← FastAPI form support
aiofiles                ← Async file serving
```

Frontend uses only CDN-loaded libraries (no npm/build step):
- **Chart.js 4.4.1** — All charts
- **Google Fonts** — DM Serif Display + IBM Plex Mono + DM Sans

---

*VaultIQ — Making institutional-grade research accessible.*
