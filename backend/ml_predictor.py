import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import MinMaxScaler
from datetime import timedelta

def predict_next_7(df: pd.DataFrame) -> dict:
    df = df.dropna(subset=["Close","MA7","MA20","RSI"])
    if len(df) < 40:
        return {"error": "insufficient data"}

    feats = [c for c in ["MA7","MA20","RSI","Volatility","Momentum","MACD","MACD_Hist"] if c in df.columns]
    X = df[feats].ffill().fillna(0).values
    y = df["Close"].values

    sx = MinMaxScaler(); sy = MinMaxScaler()
    Xs = sx.fit_transform(X)
    ys = sy.fit_transform(y.reshape(-1,1)).ravel()

    model = Ridge(alpha=1.0)
    model.fit(Xs, ys)

    last_x = X[-1].copy()
    preds  = []
    for i in range(7):
        xp   = sx.transform(last_x.reshape(1,-1))
        yp   = model.predict(xp)[0]
        price= float(sy.inverse_transform([[yp]])[0][0])
        preds.append(round(price, 2))
        if "Momentum" in feats:
            idx = feats.index("Momentum")
            last_x[idx] *= 0.92

    last_date = df.index[-1]
    dates = []
    d = last_date
    for _ in range(7):
        d = d + timedelta(days=1)
        while d.weekday() >= 5:
            d = d + timedelta(days=1)
        dates.append(str(d.date()))

    hist_n = 30
    return {
        "historical_dates":  [str(d.date()) for d in df.index[-hist_n:]],
        "historical_prices": [round(float(p),2) for p in df["Close"].values[-hist_n:]],
        "future_dates":      dates,
        "predicted_prices":  preds,
        "confidence_up":     [round(p*1.025,2) for p in preds],
        "confidence_down":   [round(p*0.975,2) for p in preds],
        "direction":         "up" if preds[-1] > preds[0] else "down",
        "expected_change_pct": round((preds[-1]-preds[0])/preds[0]*100, 2),
    }
