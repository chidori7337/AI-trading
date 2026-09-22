import os
import requests
import pandas as pd

API_KEY = os.environ["TWELVE_DATA_API_KEY"]

SYMBOL = "EUR/USD"
INTERVAL = "5min"

url = "https://api.twelvedata.com/time_series"

params = {
    "symbol": SYMBOL,
    "interval": INTERVAL,
    "outputsize": 5000,
    "apikey": API_KEY
}

response = requests.get(url, params=params)
data = response.json()

if "values" not in data:
    print("ERROR:")
    print(data)
    raise SystemExit(1)

df = pd.DataFrame(data["values"])

df["datetime"] = pd.to_datetime(df["datetime"])

for column in ["open", "high", "low", "close"]:
    df[column] = pd.to_numeric(df[column])

df = df.sort_values("datetime").reset_index(drop=True)

# =========================
# INDICADORES
# =========================

df["EMA20"] = df["close"].ewm(span=20, adjust=False).mean()
df["EMA50"] = df["close"].ewm(span=50, adjust=False).mean()
df["EMA200"] = df["close"].ewm(span=200, adjust=False).mean()

# RSI
delta = df["close"].diff()

gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)

average_gain = gain.rolling(14).mean()
average_loss = loss.rolling(14).mean()

rs = average_gain / average_loss

df["RSI"] = 100 - (100 / (1 + rs))

# MACD
ema12 = df["close"].ewm(span=12, adjust=False).mean()
ema26 = df["close"].ewm(span=26, adjust=False).mean()

df["MACD"] = ema12 - ema26
df["MACD_SIGNAL"] = df["MACD"].ewm(span=9, adjust=False).mean()

df["MACD_HIST"] = df["MACD"] - df["MACD_SIGNAL"]

# ATR
previous_close = df["close"].shift(1)

tr1 = df["high"] - df["low"]
tr2 = abs(df["high"] - previous_close)
tr3 = abs(df["low"] - previous_close)

true_range = pd.concat(
    [tr1, tr2, tr3],
    axis=1
).max(axis=1)

df["ATR"] = true_range.rolling(14).mean()

print("\nINDICADORES CALCULADOS CORRECTAMENTE")

print("VELAS OBTENIDAS:", len(df))
print("DESDE:", df["datetime"].iloc[0])
print("HASTA:", df["datetime"].iloc[-1])
