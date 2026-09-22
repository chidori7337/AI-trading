import os
import requests
import pandas as pd


# =========================
# CONFIGURACIÓN
# =========================

API_KEY = os.environ["TWELVE_DATA_API_KEY"]

SYMBOL = "EUR/USD"
INTERVAL = "5min"


# =========================
# OBTENER DATOS
# =========================

url = "https://api.twelvedata.com/time_series"

params = {
    "symbol": SYMBOL,
    "interval": INTERVAL,
    "outputsize": 250,
    "apikey": API_KEY
}

response = requests.get(url, params=params)
data = response.json()

if "values" not in data:
    print("ERROR OBTENIENDO DATOS:")
    print(data)
    raise SystemExit(1)


# =========================
# CREAR DATAFRAME
# =========================

df = pd.DataFrame(data["values"])

df["datetime"] = pd.to_datetime(df["datetime"])

for column in ["open", "high", "low", "close"]:
    df[column] = pd.to_numeric(df[column])

df = df.sort_values("datetime").reset_index(drop=True)


# =========================
# EMA
# =========================

df["EMA20"] = df["close"].ewm(span=20, adjust=False).mean()
df["EMA50"] = df["close"].ewm(span=50, adjust=False).mean()
df["EMA200"] = df["close"].ewm(span=200, adjust=False).mean()


# =========================
# RSI
# =========================

delta = df["close"].diff()

gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)

average_gain = gain.rolling(14).mean()
average_loss = loss.rolling(14).mean()

rs = average_gain / average_loss

df["RSI"] = 100 - (100 / (1 + rs))


# =========================
# MACD
# =========================

ema12 = df["close"].ewm(span=12, adjust=False).mean()
ema26 = df["close"].ewm(span=26, adjust=False).mean()

df["MACD"] = ema12 - ema26
df["MACD_SIGNAL"] = df["MACD"].ewm(span=9, adjust=False).mean()
df["MACD_HIST"] = df["MACD"] - df["MACD_SIGNAL"]


# =========================
# ATR
# =========================

previous_close = df["close"].shift(1)

tr1 = df["high"] - df["low"]
tr2 = abs(df["high"] - previous_close)
tr3 = abs(df["low"] - previous_close)

true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

df["ATR"] = true_range.rolling(14).mean()


# =========================
# VOLUMEN RELATIVO
# =========================

if "volume" in df.columns:

    df["volume"] = pd.to_numeric(
        df["volume"],
        errors="coerce"
    )

    df["AVG_VOLUME"] = df["volume"].rolling(20).mean()

    df["RELATIVE_VOLUME"] = (
        df["volume"] / df["AVG_VOLUME"]
    )

else:

    df["RELATIVE_VOLUME"] = None


# =========================
# ÚLTIMA VELA
# =========================

last = df.iloc[-1]


print("\n==============================")
print("      AI TRADING RADAR")
print("==============================")

print(f"Símbolo: {SYMBOL}")
print(f"Intervalo: {INTERVAL}")
print(f"Hora: {last['datetime']}")

print("\nPRECIO")
print(f"Close: {last['close']:.5f}")

print("\nMEDIAS")
print(f"EMA20:  {last['EMA20']:.5f}")
print(f"EMA50:  {last['EMA50']:.5f}")
print(f"EMA200: {last['EMA200']:.5f}")

print("\nMOMENTUM")
print(f"RSI: {last['RSI']:.2f}")
print(f"MACD: {last['MACD']:.6f}")
print(f"Signal: {last['MACD_SIGNAL']:.6f}")

print("\nVOLATILIDAD")
print(f"ATR: {last['ATR']:.6f}")

print("\nVOLUMEN")
print(f"Relative Volume: {last['RELATIVE_VOLUME']}")

print("\n==============================")
