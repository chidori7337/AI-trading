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
# DATAFRAME
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
# VWAP
# =========================
#
# Forex no tiene volumen centralizado.
# Usamos una referencia de precio ponderada
# intradía como aproximación, sin usarla
# como condición obligatoria.
#

typical_price = (
    df["high"] +
    df["low"] +
    df["close"]
) / 3

df["VWAP"] = typical_price.expanding().mean()


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

df["MACD_SIGNAL"] = (
    df["MACD"]
    .ewm(span=9, adjust=False)
    .mean()
)

df["MACD_HIST"] = (
    df["MACD"] -
    df["MACD_SIGNAL"]
)


# =========================
# ATR
# =========================

previous_close = df["close"].shift(1)

tr1 = df["high"] - df["low"]
tr2 = abs(df["high"] - previous_close)
tr3 = abs(df["low"] - previous_close)

true_range = pd.concat(
    [tr1, tr2, tr3],
    axis=1
).max(axis=1)

df["ATR"] = true_range.rolling(14).mean()


# =========================
# ESTRUCTURA
# =========================

df["HIGH_PREV"] = df["high"].shift(1)
df["LOW_PREV"] = df["low"].shift(1)

df["HIGHER_HIGH"] = df["high"] > df["HIGH_PREV"]
df["HIGHER_LOW"] = df["low"] > df["LOW_PREV"]

df["LOWER_HIGH"] = df["high"] < df["HIGH_PREV"]
df["LOWER_LOW"] = df["low"] < df["LOW_PREV"]


# =========================
# PUNTUACIÓN
# =========================

last = df.iloc[-1]
previous = df.iloc[-2]

long_score = 0
short_score = 0

long_reasons = []
short_reasons = []


# 1. PRECIO VS EMA20

if last["close"] > last["EMA20"]:
    long_score += 1
    long_reasons.append("Precio > EMA20")

if last["close"] < last["EMA20"]:
    short_score += 1
    short_reasons.append("Precio < EMA20")


# 2. EMA20 VS EMA50

if last["EMA20"] > last["EMA50"]:
    long_score += 1
    long_reasons.append("EMA20 > EMA50")

if last["EMA20"] < last["EMA50"]:
    short_score += 1
    short_reasons.append("EMA20 < EMA50")


# 3. EMA50 VS EMA200

if last["EMA50"] > last["EMA200"]:
    long_score += 1
    long_reasons.append("EMA50 > EMA200")

if last["EMA50"] < last["EMA200"]:
    short_score += 1
    short_reasons.append("EMA50 < EMA200")


# 4. RSI

if last["RSI"] > 50:
    long_score += 1
    long_reasons.append("RSI > 50")

if last["RSI"] < 50:
    short_score += 1
    short_reasons.append("RSI < 50")


# 5. MACD

if last["MACD"] > last["MACD_SIGNAL"]:
    long_score += 1
    long_reasons.append("MACD alcista")

if last["MACD"] < last["MACD_SIGNAL"]:
    short_score += 1
    short_reasons.append("MACD bajista")


# 6. HISTOGRAMA MACD

if last["MACD_HIST"] > previous["MACD_HIST"]:
    long_score += 1
    long_reasons.append("Histograma mejorando")

if last["MACD_HIST"] < previous["MACD_HIST"]:
    short_score += 1
    short_reasons.append("Histograma empeorando")


# 7. ESTRUCTURA

if last["HIGHER_HIGH"] and last["HIGHER_LOW"]:
    long_score += 1
    long_reasons.append("Estructura alcista")

if last["LOWER_HIGH"] and last["LOWER_LOW"]:
    short_score += 1
    short_reasons.append("Estructura bajista")


# 8. DISTANCIA A EMA20

distance = abs(last["close"] - last["EMA20"])

if distance < last["ATR"] * 1.5:

    if last["close"] > last["EMA20"]:
        long_score += 1
        long_reasons.append("Precio cerca de EMA20")

    if last["close"] < last["EMA20"]:
        short_score += 1
        short_reasons.append("Precio cerca de EMA20")


# =========================
# SEÑAL
# =========================

signal = "NO SIGNAL"

if long_score >= 6 and long_score > short_score:
    signal = "LONG"

elif short_score >= 6 and short_score > long_score:
    signal = "SHORT"


# =========================
# STOP / OBJETIVO
# =========================

entry = last["close"]
atr = last["ATR"]

if signal == "LONG":

    stop = entry - (atr * 1.5)
    target = entry + (atr * 3)

elif signal == "SHORT":

    stop = entry + (atr * 1.5)
    target = entry - (atr * 3)

else:

    stop = None
    target = None


# =========================
# RESULTADO
# =========================

print("\n==============================")
print("       AI TRADING RADAR")
print("==============================")

print(f"Símbolo: {SYMBOL}")
print(f"Timeframe: {INTERVAL}")
print(f"Hora: {last['datetime']}")

print("\nPRECIO")
print(f"Precio: {entry:.5f}")

print("\nINDICADORES")
print(f"EMA20:  {last['EMA20']:.5f}")
print(f"EMA50:  {last['EMA50']:.5f}")
print(f"EMA200: {last['EMA200']:.5f}")
print(f"VWAP:   {last['VWAP']:.5f}")
print(f"RSI:    {last['RSI']:.2f}")
print(f"MACD:   {last['MACD']:.6f}")
print(f"ATR:    {atr:.6f}")

print("\nSCORE")
print(f"LONG:  {long_score}/8")
print(f"SHORT: {short_score}/8")

print("\nSEÑAL")
print(signal)

if signal == "LONG" or signal == "SHORT":

    print("\nGESTIÓN DEL SETUP")
    print(f"Entrada: {entry:.5f}")
    print(f"Stop:    {stop:.5f}")
    print(f"Objetivo:{target:.5f}")

print("\nMOTIVOS LONG")
for reason in long_reasons:
    print(f"+ {reason}")

print("\nMOTIVOS SHORT")
for reason in short_reasons:
    print(f"+ {reason}")

print("\n==============================")
