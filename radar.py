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

df["EMA20"] = df["close"].ewm(
    span=20,
    adjust=False
).mean()

df["EMA50"] = df["close"].ewm(
    span=50,
    adjust=False
).mean()

df["EMA200"] = df["close"].ewm(
    span=200,
    adjust=False
).mean()


# =========================
# VWAP APROXIMADO
# =========================

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

ema12 = df["close"].ewm(
    span=12,
    adjust=False
).mean()

ema26 = df["close"].ewm(
    span=26,
    adjust=False
).mean()

df["MACD"] = ema12 - ema26

df["MACD_SIGNAL"] = df["MACD"].ewm(
    span=9,
    adjust=False
).mean()

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

df["HIGHER_HIGH"] = (
    df["high"] > df["high"].shift(1)
)

df["HIGHER_LOW"] = (
    df["low"] > df["low"].shift(1)
)

df["LOWER_HIGH"] = (
    df["high"] < df["high"].shift(1)
)

df["LOWER_LOW"] = (
    df["low"] < df["low"].shift(1)
)


# =========================
# ÚLTIMA VELA
# =========================

last = df.iloc[-1]
previous = df.iloc[-2]

price = last["close"]
atr = last["ATR"]


# =========================
# SCORES
# =========================

long_score = 0
short_score = 0

long_reasons = []
short_reasons = []


# ==================================================
# BLOQUE 1: TENDENCIA
# Máximo 3 puntos
# ==================================================

# LONG

if price > last["EMA20"]:
    long_score += 1
    long_reasons.append("Precio > EMA20")

if last["EMA20"] > last["EMA50"]:
    long_score += 1
    long_reasons.append("EMA20 > EMA50")

if last["EMA50"] > last["EMA200"]:
    long_score += 1
    long_reasons.append("EMA50 > EMA200")


# SHORT

if price < last["EMA20"]:
    short_score += 1
    short_reasons.append("Precio < EMA20")

if last["EMA20"] < last["EMA50"]:
    short_score += 1
    short_reasons.append("EMA20 < EMA50")

if last["EMA50"] < last["EMA200"]:
    short_score += 1
    short_reasons.append("EMA50 < EMA200")


# ==================================================
# BLOQUE 2: MOMENTUM
# Máximo 3 puntos
# ==================================================

# LONG

if last["RSI"] > 52:
    long_score += 1
    long_reasons.append("RSI > 52")

if last["MACD"] > last["MACD_SIGNAL"]:
    long_score += 1
    long_reasons.append("MACD alcista")

if last["MACD_HIST"] > previous["MACD_HIST"]:
    long_score += 1
    long_reasons.append("Momentum MACD mejorando")


# SHORT

if last["RSI"] < 48:
    short_score += 1
    short_reasons.append("RSI < 48")

if last["MACD"] < last["MACD_SIGNAL"]:
    short_score += 1
    short_reasons.append("MACD bajista")

if last["MACD_HIST"] < previous["MACD_HIST"]:
    short_score += 1
    short_reasons.append("Momentum MACD empeorando")


# ==================================================
# BLOQUE 3: ESTRUCTURA + UBICACIÓN
# Máximo 2 puntos
# ==================================================

# LONG: estructura

if last["HIGHER_HIGH"] and last["HIGHER_LOW"]:
    long_score += 1
    long_reasons.append("Estructura alcista")


# LONG: ubicación respecto a VWAP

if price > last["VWAP"]:
    long_score += 1
    long_reasons.append("Precio > VWAP")


# SHORT: estructura

if last["LOWER_HIGH"] and last["LOWER_LOW"]:
    short_score += 1
    short_reasons.append("Estructura bajista")


# SHORT: ubicación respecto a VWAP

if price < last["VWAP"]:
    short_score += 1
    short_reasons.append("Precio < VWAP")


# =========================
# FILTRO DE TENDENCIA
# =========================

long_trend = (
    price > last["EMA20"]
    and last["EMA20"] > last["EMA50"]
)

short_trend = (
    price < last["EMA20"]
    and last["EMA20"] < last["EMA50"]
)


# =========================
# SEÑAL
# =========================

signal = "NO SIGNAL"

if long_score >= 6 and long_trend and long_score > short_score:
    signal = "LONG"

elif short_score >= 6 and short_trend and short_score > long_score:
    signal = "SHORT"


# =========================
# STOP / OBJETIVO
# =========================

entry = price

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

print()
print("==============================")
print("       AI TRADING RADAR")
print("==============================")

print(f"Símbolo:   {SYMBOL}")
print(f"Timeframe: {INTERVAL}")
print(f"Hora:      {last['datetime']}")

print()
print("PRECIO")
print(f"Precio: {price:.5f}")

print()
print("TENDENCIA")
print(f"EMA20:  {last['EMA20']:.5f}")
print(f"EMA50:  {last['EMA50']:.5f}")
print(f"EMA200: {last['EMA200']:.5f}")
print(f"VWAP:   {last['VWAP']:.5f}")

print()
print("MOMENTUM")
print(f"RSI:    {last['RSI']:.2f}")
print(f"MACD:   {last['MACD']:.6f}")
print(f"Signal: {last['MACD_SIGNAL']:.6f}")

print()
print("VOLATILIDAD")
print(f"ATR:    {atr:.6f}")

print()
print("SCORE")
print(f"LONG:  {long_score}/8")
print(f"SHORT: {short_score}/8")

print()
print("SEÑAL")
print(signal)


# =========================
# SETUP
# =========================

if signal == "LONG" or signal == "SHORT":

    print()
    print("GESTIÓN DEL SETUP")
    print(f"Entrada:  {entry:.5f}")
    print(f"Stop:     {stop:.5f}")
    print(f"Objetivo: {target:.5f}")


# =========================
# MOTIVOS
# =========================

print()
print("MOTIVOS LONG")

if long_reasons:
    for reason in long_reasons:
        print(f"+ {reason}")
else:
    print("Ninguno")


print()
print("MOTIVOS SHORT")

if short_reasons:
    for reason in short_reasons:
        print(f"+ {reason}")
else:
    print("Ninguno")


print()
print("==============================")
