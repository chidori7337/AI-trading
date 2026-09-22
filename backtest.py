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

# =========================
# BACKTEST
# =========================

trades = []
next_available_index = 200

for i in range(next_available_index, len(df) - 1):
    current = df.iloc[i]

    price = current["close"]
    atr = current["ATR"]

    if pd.isna(atr):
        continue

    # -------------------------
    # PUNTUACIÓN LONG
    # -------------------------

    long_score = 0

    if price > current["EMA20"]:
        long_score += 1

    if current["EMA20"] > current["EMA50"]:
        long_score += 1

    if current["EMA50"] > current["EMA200"]:
        long_score += 1

    if current["RSI"] > 52:
        long_score += 1

    if current["MACD"] > current["MACD_SIGNAL"]:
        long_score += 1

    if current["MACD_HIST"] > df.iloc[i - 1]["MACD_HIST"]:
        long_score += 1

    if current["high"] > df.iloc[i - 1]["high"] and current["low"] > df.iloc[i - 1]["low"]:
        long_score += 1

    # -------------------------
    # PUNTUACIÓN SHORT
    # -------------------------

    short_score = 0

    if price < current["EMA20"]:
        short_score += 1

    if current["EMA20"] < current["EMA50"]:
        short_score += 1

    if current["EMA50"] < current["EMA200"]:
        short_score += 1

    if current["RSI"] < 48:
        short_score += 1

    if current["MACD"] < current["MACD_SIGNAL"]:
        short_score += 1

    if current["MACD_HIST"] < df.iloc[i - 1]["MACD_HIST"]:
        short_score += 1

    if current["high"] < df.iloc[i - 1]["high"] and current["low"] < df.iloc[i - 1]["low"]:
        short_score += 1

    # -------------------------
    # SEÑAL
    # -------------------------

    signal = None

    if (
        long_score >= 6
        and price > current["EMA20"]
        and current["EMA20"] > current["EMA50"]
        and long_score > short_score
    ):
        signal = "LONG"

    elif (
        short_score >= 6
        and price < current["EMA20"]
        and current["EMA20"] < current["EMA50"]
        and short_score > long_score
    ):
        signal = "SHORT"

    if signal is None:
        continue

    entry = price

    if signal == "LONG":
        stop = entry - atr * 1.5
        target = entry + atr * 3

    else:
        stop = entry + atr * 1.5
        target = entry - atr * 3

    # -------------------------
    # BUSCAR RESULTADO
    # -------------------------

    result = None

    for j in range(i + 1, len(df)):

        future = df.iloc[j]

        if signal == "LONG":

            if future["low"] <= stop:
                result = "LOSS"
                exit_price = stop
                break

            if future["high"] >= target:
                result = "WIN"
                exit_price = target
                break

        else:

            if future["high"] >= stop:
                result = "LOSS"
                exit_price = stop
                break

            if future["low"] <= target:
                result = "WIN"
                exit_price = target
                break

if result is None:
    continue

# La siguiente operación solo puede empezar
# después de cerrar esta operación
next_available_index = j + 1

trades.append({
        "datetime": current["datetime"],
        "signal": signal,
        "entry": entry,
        "stop": stop,
        "target": target,
        "result": result,
        "exit": exit_price
    })


# =========================
# RESULTADOS
# =========================

results = pd.DataFrame(trades)

print("\n=========================")
print("RESULTADOS BACKTEST")
print("=========================")

print("OPERACIONES:", len(results))

if len(results) > 0:

    wins = (results["result"] == "WIN").sum()
    losses = (results["result"] == "LOSS").sum()

    winrate = wins / len(results) * 100

    print("GANADORAS:", wins)
    print("PERDEDORAS:", losses)
    print(f"WIN RATE: {winrate:.2f}%")

    print("\nÚLTIMAS OPERACIONES:")

    print(
        results[
            ["datetime", "signal", "entry", "result"]
        ].tail(10).to_string(index=False)
    )

else:

    print("NO SE ENCONTRARON OPERACIONES")
