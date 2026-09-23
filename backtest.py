import os
import requests
import pandas as pd


# ============================================================
# CONFIGURACIÓN
# ============================================================

API_KEY = os.getenv("TWELVE_DATA_API_KEY")

SYMBOL = "EUR/USD"
INTERVAL = "5min"
OUTPUT_SIZE = 5000

# Estrategia
SCORE_THRESHOLD = 4

# Gestión de riesgo
ATR_PERIOD = 14
SL_ATR = 1.5
TP_ATR = 2.5

# Estructura
STRUCTURE_LOOKBACK = 1


# ============================================================
# COMPROBAR API KEY
# ============================================================

if not API_KEY:
    raise ValueError("No se encontró TWELVE_DATA_API_KEY")


# ============================================================
# DESCARGAR DATOS
# ============================================================

url = "https://api.twelvedata.com/time_series"

params = {
    "symbol": SYMBOL,
    "interval": INTERVAL,
    "outputsize": OUTPUT_SIZE,
    "apikey": API_KEY,
    "format": "JSON"
}

response = requests.get(url, params=params, timeout=30)
data = response.json()

if "values" not in data:
    raise ValueError(f"Error descargando datos: {data}")


df = pd.DataFrame(data["values"])


# ============================================================
# PREPARAR DATOS
# ============================================================

df["datetime"] = pd.to_datetime(df["datetime"])

for column in ["open", "high", "low", "close"]:
    df[column] = pd.to_numeric(df[column], errors="coerce")

df = df.sort_values("datetime").reset_index(drop=True)

df.rename(
    columns={
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close"
    },
    inplace=True
)

df.dropna(
    subset=["Open", "High", "Low", "Close"],
    inplace=True
)

df.reset_index(drop=True, inplace=True)


print()
print("VELAS OBTENIDAS:", len(df))
print()
print("DESDE:", df["datetime"].iloc[0])
print()
print("HASTA:", df["datetime"].iloc[-1])


# ============================================================
# EMA
# ============================================================

df["EMA20"] = df["Close"].ewm(
    span=20,
    adjust=False
).mean()

df["EMA50"] = df["Close"].ewm(
    span=50,
    adjust=False
).mean()

df["EMA200"] = df["Close"].ewm(
    span=200,
    adjust=False
).mean()


# ============================================================
# RSI
# ============================================================

delta = df["Close"].diff()

gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)

avg_gain = gain.ewm(
    alpha=1 / 14,
    adjust=False
).mean()

avg_loss = loss.ewm(
    alpha=1 / 14,
    adjust=False
).mean()

rs = avg_gain / avg_loss

df["RSI"] = 100 - (100 / (1 + rs))


# ============================================================
# ATR
# ============================================================

previous_close = df["Close"].shift(1)

tr1 = df["High"] - df["Low"]
tr2 = (df["High"] - previous_close).abs()
tr3 = (df["Low"] - previous_close).abs()

true_range = pd.concat(
    [tr1, tr2, tr3],
    axis=1
).max(axis=1)

df["ATR"] = true_range.ewm(
    alpha=1 / ATR_PERIOD,
    adjust=False
).mean()


# ============================================================
# ELIMINAR FILAS SIN INDICADORES
# ============================================================

df.dropna(
    subset=[
        "EMA20",
        "EMA50",
        "EMA200",
        "RSI",
        "ATR"
    ],
    inplace=True
)

df.reset_index(drop=True, inplace=True)


print()
print("INDICADORES CALCULADOS CORRECTAMENTE")


# ============================================================
# BACKTEST
# ============================================================

trades = []

i = 200


while i < len(df) - 1:

    current = df.iloc[i]

    long_score = 0
    short_score = 0


    # ========================================================
    # LONG
    # ========================================================

    # 1. Precio por encima de EMA20
    if current["Close"] > current["EMA20"]:
        long_score += 1

    # 2. EMA20 por encima de EMA50
    if current["EMA20"] > current["EMA50"]:
        long_score += 1

    # 3. EMA50 por encima de EMA200
    if current["EMA50"] > current["EMA200"]:
        long_score += 1

    # 4. RSI alcista
    if current["RSI"] > 52:
        long_score += 1

    # 5. Estructura alcista
    if (
        current["High"] > df.iloc[i - 1]["High"]
        and current["Low"] > df.iloc[i - 1]["Low"]
    ):
        long_score += 1


    # ========================================================
    # SHORT
    # ========================================================

    # 1. Precio por debajo de EMA20
    if current["Close"] < current["EMA20"]:
        short_score += 1

    # 2. EMA20 por debajo de EMA50
    if current["EMA20"] < current["EMA50"]:
        short_score += 1

    # 3. EMA50 por debajo de EMA200
    if current["EMA50"] < current["EMA200"]:
        short_score += 1

    # 4. RSI bajista
    if current["RSI"] < 48:
        short_score += 1

    # 5. Estructura bajista
    if (
        current["High"] < df.iloc[i - 1]["High"]
        and current["Low"] < df.iloc[i - 1]["Low"]
    ):
        short_score += 1


    # ========================================================
    # DECIDIR SEÑAL
    # ========================================================

    signal = None

    if long_score >= SCORE_THRESHOLD:
        signal = "LONG"

    elif short_score >= SCORE_THRESHOLD:
        signal = "SHORT"


    # Sin señal
    if signal is None:
        i += 1
        continue


    # ========================================================
    # ENTRADA
    # ========================================================

    entry = current["Close"]
    atr = current["ATR"]

    if atr <= 0:
        i += 1
        continue


    # ========================================================
    # STOP Y TARGET
    # ========================================================

    if signal == "LONG":

        stop = entry - (SL_ATR * atr)
        target = entry + (TP_ATR * atr)

    else:

        stop = entry + (SL_ATR * atr)
        target = entry - (TP_ATR * atr)


    # ========================================================
    # BUSCAR SALIDA
    # ========================================================

    result = None
    result_r = None
    exit_index = None

    j = i + 1


    while j < len(df):

        future = df.iloc[j]

        high = future["High"]
        low = future["Low"]


        # ----------------------------------------------------
        # LONG
        # ----------------------------------------------------

        if signal == "LONG":

            hit_stop = low <= stop
            hit_target = high >= target

            # Si toca ambos en la misma vela,
            # asumimos primero el SL.
            if hit_stop:
                result = "LOSS"
                result_r = -1
                exit_index = j
                break

            if hit_target:
                result = "WIN"
                result_r = TP_ATR / SL_ATR
                exit_index = j
                break


        # ----------------------------------------------------
        # SHORT
        # ----------------------------------------------------

        else:

            hit_stop = high >= stop
            hit_target = low <= target

            # Si toca ambos en la misma vela,
            # asumimos primero el SL.
            if hit_stop:
                result = "LOSS"
                result_r = -1
                exit_index = j
                break

            if hit_target:
                result = "WIN"
                result_r = TP_ATR / SL_ATR
                exit_index = j
                break


        j += 1


    # ========================================================
    # SI NO SE CERRÓ LA OPERACIÓN
    # ========================================================

    if result is None:
        break


    # ========================================================
    # GUARDAR OPERACIÓN
    # ========================================================

    trades.append(
        {
            "datetime": current["datetime"],
            "signal": signal,
            "entry": entry,
            "stop": stop,
            "target": target,
            "result": result,
            "R": result_r
        }
    )


    # ========================================================
    # UNA SOLA OPERACIÓN ABIERTA A LA VEZ
    # ========================================================

    i = exit_index + 1


# ============================================================
# RESULTADOS
# ============================================================

results = pd.DataFrame(trades)


print()
print("=========================")
print("RESULTADOS BACKTEST")
print("=========================")


if results.empty:

    print()
    print("No se encontraron operaciones.")
    print()

else:

    total_trades = len(results)

    winners = (results["result"] == "WIN").sum()
    losers = (results["result"] == "LOSS").sum()

    win_rate = (winners / total_trades) * 100

    total_r = results["R"].sum()
    average_r = results["R"].mean()


    print()
    print("OPERACIONES:", total_trades)

    print()
    print("GANADORAS:", winners)

    print()
    print("PERDEDORAS:", losers)

    print()
    print(f"WIN RATE: {win_rate:.2f}%")

    print()
    print(f"RESULTADO TOTAL: {total_r:.2f} R")

    print()
    print(f"PROMEDIO POR OPERACIÓN: {average_r:.3f} R")


    # ========================================================
    # ÚLTIMAS OPERACIONES
    # ========================================================

    print()
    print("ÚLTIMAS OPERACIONES:")
    print()

    print(
        results[
            [
                "datetime",
                "signal",
                "entry",
                "result",
                "R"
            ]
        ].tail(10).to_string(index=False)
    )
