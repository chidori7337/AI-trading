import os
import time
import requests
import pandas as pd
import numpy as np

API_KEY = os.getenv("TWELVE_DATA_API_KEY")

if not API_KEY:
    raise ValueError("Falta TWELVE_DATA_API_KEY")

SYMBOL = "EUR/USD"
INTERVAL = "5min"
OUTPUTSIZE = 5000

START_YEAR = 2025
END_YEAR = 2025

SL_ATR = 1.5
TP_ATR = 3.0
BODY_ATR_MIN = 0.25


# ============================================================
# DATOS
# ============================================================

def get_data(end_date):

    url = "https://api.twelvedata.com/time_series"

    params = {
        "symbol": SYMBOL,
        "interval": INTERVAL,
        "outputsize": OUTPUTSIZE,
        "end_date": end_date,
        "apikey": API_KEY,
        "format": "JSON"
    }

    for attempt in range(5):

        try:
            r = requests.get(url, params=params, timeout=30)
            data = r.json()

            if "values" in data:
                df = pd.DataFrame(data["values"])

                df["datetime"] = pd.to_datetime(df["datetime"])

                for col in ["open", "high", "low", "close"]:
                    df[col] = pd.to_numeric(df[col])

                df = df.sort_values("datetime").reset_index(drop=True)

                return df

            if data.get("code") == 429:
                wait = 30 * (attempt + 1)
                print(f"429. Esperando {wait}s...")
                time.sleep(wait)
                continue

            print("ERROR API:", data)
            return None

        except Exception as e:
            print("ERROR:", e)
            time.sleep(10 * (attempt + 1))

    return None


# ============================================================
# INDICADORES
# ============================================================

def indicators(df):

    df = df.copy()

    df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()
    df["ema200"] = df["close"].ewm(span=200, adjust=False).mean()

    prev_close = df["close"].shift(1)

    tr1 = df["high"] - df["low"]
    tr2 = abs(df["high"] - prev_close)
    tr3 = abs(df["low"] - prev_close)

    df["tr"] = pd.concat(
        [tr1, tr2, tr3],
        axis=1
    ).max(axis=1)

    df["atr"] = df["tr"].rolling(14).mean()

    return df


# ============================================================
# AUDITORÍA
# ============================================================

def run_audit(df):

    trades = []

    i = 200

    while i < len(df) - 2:

        row = df.iloc[i]

        if pd.isna(row["atr"]):
            i += 1
            continue

        # ----------------------------------------------------
        # SEÑAL ORIGINAL
        # ----------------------------------------------------

        original_long = (
            row["close"] > row["ema20"]
            and row["ema20"] > row["ema50"]
            and row["ema50"] > row["ema200"]
            and row["close"] > row["open"]
        )

        original_short = (
            row["close"] < row["ema20"]
            and row["ema20"] < row["ema50"]
            and row["ema50"] < row["ema200"]
            and row["close"] < row["open"]
        )

        if not original_long and not original_short:
            i += 1
            continue

        # ----------------------------------------------------
        # CONFIRMACIÓN
        # ----------------------------------------------------

        confirm = df.iloc[i + 1]

        if original_long:

            confirmed = (
                confirm["close"] > confirm["ema20"]
                and confirm["ema20"] > confirm["ema50"]
                and confirm["ema50"] > confirm["ema200"]
                and confirm["close"] > confirm["open"]
            )

            direction = "SHORT"

        else:

            confirmed = (
                confirm["close"] < confirm["ema20"]
                and confirm["ema20"] < confirm["ema50"]
                and confirm["ema50"] < confirm["ema200"]
                and confirm["close"] < confirm["open"]
            )

            direction = "LONG"

        if not confirmed:
            i += 1
            continue

        # ----------------------------------------------------
        # VARIABLES DE ENTRADA
        # ----------------------------------------------------

        atr = confirm["atr"]

        if pd.isna(atr) or atr <= 0:
            i += 1
            continue

        body = abs(confirm["close"] - confirm["open"])
        body_atr = body / atr

        if body_atr < BODY_ATR_MIN:
            i += 1
            continue

        close = confirm["close"]

        distance_ema20 = abs(close - confirm["ema20"]) / atr
        distance_ema50 = abs(close - confirm["ema50"]) / atr
        distance_ema200 = abs(close - confirm["ema200"]) / atr

        ema20_50 = abs(confirm["ema20"] - confirm["ema50"]) / atr
        ema50_200 = abs(confirm["ema50"] - confirm["ema200"]) / atr

        ema20_slope = (
            confirm["ema20"] - df.iloc[i]["ema20"]
        ) / atr

        ema50_slope = (
            confirm["ema50"] - df.iloc[i]["ema50"]
        ) / atr

        entry_index = i + 1

        entry = close

        risk = SL_ATR * atr
        target = TP_ATR * atr

        if direction == "LONG":

            stop = entry - risk
            tp = entry + target

        else:

            stop = entry + risk
            tp = entry - target

        # ----------------------------------------------------
        # BUSCAR SALIDA
        # ----------------------------------------------------

        exit_index = None
        result_r = None

        for j in range(entry_index + 1, len(df)):

            candle = df.iloc[j]

            if direction == "LONG":

                hit_stop = candle["low"] <= stop
                hit_tp = candle["high"] >= tp

            else:

                hit_stop = candle["high"] >= stop
                hit_tp = candle["low"] <= tp

            # STOP FIRST
            if hit_stop and hit_tp:

                result_r = -1.0
                exit_index = j
                break

            if hit_stop:

                result_r = -1.0
                exit_index = j
                break

            if hit_tp:

                result_r = 2.0
                exit_index = j
                break

        if exit_index is None:
            break

        duration_min = (
            df.iloc[exit_index]["datetime"]
            - confirm["datetime"]
        ).total_seconds() / 60

        trades.append({
            "direction": direction,
            "body_atr": body_atr,
            "distance_ema20": distance_ema20,
            "distance_ema50": distance_ema50,
            "distance_ema200": distance_ema200,
            "ema20_50": ema20_50,
            "ema50_200": ema50_200,
            "ema20_slope_atr": ema20_slope,
            "ema50_slope_atr": ema50_slope,
            "duration_min": duration_min,
            "R": result_r
        })

        i = exit_index + 1

    return pd.DataFrame(trades)


# ============================================================
# ESTADÍSTICAS
# ============================================================

def stats(df):

    if len(df) == 0:
        return {
            "ops": 0,
            "wins": 0,
            "win_rate": 0,
            "R": 0,
            "R_medio": 0
        }

    wins = (df["R"] > 0).sum()

    return {
        "ops": len(df),
        "wins": wins,
        "losses": len(df) - wins,
        "win_rate": wins / len(df) * 100,
        "R": df["R"].sum(),
        "R_medio": df["R"].mean()
    }


def print_stats(title, df):

    s = stats(df)

    print()
    print("=" * 70)
    print(title)
    print("=" * 70)

    print(f"OPERACIONES: {s['ops']}")
    print(f"GANADORAS: {s['wins']}")
    print(f"PERDEDORAS: {s['losses']}")
    print(f"WIN RATE: {s['win_rate']:.2f}%")
    print(f"R TOTAL: {s['R']:.2f}")
    print(f"R MEDIO: {s['R_medio']:.3f}")


# ============================================================
# ANÁLISIS
# ============================================================

all_trades = []

for month in range(1, 13):

    end_date = f"{START_YEAR}-{month:02d}-28"

    print(f"\nDescargando {end_date}...")

    df = get_data(end_date)

    if df is None:
        continue

    df = indicators(df)

    trades = run_audit(df)

    if len(trades) > 0:
        all_trades.append(trades)

    if month != 12:
        time.sleep(20)


if not all_trades:
    raise ValueError("No se obtuvieron operaciones.")


trades = pd.concat(all_trades, ignore_index=True)


# ============================================================
# COMPLETO
# ============================================================

print_stats("MUESTRA COMPLETA", trades)


# ============================================================
# DURACIÓN
# ============================================================

trades["duracion_grupo"] = np.where(
    trades["duration_min"] < 30,
    "<30 MIN",
    ">=30 MIN"
)

print()
print("=" * 70)
print("DURACIÓN: <30 MIN VS >=30 MIN")
print("=" * 70)

for name, group in trades.groupby("duracion_grupo"):

    s = stats(group)

    print(
        f"{name:10} | "
        f"OPS {s['ops']:4} | "
        f"WR {s['win_rate']:6.2f}% | "
        f"R {s['R']:7.2f} | "
        f"R/op {s['R_medio']:7.3f}"
    )


# ============================================================
# DIRECCIÓN × DURACIÓN
# ============================================================

print()
print("=" * 70)
print("DIRECCIÓN × DURACIÓN")
print("=" * 70)

for direction in ["LONG", "SHORT"]:

    for duration in ["<30 MIN", ">=30 MIN"]:

        group = trades[
            (trades["direction"] == direction)
            & (trades["duracion_grupo"] == duration)
        ]

        s = stats(group)

        print(
            f"{direction:5} {duration:8} | "
            f"OPS {s['ops']:4} | "
            f"WR {s['win_rate']:6.2f}% | "
            f"R {s['R']:7.2f} | "
            f"R/op {s['R_medio']:7.3f}"
        )


# ============================================================
# SLOPE × DIRECCIÓN
# ============================================================

def slope_group(x):

    if x < -0.10:
        return "< -0.10"

    if x < -0.05:
        return "-0.10 a -0.05"

    if x < 0:
        return "-0.05 a 0"

    if x < 0.05:
        return "0 a 0.05"

    if x < 0.10:
        return "0.05 a 0.10"

    return "> 0.10"


trades["slope_grupo"] = trades["ema20_slope_atr"].apply(slope_group)

print()
print("=" * 70)
print("DIRECCIÓN × EMA20 SLOPE")
print("=" * 70)

for direction in ["LONG", "SHORT"]:

    print()
    print(f"--- {direction} ---")

    subset = trades[trades["direction"] == direction]

    for group_name, group in subset.groupby("slope_grupo", sort=False):

        s = stats(group)

        print(
            f"{group_name:14} | "
            f"OPS {s['ops']:4} | "
            f"WR {s['win_rate']:6.2f}% | "
            f"R {s['R']:7.2f} | "
            f"R/op {s['R_medio']:7.3f}"
        )


# ============================================================
# SLOPE × DURACIÓN
# ============================================================

print()
print("=" * 70)
print("EMA20 SLOPE × DURACIÓN")
print("=" * 70)

for group_name, group in trades.groupby("slope_grupo", sort=False):

    short = group[group["duracion_grupo"] == "<30 MIN"]
    long = group[group["duracion_grupo"] == ">=30 MIN"]

    s1 = stats(short)
    s2 = stats(long)

    print()
    print(group_name)

    print(
        f"  <30 min  : "
        f"{s1['ops']:4} ops | "
        f"{s1['win_rate']:6.2f}% | "
        f"{s1['R']:7.2f}R"
    )

    print(
        f"  >=30 min : "
        f"{s2['ops']:4} ops | "
        f"{s2['win_rate']:6.2f}% | "
        f"{s2['R']:7.2f}R"
    )


# ============================================================
# BODY > 1.5 ATR
# ============================================================

print()
print("=" * 70)
print("BODY > 1.5 ATR")
print("=" * 70)

for condition, group in [
    ("BODY <= 1.5 ATR", trades[trades["body_atr"] <= 1.5]),
    ("BODY > 1.5 ATR", trades[trades["body_atr"] > 1.5])
]:

    s = stats(group)

    print(
        f"{condition:18} | "
        f"OPS {s['ops']:4} | "
        f"WR {s['win_rate']:6.2f}% | "
        f"R {s['R']:7.2f} | "
        f"R/op {s['R_medio']:7.3f}"
    )


# ============================================================
# EMA20-EMA50
# ============================================================

print()
print("=" * 70)
print("EMA20-EMA50: <= 1 ATR VS > 1 ATR")
print("=" * 70)

for condition, group in [
    ("<= 1 ATR", trades[trades["ema20_50"] <= 1]),
    ("> 1 ATR", trades[trades["ema20_50"] > 1])
]:

    s = stats(group)

    print(
        f"{condition:10} | "
        f"OPS {s['ops']:4} | "
        f"WR {s['win_rate']:6.2f}% | "
        f"R {s['R']:7.2f} | "
        f"R/op {s['R_medio']:7.3f}"
    )


print()
print("=" * 70)
print("FIN DE LA AUDITORÍA")
print("=" * 70)
