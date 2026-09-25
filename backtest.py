import os
import time
import requests
import pandas as pd
import numpy as np

# ============================================================
# CONFIGURACIÓN
# ============================================================

API_KEY = os.getenv("TWELVE_DATA_API_KEY")

if not API_KEY:
    raise ValueError("ERROR: No se encontró TWELVE_DATA_API_KEY")

SYMBOL = "EUR/USD"
INTERVAL = "5min"

ATR_PERIOD = 14

EMA_FAST = 20
EMA_MID = 50
EMA_SLOW = 200

SL_ATR = 1.5
TP_ATR = 3.0

BODY_ATR_FILTER = 0.25

PERIODS = [
    ("ENERO 2025", "2025-01-31 23:55:00"),
    ("FEBRERO 2025", "2025-02-28 23:55:00"),
    ("MARZO 2025", "2025-03-31 23:55:00"),
    ("ABRIL 2025", "2025-04-30 23:55:00"),
    ("MAYO 2025", "2025-05-31 23:55:00"),
    ("JUNIO 2025", "2025-06-30 23:55:00"),
    ("JULIO 2025", "2025-07-31 23:55:00"),
    ("AGOSTO 2025", "2025-08-31 23:55:00"),
    ("SEPTIEMBRE 2025", "2025-09-30 23:55:00"),
    ("OCTUBRE 2025", "2025-10-31 23:55:00"),
    ("NOVIEMBRE 2025", "2025-11-30 23:55:00"),
    ("DICIEMBRE 2025", "2025-12-31 23:55:00"),
]


# ============================================================
# DATOS
# ============================================================

def get_data(end_date):

    url = "https://api.twelvedata.com/time_series"

    params = {
        "symbol": SYMBOL,
        "interval": INTERVAL,
        "outputsize": 5000,
        "end_date": end_date,
        "apikey": API_KEY,
        "format": "JSON",
    }

    for attempt in range(5):

        try:

            response = requests.get(
                url,
                params=params,
                timeout=30
            )

            if response.status_code == 429:

                wait = 30 * (attempt + 1)

                print(
                    f"Rate limit 429. "
                    f"Esperando {wait} segundos..."
                )

                time.sleep(wait)
                continue

            response.raise_for_status()

            data = response.json()

            if "values" not in data:

                print(data)

                raise ValueError(
                    "Twelve Data no devolvió velas."
                )

            df = pd.DataFrame(data["values"])

            df["datetime"] = pd.to_datetime(
                df["datetime"]
            )

            for col in [
                "open",
                "high",
                "low",
                "close"
            ]:
                df[col] = pd.to_numeric(
                    df[col]
                )

            df = (
                df
                .sort_values("datetime")
                .reset_index(drop=True)
            )

            return df

        except requests.RequestException as e:

            if attempt == 4:
                raise

            wait = 10 * (attempt + 1)

            print(f"Error: {e}")
            print(
                f"Reintentando en {wait} segundos..."
            )

            time.sleep(wait)

    raise RuntimeError(
        "No se pudieron descargar los datos."
    )


# ============================================================
# INDICADORES
# ============================================================

def calculate_indicators(df):

    df = df.copy()

    df["ema20"] = df["close"].ewm(
        span=EMA_FAST,
        adjust=False
    ).mean()

    df["ema50"] = df["close"].ewm(
        span=EMA_MID,
        adjust=False
    ).mean()

    df["ema200"] = df["close"].ewm(
        span=EMA_SLOW,
        adjust=False
    ).mean()

    previous_close = df["close"].shift(1)

    tr1 = df["high"] - df["low"]

    tr2 = abs(
        df["high"] - previous_close
    )

    tr3 = abs(
        df["low"] - previous_close
    )

    df["tr"] = pd.concat(
        [tr1, tr2, tr3],
        axis=1
    ).max(axis=1)

    df["atr"] = (
        df["tr"]
        .rolling(ATR_PERIOD)
        .mean()
    )

    # Pendientes simples de EMA
    df["ema20_slope"] = (
        df["ema20"] - df["ema20"].shift(1)
    )

    df["ema50_slope"] = (
        df["ema50"] - df["ema50"].shift(1)
    )

    return df


# ============================================================
# SEÑAL ORIGINAL
# ============================================================

def get_original_signal(row):

    long_conditions = [
        row["close"] > row["ema20"],
        row["ema20"] > row["ema50"],
        row["ema50"] > row["ema200"],
        row["close"] > row["open"],
    ]

    short_conditions = [
        row["close"] < row["ema20"],
        row["ema20"] < row["ema50"],
        row["ema50"] < row["ema200"],
        row["close"] < row["open"],
    ]

    if all(long_conditions):
        return "LONG"

    if all(short_conditions):
        return "SHORT"

    return None


# ============================================================
# GENERAR CANDIDATAS
# ============================================================

def generate_candidates(df):

    candidates = []

    for i in range(len(df) - 1):

        row1 = df.iloc[i]
        row2 = df.iloc[i + 1]

        signal1 = get_original_signal(row1)
        signal2 = get_original_signal(row2)

        if signal1 is None:
            continue

        # Confirmación
        if signal1 != signal2:
            continue

        atr = row2["atr"]

        if pd.isna(atr) or atr <= 0:
            continue

        body = abs(
            row2["close"] - row2["open"]
        )

        body_atr = body / atr

        if body_atr < BODY_ATR_FILTER:
            continue

        # Inversión de la señal
        if signal1 == "LONG":
            direction = "SHORT"
        else:
            direction = "LONG"

        candidates.append({
            "entry_index": i + 1,
            "direction": direction,
            "atr": atr,
        })

    return candidates


# ============================================================
# SIMULACIÓN + CARACTERÍSTICAS
# ============================================================

def simulate(df, candidates):

    trades = []

    next_available_index = 0

    for candidate in candidates:

        entry_index = candidate["entry_index"]

        if entry_index < next_available_index:
            continue

        row = df.iloc[entry_index]

        direction = candidate["direction"]
        atr = candidate["atr"]

        entry_price = row["close"]

        # --------------------------------------------
        # CARACTERÍSTICAS DE LA ENTRADA
        # --------------------------------------------

        body = abs(
            row["close"] - row["open"]
        )

        body_atr = body / atr

        distance_ema20 = (
            abs(row["close"] - row["ema20"])
            / atr
        )

        distance_ema50 = (
            abs(row["close"] - row["ema50"])
            / atr
        )

        distance_ema200 = (
            abs(row["close"] - row["ema200"])
            / atr
        )

        ema20_50 = (
            abs(row["ema20"] - row["ema50"])
            / atr
        )

        ema50_200 = (
            abs(row["ema50"] - row["ema200"])
            / atr
        )

        ema20_slope_atr = (
            row["ema20_slope"] / atr
        )

        ema50_slope_atr = (
            row["ema50_slope"] / atr
        )

        hour = row["datetime"].hour

        # --------------------------------------------
        # SL / TP
        # --------------------------------------------

        if direction == "LONG":

            stop = (
                entry_price
                - SL_ATR * atr
            )

            target = (
                entry_price
                + TP_ATR * atr
            )

        else:

            stop = (
                entry_price
                + SL_ATR * atr
            )

            target = (
                entry_price
                - TP_ATR * atr
            )

        result = None
        exit_index = None

        # --------------------------------------------
        # BUSCAR SALIDA
        # --------------------------------------------

        for j in range(
            entry_index + 1,
            len(df)
        ):

            high = df.iloc[j]["high"]
            low = df.iloc[j]["low"]

            if direction == "LONG":

                # STOP primero si toca ambos
                if low <= stop:

                    result = -1.0
                    exit_index = j
                    break

                if high >= target:

                    result = 2.0
                    exit_index = j
                    break

            else:

                if high >= stop:

                    result = -1.0
                    exit_index = j
                    break

                if low <= target:

                    result = 2.0
                    exit_index = j
                    break

        if result is None:
            continue

        duration = (
            df.iloc[exit_index]["datetime"]
            - df.iloc[entry_index]["datetime"]
        ).total_seconds() / 60

        trades.append({

            "direction": direction,

            "result_R": result,

            "duration_min": duration,

            "body_atr": body_atr,

            "distance_ema20": distance_ema20,

            "distance_ema50": distance_ema50,

            "distance_ema200": distance_ema200,

            "ema20_50": ema20_50,

            "ema50_200": ema50_200,

            "ema20_slope_atr": ema20_slope_atr,

            "ema50_slope_atr": ema50_slope_atr,

            "hour": hour,

            "atr": atr,

            "entry_time": row["datetime"],

        })

        next_available_index = exit_index + 1

    return trades


# ============================================================
# ESTADÍSTICAS
# ============================================================

def stats(results):

    results = list(results)

    if not results:
        return 0, 0, 0, 0

    wins = sum(
        r > 0 for r in results
    )

    total = sum(results)

    win_rate = (
        wins / len(results) * 100
    )

    avg = (
        total / len(results)
    )

    return (
        len(results),
        win_rate,
        total,
        avg
    )


# ============================================================
# ANÁLISIS DE UNA VARIABLE
# ============================================================

def analyze_bins(df, column, bins, labels):

    print()
    print("=" * 70)
    print(f"ANÁLISIS: {column}")
    print("=" * 70)

    temp = df.copy()

    temp["grupo"] = pd.cut(
        temp[column],
        bins=bins,
        labels=labels,
        include_lowest=True
    )

    rows = []

    for group in labels:

        subset = temp[
            temp["grupo"] == group
        ]

        if len(subset) == 0:
            continue

        results = subset["result_R"].tolist()

        n, wr, total_R, avg_R = stats(
            results
        )

        rows.append({
            "rango": str(group),
            "ops": n,
            "win_rate": wr,
            "R": total_R,
            "R_medio": avg_R,
        })

    result_df = pd.DataFrame(rows)

    if not result_df.empty:
        print(
            result_df.to_string(
                index=False,
                formatters={
                    "win_rate": "{:.2f}".format,
                    "R": "{:.2f}".format,
                    "R_medio": "{:.3f}".format,
                }
            )
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("AUDITORÍA DE OPERACIONES - 2025")
    print("=" * 70)

    print()
    print("ESTRATEGIA CONGELADA")
    print("---------------------")
    print("EMA 20 / 50 / 200")
    print("4/4")
    print("INVERSIÓN")
    print("CONFIRMACIÓN 1 VELA")
    print("CUERPO >= 0.25 ATR")
    print("SL 1.5 ATR")
    print("TP 3 ATR")
    print("UNA OPERACIÓN A LA VEZ")

    all_trades = []

    # ========================================================
    # RECORRER TODO 2025
    # ========================================================

    for index, (name, end_date) in enumerate(PERIODS):

        print()
        print("=" * 50)
        print(name)
        print("=" * 50)

        df = get_data(end_date)

        df = calculate_indicators(df)

        candidates = generate_candidates(df)

        trades = simulate(
            df,
            candidates
        )

        all_trades.extend(trades)

        print(
            f"Candidatas: {len(candidates)}"
        )

        print(
            f"Operaciones: {len(trades)}"
        )

        if index < len(PERIODS) - 1:

            print(
                "Esperando 20 segundos..."
            )

            time.sleep(20)

    # ========================================================
    # DATAFRAME FINAL
    # ========================================================

    trades_df = pd.DataFrame(
        all_trades
    )

    print()
    print("=" * 70)
    print("MUESTRA COMPLETA")
    print("=" * 70)

    print(
        f"OPERACIONES: {len(trades_df)}"
    )

    wins = (
        trades_df["result_R"] > 0
    ).sum()

    losses = (
        trades_df["result_R"] < 0
    ).sum()

    print(
        f"GANADORAS: {wins}"
    )

    print(
        f"PERDEDORAS: {losses}"
    )

    print(
        f"WIN RATE: "
        f"{wins / len(trades_df) * 100:.2f}%"
    )

    print(
        f"R TOTAL: "
        f"{trades_df['result_R'].sum():.2f}"
    )

    print(
        f"R MEDIO: "
        f"{trades_df['result_R'].mean():.3f}"
    )

    # ========================================================
    # GANADORAS VS PERDEDORAS
    # ========================================================

    winners = trades_df[
        trades_df["result_R"] > 0
    ]

    losers = trades_df[
        trades_df["result_R"] < 0
    ]

    variables = [
        "body_atr",
        "distance_ema20",
        "distance_ema50",
        "distance_ema200",
        "ema20_50",
        "ema50_200",
        "ema20_slope_atr",
        "ema50_slope_atr",
        "duration_min",
        "atr",
    ]

    print()
    print("=" * 70)
    print("GANADORAS VS PERDEDORAS")
    print("=" * 70)

    comparison = []

    for variable in variables:

        comparison.append({
            "variable": variable,
            "ganadoras": winners[variable].mean(),
            "perdedoras": losers[variable].mean(),
        })

    comparison_df = pd.DataFrame(
        comparison
    )

    print(
        comparison_df.to_string(
            index=False,
            formatters={
                "ganadoras": "{:.4f}".format,
                "perdedoras": "{:.4f}".format,
            }
        )
    )

    # ========================================================
    # LONG VS SHORT
    # ========================================================

    print()
    print("=" * 70)
    print("LONG VS SHORT")
    print("=" * 70)

    direction_rows = []

    for direction in ["LONG", "SHORT"]:

        subset = trades_df[
            trades_df["direction"] == direction
        ]

        results = subset[
            "result_R"
        ].tolist()

        n, wr, total_R, avg_R = stats(
            results
        )

        direction_rows.append({
            "direccion": direction,
            "ops": n,
            "win_rate": wr,
            "R": total_R,
            "R_medio": avg_R,
        })

    direction_df = pd.DataFrame(
        direction_rows
    )

    print(
        direction_df.to_string(
            index=False,
            formatters={
                "win_rate": "{:.2f}".format,
                "R": "{:.2f}".format,
                "R_medio": "{:.3f}".format,
            }
        )
    )

    # ========================================================
    # HORA
    # ========================================================

    print()
    print("=" * 70)
    print("RESULTADOS POR HORA UTC")
    print("=" * 70)

    hour_rows = []

    for hour in sorted(
        trades_df["hour"].unique()
    ):

        subset = trades_df[
            trades_df["hour"] == hour
        ]

        results = subset[
            "result_R"
        ].tolist()

        n, wr, total_R, avg_R = stats(
            results
        )

        hour_rows.append({
            "hora": hour,
            "ops": n,
            "win_rate": wr,
            "R": total_R,
            "R_medio": avg_R,
        })

    hour_df = pd.DataFrame(
        hour_rows
    )

    print(
        hour_df.to_string(
            index=False,
            formatters={
                "win_rate": "{:.2f}".format,
                "R": "{:.2f}".format,
                "R_medio": "{:.3f}".format,
            }
        )
    )

    # ========================================================
    # RANGOS
    # ========================================================

    analyze_bins(
        trades_df,
        "body_atr",
        [0.25, 0.50, 0.75, 1.00, 1.50, 3.00, np.inf],
        [
            "0.25-0.50",
            "0.50-0.75",
            "0.75-1.00",
            "1.00-1.50",
            "1.50-3.00",
            ">3.00",
        ]
    )

    analyze_bins(
        trades_df,
        "distance_ema20",
        [0, 0.25, 0.50, 0.75, 1.00, 1.50, 2.00, np.inf],
        [
            "0-0.25",
            "0.25-0.50",
            "0.50-0.75",
            "0.75-1.00",
            "1.00-1.50",
            "1.50-2.00",
            ">2.00",
        ]
    )

    analyze_bins(
        trades_df,
        "ema20_50",
        [0, 0.10, 0.25, 0.50, 0.75, 1.00, np.inf],
        [
            "0-0.10",
            "0.10-0.25",
            "0.25-0.50",
            "0.50-0.75",
            "0.75-1.00",
            ">1.00",
        ]
    )

    analyze_bins(
        trades_df,
        "duration_min",
        [0, 30, 60, 120, 180, 360, np.inf],
        [
            "<30",
            "30-60",
            "60-120",
            "120-180",
            "180-360",
            ">360",
        ]
    )

    analyze_bins(
        trades_df,
        "ema20_slope_atr",
        [-np.inf, -0.10, -0.05, -0.02, 0, 0.02, 0.05, 0.10, np.inf],
        [
            "<-0.10",
            "-0.10--0.05",
            "-0.05--0.02",
            "-0.02-0",
            "0-0.02",
            "0.02-0.05",
            "0.05-0.10",
            ">0.10",
        ]
    )

    print()
    print("=" * 70)
    print("FIN DE LA AUDITORÍA")
    print("=" * 70)


if __name__ == "__main__":
    main()
