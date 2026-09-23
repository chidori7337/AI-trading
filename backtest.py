"""
VALIDACIÓN FUERA DE MUESTRA — versión congelada 0,25 ATR
=========================================================

Este script NO cambia ni una coma de la lógica de señales, indicadores,
SL/TP o gestión de operaciones que ya usasteis para elegir la versión
de 0,25 ATR. Lo único que cambia son los períodos: en vez de 2026
(datos usados para construir y elegir el filtro), aquí se prueba sobre
octubre, noviembre y diciembre de 2025, que el sistema nunca ha visto.

Si el resultado también es razonable aquí, es una señal mucho más
fuerte que cualquier mejora conseguida ajustando parámetros dentro de
la misma muestra.

Requisitos:
    pip install pandas requests
    export TWELVE_DATA_API_KEY="tu_api_key"
"""

import time
import os
import requests
import pandas as pd

API_KEY = os.getenv("TWELVE_DATA_API_KEY")

# Períodos fuera de muestra: el sistema nunca se ha ajustado con estos datos.
PERIODS = [
    ("OCTUBRE 2025", "2025-10-31 23:55:00"),
    ("NOVIEMBRE 2025", "2025-11-30 23:55:00"),
    ("DICIEMBRE 2025", "2025-12-31 23:55:00"),
]

# Se mantiene SIN FILTRO como referencia, y se congela 0.25 ATR
# (la versión elegida) en vez de 0.5 ATR.
FILTERS = {
    "SIN FILTRO": 0.0,
    "0.25 ATR": 0.25,
}


def download_data(end_date):
    url = "https://api.twelvedata.com/time_series"

    params = {
        "symbol": "EUR/USD",
        "interval": "5min",
        "outputsize": 5000,
        "end_date": end_date,
        "apikey": API_KEY,
    }

    max_retries = 5

    for attempt in range(max_retries):

        try:
            response = requests.get(url, params=params, timeout=30)
            data = response.json()

            if "values" not in data:

                message = data.get("message", "Error desconocido")
                print(f"Error Twelve Data: {message}")

                if response.status_code == 429 or "limit" in message.lower():
                    wait_time = 20 * (attempt + 1)
                    print(f"Rate limit. Esperando {wait_time} segundos...")
                    time.sleep(wait_time)
                    continue

                return None

            df = pd.DataFrame(data["values"])
            df["datetime"] = pd.to_datetime(df["datetime"])

            numeric_cols = ["open", "high", "low", "close"]
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            df = df.sort_values("datetime").reset_index(drop=True)

            df.rename(
                columns={
                    "open": "Open",
                    "high": "High",
                    "low": "Low",
                    "close": "Close",
                },
                inplace=True,
            )

            return df

        except Exception as e:
            print(f"Error descargando datos: {e}")

            if attempt < max_retries - 1:
                wait_time = 10 * (attempt + 1)
                print(f"Esperando {wait_time} segundos...")
                time.sleep(wait_time)

    return None


def calculate_indicators(df):
    df = df.copy()

    df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["EMA50"] = df["Close"].ewm(span=50, adjust=False).mean()
    df["EMA200"] = df["Close"].ewm(span=200, adjust=False).mean()

    df["PrevClose"] = df["Close"].shift(1)

    df["TR"] = df.apply(
        lambda row: max(
            row["High"] - row["Low"],
            abs(row["High"] - row["PrevClose"]) if pd.notna(row["PrevClose"]) else 0,
            abs(row["Low"] - row["PrevClose"]) if pd.notna(row["PrevClose"]) else 0,
        ),
        axis=1,
    )

    df["ATR"] = df["TR"].rolling(14).mean()
    df["Body"] = abs(df["Close"] - df["Open"])

    return df


def get_original_signal(df, index):
    current = df.iloc[index]

    if pd.isna(current["EMA200"]) or pd.isna(current["ATR"]):
        return None

    long_score = 0
    short_score = 0

    if current["Close"] > current["EMA20"]:
        long_score += 1
    if current["Close"] < current["EMA20"]:
        short_score += 1

    if current["EMA20"] > current["EMA50"]:
        long_score += 1
    if current["EMA20"] < current["EMA50"]:
        short_score += 1

    if current["EMA50"] > current["EMA200"]:
        long_score += 1
    if current["EMA50"] < current["EMA200"]:
        short_score += 1

    if current["Close"] > current["Open"]:
        long_score += 1
    if current["Close"] < current["Open"]:
        short_score += 1

    if long_score >= 4:
        return "LONG"
    if short_score >= 4:
        return "SHORT"

    return None


def generate_candidate_signals(df):
    candidates = []
    i = 200

    while i < len(df) - 2:

        original_signal = get_original_signal(df, i)

        if original_signal is None:
            i += 1
            continue

        confirmation_signal = get_original_signal(df, i + 1)

        if confirmation_signal != original_signal:
            i += 1
            continue

        confirmation_candle = df.iloc[i + 1]
        body = confirmation_candle["Body"]
        atr = confirmation_candle["ATR"]

        if pd.isna(body) or pd.isna(atr) or atr <= 0:
            i += 1
            continue

        candidates.append(
            {
                "signal_index": i,
                "entry_index": i + 1,
                "original_signal": original_signal,
                "body": body,
                "atr": atr,
                "body_atr_ratio": body / atr,
            }
        )

        i += 1

    return candidates


def calculate_max_drawdown(results):
    equity = 0
    peak = 0
    max_drawdown = 0

    for r in results:
        equity += r
        if equity > peak:
            peak = equity
        drawdown = equity - peak
        if drawdown < max_drawdown:
            max_drawdown = drawdown

    return max_drawdown


def calculate_max_losing_streak(results):
    current = 0
    maximum = 0

    for r in results:
        if r < 0:
            current += 1
            if current > maximum:
                maximum = current
        else:
            current = 0

    return maximum


def calculate_profit_factor(results):
    gross_profit = sum(r for r in results if r > 0)
    gross_loss = abs(sum(r for r in results if r < 0))

    if gross_loss == 0:
        return float("inf")

    return gross_profit / gross_loss


def duration_bucket(minutes):
    if minutes < 30:
        return "< 30 min"
    if minutes < 60:
        return "30-60 min"
    if minutes < 120:
        return "60-120 min"
    if minutes < 180:
        return "120-180 min"
    if minutes < 360:
        return "180-360 min"
    return "> 360 min"


def simulate_filter(df, candidates, threshold):
    trades = []
    skipped_by_filter = 0
    next_available_index = 0

    for candidate in candidates:

        entry_index = candidate["entry_index"]

        if entry_index < next_available_index:
            continue

        body_atr_ratio = candidate["body_atr_ratio"]

        if body_atr_ratio < threshold:
            skipped_by_filter += 1
            continue

        original_signal = candidate["original_signal"]

        # Inversión de señal (igual que en la versión original)
        if original_signal == "LONG":
            signal = "SHORT"
        else:
            signal = "LONG"

        current = df.iloc[entry_index]
        entry = current["Close"]
        atr = current["ATR"]

        if pd.isna(atr) or atr <= 0:
            continue

        risk_distance = 1.5 * atr

        if signal == "LONG":
            stop = entry - risk_distance
            target = entry + 3 * atr
        else:
            stop = entry + risk_distance
            target = entry - 3 * atr

        result = None
        R = None
        exit_index = None

        j = entry_index + 1

        while j < len(df):

            future = df.iloc[j]

            if signal == "LONG":
                hit_stop = future["Low"] <= stop
                hit_target = future["High"] >= target
            else:
                hit_stop = future["High"] >= stop
                hit_target = future["Low"] <= target

            if hit_stop:
                result = "LOSS"
                R = -1
                exit_index = j
                break

            if hit_target:
                result = "WIN"
                R = 2
                exit_index = j
                break

            j += 1

        if result is None:
            continue

        entry_time = current["datetime"]
        exit_time = df.iloc[exit_index]["datetime"]
        duration = (exit_time - entry_time).total_seconds() / 60

        trades.append(
            {
                "signal": signal,
                "result": result,
                "R": R,
                "duration": duration,
                "bucket": duration_bucket(duration),
                "body_atr_ratio": body_atr_ratio,
            }
        )

        next_available_index = exit_index + 1

    results = [trade["R"] for trade in trades]

    total_ops = len(trades)
    wins = sum(1 for r in results if r > 0)
    losses = sum(1 for r in results if r < 0)
    win_rate = wins / total_ops * 100 if total_ops > 0 else 0
    total_r = sum(results)
    avg_r = total_r / total_ops if total_ops > 0 else 0
    max_dd = calculate_max_drawdown(results)
    losing_streak = calculate_max_losing_streak(results)
    profit_factor = calculate_profit_factor(results)

    durations = [trade["duration"] for trade in trades]
    avg_duration = sum(durations) / len(durations) if durations else 0

    sorted_durations = sorted(durations)
    median_duration = (
        sorted_durations[len(sorted_durations) // 2] if sorted_durations else 0
    )

    return {
        "trades": trades,
        "operations": total_ops,
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,
        "R": total_r,
        "avg_R": avg_r,
        "drawdown": max_dd,
        "losing_streak": losing_streak,
        "profit_factor": profit_factor,
        "avg_duration": avg_duration,
        "median_duration": median_duration,
        "skipped_by_filter": skipped_by_filter,
    }


def print_test_result(name, result, total_candidates):
    print(f"\n{name}")
    print("-" * 35)
    print(f"OPERACIONES: {result['operations']}")
    print(f"GANADORAS: {result['wins']}")
    print(f"PERDEDORAS: {result['losses']}")
    print(f"WIN RATE: {result['win_rate']:.2f}%")
    print(f"R TOTAL: {result['R']:.2f}")
    print(f"R MEDIO: {result['avg_R']:.3f}")
    print(f"DRAWDOWN MÁXIMO: {result['drawdown']:.2f} R")
    print(f"PEOR RACHA: {result['losing_streak']}")

    if result["profit_factor"] == float("inf"):
        print("PROFIT FACTOR: inf")
    else:
        print(f"PROFIT FACTOR: {result['profit_factor']:.3f}")

    print(f"DURACIÓN MEDIA: {result['avg_duration']:.1f} min")
    print(f"DURACIÓN MEDIANA: {result['median_duration']:.1f} min")

    if total_candidates > 0:
        skipped_percentage = result["skipped_by_filter"] / total_candidates * 100
    else:
        skipped_percentage = 0

    print(f"DESCARTADAS POR FILTRO: {result['skipped_by_filter']}")
    print(f"% CANDIDATAS DESCARTADAS: {skipped_percentage:.2f}%")


def main():

    combined = {name: [] for name in FILTERS}
    period_results = []

    for period_name, end_date in PERIODS:

        print("\n")
        print("#" * 40)
        print(period_name)
        print("#" * 40)

        df = download_data(end_date)

        if df is None or df.empty:
            print("No se pudieron descargar los datos.")
            continue

        print(f"\nFINAL DE DATOS: {df['datetime'].iloc[-1]}")
        print(f"VELAS OBTENIDAS: {len(df)}")
        print(f"DESDE: {df['datetime'].iloc[0]}")
        print(f"HASTA: {df['datetime'].iloc[-1]}")

        df = calculate_indicators(df)
        print("\nINDICADORES CALCULADOS CORRECTAMENTE")

        candidates = generate_candidate_signals(df)
        total_candidates = len(candidates)

        print(f"\nCANDIDATAS ORIGINALES: {total_candidates}")
        print("\nLas dos pruebas usan exactamente estas mismas candidatas.")

        results_for_period = {}

        for name, threshold in FILTERS.items():
            result = simulate_filter(df, candidates, threshold)
            results_for_period[name] = result
            combined[name].extend(result["trades"])
            print_test_result(name, result, total_candidates)

        a = results_for_period["SIN FILTRO"]
        b = results_for_period["0.25 ATR"]

        print("\n")
        print("=" * 40)
        print("CAMBIO DEL FILTRO 0.25 ATR (versión congelada)")
        print("=" * 40)

        print(f"\nCAMBIO OPERACIONES: {b['operations'] - a['operations']:+d}")
        print(f"CAMBIO WIN RATE: {b['win_rate'] - a['win_rate']:+.2f} puntos")
        print(f"CAMBIO R: {b['R'] - a['R']:+.2f}")
        print(f"CAMBIO R MEDIO: {b['avg_R'] - a['avg_R']:+.3f}")
        print(f"CAMBIO DRAWDOWN: {b['drawdown'] - a['drawdown']:+.2f} R")
        print(f"CAMBIO RACHA: {b['losing_streak'] - a['losing_streak']:+d}")
        print(f"CAMBIO PROFIT FACTOR: {b['profit_factor'] - a['profit_factor']:+.3f}")

        period_results.append(
            {
                "periodo": period_name,
                "candidatas": total_candidates,
                "ops_sin_filtro": a["operations"],
                "R_sin_filtro": a["R"],
                "DD_sin_filtro": a["drawdown"],
                "ops_0_25": b["operations"],
                "R_0_25": b["R"],
                "DD_0_25": b["drawdown"],
                "cambio_R": b["R"] - a["R"],
            }
        )

        print("\nEsperando 20 segundos...")
        time.sleep(20)

    print("\n")
    print("=" * 50)
    print("AUDITORÍA COMBINADA OCT-DIC 2025 (fuera de muestra)")
    print("=" * 50)

    combined_results = {}

    for name, trades in combined.items():

        results = [trade["R"] for trade in trades]
        total_ops = len(results)
        wins = sum(1 for r in results if r > 0)
        losses = sum(1 for r in results if r < 0)
        total_r = sum(results)
        avg_r = total_r / total_ops if total_ops > 0 else 0
        win_rate = wins / total_ops * 100 if total_ops > 0 else 0
        max_dd = calculate_max_drawdown(results)
        losing_streak = calculate_max_losing_streak(results)
        profit_factor = calculate_profit_factor(results)

        durations = [trade["duration"] for trade in trades]
        avg_duration = sum(durations) / len(durations) if durations else 0

        combined_results[name] = {
            "operations": total_ops,
            "wins": wins,
            "losses": losses,
            "win_rate": win_rate,
            "R": total_r,
            "avg_R": avg_r,
            "drawdown": max_dd,
            "losing_streak": losing_streak,
            "profit_factor": profit_factor,
            "avg_duration": avg_duration,
        }

        print("\n")
        print(name)
        print("-" * 35)
        print(f"OPERACIONES: {total_ops}")
        print(f"GANADORAS: {wins}")
        print(f"PERDEDORAS: {losses}")
        print(f"WIN RATE: {win_rate:.2f}%")
        print(f"R TOTAL: {total_r:.2f}")
        print(f"R MEDIO: {avg_r:.3f}")
        print(f"DRAWDOWN MÁXIMO: {max_dd:.2f} R")
        print(f"PEOR RACHA: {losing_streak}")

        if profit_factor == float("inf"):
            print("PROFIT FACTOR: inf")
        else:
            print(f"PROFIT FACTOR: {profit_factor:.3f}")

        print(f"DURACIÓN MEDIA: {avg_duration:.1f} min")

    a = combined_results["SIN FILTRO"]
    b = combined_results["0.25 ATR"]

    print("\n")
    print("=" * 50)
    print("COMPARACIÓN COMBINADA (fuera de muestra)")
    print("=" * 50)

    print(f"\nCAMBIO OPERACIONES: {b['operations'] - a['operations']:+d}")
    print(f"CAMBIO WIN RATE: {b['win_rate'] - a['win_rate']:+.2f} puntos")
    print(f"CAMBIO R TOTAL: {b['R'] - a['R']:+.2f} R")
    print(f"CAMBIO R MEDIO: {b['avg_R'] - a['avg_R']:+.3f}")
    print(f"CAMBIO DRAWDOWN: {b['drawdown'] - a['drawdown']:+.2f} R")
    print(f"CAMBIO PEOR RACHA: {b['losing_streak'] - a['losing_streak']:+d}")
    print(f"CAMBIO PROFIT FACTOR: {b['profit_factor'] - a['profit_factor']:+.3f}")

    print("\n")
    print("=" * 50)
    print("RESUMEN POR PERÍODO")
    print("=" * 50)

    summary_df = pd.DataFrame(period_results)

    if not summary_df.empty:
        print(summary_df.to_string(index=False))

    print("\n")
    print("=" * 50)
    print("LECTURA — ¿SOBREVIVE FUERA DE MUESTRA?")
    print("=" * 50)

    print()

    if b["R"] > 0:
        print(
            f"La versión 0,25 ATR es rentable fuera de muestra: "
            f"+{b['R']:.2f} R en oct-dic 2025."
        )
    else:
        print(
            f"La versión 0,25 ATR pierde dinero fuera de muestra: "
            f"{b['R']:.2f} R en oct-dic 2025. Esto es una señal de "
            f"alerta importante sobre sobreajuste."
        )

    meses_positivos = sum(1 for p in period_results if p["R_0_25"] > 0)
    print(
        f"Meses positivos con 0,25 ATR: {meses_positivos}/{len(period_results)}"
    )

    print("\n")
    print("=" * 50)
    print("FIN DE LA VALIDACIÓN FUERA DE MUESTRA")
    print("=" * 50)


if __name__ == "__main__":
    main()
