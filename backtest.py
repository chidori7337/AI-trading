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

# Enero - Septiembre 2025
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
]

# ============================================================
# DESCARGAR DATOS
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

    max_retries = 5

    for attempt in range(max_retries):
        try:
            response = requests.get(url, params=params, timeout=30)

            if response.status_code == 429:
                wait = 30 * (attempt + 1)
                print(f"Rate limit (429). Esperando {wait} segundos...")
                time.sleep(wait)
                continue

            response.raise_for_status()

            data = response.json()

            if "values" not in data:
                print("Respuesta de Twelve Data:")
                print(data)
                raise ValueError("No se recibieron velas.")

            df = pd.DataFrame(data["values"])

            df["datetime"] = pd.to_datetime(df["datetime"])

            for col in ["open", "high", "low", "close"]:
                df[col] = pd.to_numeric(df[col])

            df = df.sort_values("datetime").reset_index(drop=True)

            return df

        except requests.RequestException as e:
            if attempt == max_retries - 1:
                raise

            wait = 10 * (attempt + 1)
            print(f"Error de conexión: {e}")
            print(f"Reintentando en {wait} segundos...")
            time.sleep(wait)

    raise RuntimeError("No se pudieron obtener los datos.")


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
    tr2 = abs(df["high"] - previous_close)
    tr3 = abs(df["low"] - previous_close)

    df["tr"] = pd.concat(
        [tr1, tr2, tr3],
        axis=1
    ).max(axis=1)

    # MISMA FÓRMULA UTILIZADA EN LA PRUEBA A/B
    df["atr"] = df["tr"].rolling(
        ATR_PERIOD
    ).mean()

    return df


# ============================================================
# SEÑAL ORIGINAL 4/4
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

        # CONFIRMACIÓN:
        # La misma señal debe mantenerse en la siguiente vela
        if signal1 != signal2:
            continue

        atr = row2["atr"]

        if pd.isna(atr) or atr <= 0:
            continue

        # CUERPO DE LA VELA DE CONFIRMACIÓN
        body = abs(row2["close"] - row2["open"])

        body_ratio = body / atr

        if body_ratio < BODY_ATR_FILTER:
            continue

        # ====================================================
        # INVERSIÓN DE LA SEÑAL
        # ====================================================

        if signal1 == "LONG":
            actual_direction = "SHORT"
        else:
            actual_direction = "LONG"

        candidates.append({
            "signal_index": i,
            "entry_index": i + 1,
            "direction": actual_direction,
            "atr": atr,
        })

    return candidates


# ============================================================
# SIMULACIÓN
# ============================================================

def simulate(df, candidates):

    trades = []

    next_available_index = 0

    for candidate in candidates:

        entry_index = candidate["entry_index"]

        # Solo una operación abierta a la vez
        if entry_index < next_available_index:
            continue

        direction = candidate["direction"]
        atr = candidate["atr"]

        entry_price = df.iloc[entry_index]["close"]

        if direction == "LONG":

            stop = entry_price - SL_ATR * atr
            target = entry_price + TP_ATR * atr

        else:

            stop = entry_price + SL_ATR * atr
            target = entry_price - TP_ATR * atr

        result = None
        exit_index = None
        exit_price = None

        # Buscar salida
        for j in range(entry_index + 1, len(df)):

            high = df.iloc[j]["high"]
            low = df.iloc[j]["low"]

            if direction == "LONG":

                # Conservador:
                # si toca SL y TP en la misma vela,
                # contamos STOP primero.
                if low <= stop:
                    result = -1.0
                    exit_index = j
                    exit_price = stop
                    break

                if high >= target:
                    result = 2.0
                    exit_index = j
                    exit_price = target
                    break

            else:

                if high >= stop:
                    result = -1.0
                    exit_index = j
                    exit_price = stop
                    break

                if low <= target:
                    result = 2.0
                    exit_index = j
                    exit_price = target
                    break

        # Si no encontró salida, no contamos la operación
        if result is None:
            continue

        duration_minutes = (
            df.iloc[exit_index]["datetime"]
            - df.iloc[entry_index]["datetime"]
        ).total_seconds() / 60

        trades.append({
            "entry_index": entry_index,
            "exit_index": exit_index,
            "direction": direction,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "result_R": result,
            "duration_min": duration_minutes,
        })

        next_available_index = exit_index + 1

    return trades


# ============================================================
# ESTADÍSTICAS
# ============================================================

def calculate_stats(trades):

    if not trades:
        return {
            "ops": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0,
            "total_R": 0,
            "avg_R": 0,
            "max_dd": 0,
            "worst_streak": 0,
            "profit_factor": 0,
            "avg_duration": 0,
        }

    results = [t["result_R"] for t in trades]

    wins = sum(r > 0 for r in results)
    losses = sum(r < 0 for r in results)

    total_R = sum(results)

    win_rate = wins / len(results) * 100

    avg_R = total_R / len(results)

    # Equity y drawdown
    equity = np.cumsum(results)
    running_max = np.maximum.accumulate(
        np.insert(equity, 0, 0)
    )[1:]

    drawdowns = equity - running_max
    max_dd = drawdowns.min()

    # Peor racha de pérdidas
    worst_streak = 0
    current_streak = 0

    for r in results:

        if r < 0:
            current_streak += 1
            worst_streak = max(
                worst_streak,
                current_streak
            )
        else:
            current_streak = 0

    gross_profit = sum(
        r for r in results if r > 0
    )

    gross_loss = abs(sum(
        r for r in results if r < 0
    ))

    if gross_loss > 0:
        profit_factor = gross_profit / gross_loss
    else:
        profit_factor = float("inf")

    avg_duration = np.mean([
        t["duration_min"]
        for t in trades
    ])

    return {
        "ops": len(trades),
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,
        "total_R": total_R,
        "avg_R": avg_R,
        "max_dd": max_dd,
        "worst_streak": worst_streak,
        "profit_factor": profit_factor,
        "avg_duration": avg_duration,
    }


# ============================================================
# EJECUCIÓN DE UN PERÍODO
# ============================================================

def run_period(name, end_date):

    print("\n" + "=" * 50)
    print(name)
    print("=" * 50)

    df = get_data(end_date)

    print(
        f"VELAS OBTENIDAS: {len(df)}"
    )

    print(
        f"DESDE: {df['datetime'].iloc[0]}"
    )

    print(
        f"HASTA: {df['datetime'].iloc[-1]}"
    )

    df = calculate_indicators(df)

    candidates = generate_candidates(df)

    print(
        f"CANDIDATAS 0.25 ATR: {len(candidates)}"
    )

    trades = simulate(
        df,
        candidates
    )

    stats = calculate_stats(trades)

    print("\nRESULTADOS")
    print("-----------------------------------")

    print(
        f"OPERACIONES: {stats['ops']}"
    )

    print(
        f"GANADORAS: {stats['wins']}"
    )

    print(
        f"PERDEDORAS: {stats['losses']}"
    )

    print(
        f"WIN RATE: {stats['win_rate']:.2f}%"
    )

    print(
        f"R TOTAL: {stats['total_R']:.2f}"
    )

    print(
        f"R MEDIO: {stats['avg_R']:.3f}"
    )

    print(
        f"DRAWDOWN MÁXIMO: {stats['max_dd']:.2f} R"
    )

    print(
        f"PEOR RACHA: {stats['worst_streak']}"
    )

    print(
        f"PROFIT FACTOR: {stats['profit_factor']:.3f}"
    )

    print(
        f"DURACIÓN MEDIA: {stats['avg_duration']:.1f} min"
    )

    return stats


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 60)
    print("VALIDACIÓN FUERA DE MUESTRA 2025")
    print("ESTRATEGIA 0.25 ATR")
    print("=" * 60)

    print("\nCONFIGURACIÓN CONGELADA:")
    print("EMA: 20 / 50 / 200")
    print("SEÑAL: 4/4")
    print("INVERSIÓN: ACTIVADA")
    print("CONFIRMACIÓN: 1 VELA")
    print("FILTRO CUERPO: 0.25 ATR")
    print("SL: 1.5 ATR")
    print("TP: 3 ATR")
    print("UNA OPERACIÓN A LA VEZ")

    all_trades = []

    period_results = []

    for index, (name, end_date) in enumerate(PERIODS):

        stats = run_period(
            name,
            end_date
        )

        period_results.append({
            "periodo": name,
            "operaciones": stats["ops"],
            "R": stats["total_R"],
            "win_rate": stats["win_rate"],
            "DD": stats["max_dd"],
            "racha": stats["worst_streak"],
            "PF": stats["profit_factor"],
        })

        # Evitar rate limit de Twelve Data
        if index < len(PERIODS) - 1:
            print("\nEsperando 20 segundos...")
            time.sleep(20)

    # ========================================================
    # RESUMEN
    # ========================================================

    print("\n")
    print("=" * 60)
    print("RESUMEN ENERO-SEPTIEMBRE 2025")
    print("=" * 60)

    summary = pd.DataFrame(period_results)

    print(summary.to_string(index=False))

    print("\n")
    print("=" * 60)
    print("FIN DE LA VALIDACIÓN")
    print("=" * 60)


if __name__ == "__main__":
    main()
