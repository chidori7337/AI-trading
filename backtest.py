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

# Costes que vamos a probar
COSTS_PIPS = [0.0, 0.5, 1.0, 1.5, 2.0]

# EUR/USD:
# 1 pip = 0.0001
PIP_SIZE = 0.0001

# Todo 2025
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

            response = requests.get(
                url,
                params=params,
                timeout=30
            )

            if response.status_code == 429:

                wait = 30 * (attempt + 1)

                print(
                    f"Rate limit (429). "
                    f"Esperando {wait} segundos..."
                )

                time.sleep(wait)
                continue

            response.raise_for_status()

            data = response.json()

            if "values" not in data:

                print("Respuesta de Twelve Data:")
                print(data)

                raise ValueError(
                    "No se recibieron velas."
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

            if attempt == max_retries - 1:
                raise

            wait = 10 * (attempt + 1)

            print(
                f"Error de conexión: {e}"
            )

            print(
                f"Reintentando en {wait} segundos..."
            )

            time.sleep(wait)

    raise RuntimeError(
        "No se pudieron obtener los datos."
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

    # MISMA FÓRMULA DE LAS PRUEBAS ANTERIORES
    df["atr"] = (
        df["tr"]
        .rolling(ATR_PERIOD)
        .mean()
    )

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
# CANDIDATAS
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

        # Confirmación de una vela
        if signal1 != signal2:
            continue

        atr = row2["atr"]

        if pd.isna(atr) or atr <= 0:
            continue

        # Cuerpo de la vela de confirmación
        body = abs(
            row2["close"] - row2["open"]
        )

        body_ratio = body / atr

        if body_ratio < BODY_ATR_FILTER:
            continue

        # INVERTIMOS LA SEÑAL
        if signal1 == "LONG":
            actual_direction = "SHORT"
        else:
            actual_direction = "LONG"

        candidates.append({
            "entry_index": i + 1,
            "direction": actual_direction,
            "atr": atr,
        })

    return candidates


# ============================================================
# SIMULAR OPERACIONES
# ============================================================

def simulate(df, candidates):

    trades = []

    next_available_index = 0

    for candidate in candidates:

        entry_index = candidate["entry_index"]

        # Una sola operación a la vez
        if entry_index < next_available_index:
            continue

        direction = candidate["direction"]
        atr = candidate["atr"]

        entry_price = df.iloc[
            entry_index
        ]["close"]

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
            "entry_index": entry_index,
            "exit_index": exit_index,
            "direction": direction,
            "result_R": result,
            "duration_min": duration,
        })

        next_available_index = (
            exit_index + 1
        )

    return trades


# ============================================================
# ESTADÍSTICAS
# ============================================================

def calculate_stats(results):

    if not results:

        return {
            "ops": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0,
            "R": 0,
            "avg_R": 0,
            "DD": 0,
            "streak": 0,
            "PF": 0,
        }

    wins = sum(
        r > 0 for r in results
    )

    losses = sum(
        r < 0 for r in results
    )

    total_R = sum(results)

    win_rate = (
        wins / len(results) * 100
    )

    avg_R = (
        total_R / len(results)
    )

    equity = np.cumsum(results)

    running_max = np.maximum.accumulate(
        np.insert(equity, 0, 0)
    )[1:]

    drawdowns = (
        equity - running_max
    )

    max_dd = drawdowns.min()

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
        r for r in results
        if r > 0
    )

    gross_loss = abs(sum(
        r for r in results
        if r < 0
    ))

    if gross_loss > 0:

        profit_factor = (
            gross_profit / gross_loss
        )

    else:

        profit_factor = float("inf")

    return {
        "ops": len(results),
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,
        "R": total_R,
        "avg_R": avg_R,
        "DD": max_dd,
        "streak": worst_streak,
        "PF": profit_factor,
    }


# ============================================================
# APLICAR COSTES
# ============================================================

def apply_costs(trades, cost_pips):

    # Cada operación paga el coste completo.
    # El coste se expresa como una fracción de R
    # calculada respecto al riesgo de 1.5 ATR.

    adjusted_results = []

    for trade in trades:

        # El coste en precio
        cost_price = (
            cost_pips * PIP_SIZE
        )

        # Para reconstruir el ATR necesitamos
        # conservarlo. Lo añadimos posteriormente.
        risk_price = trade["risk_price"]

        cost_R = (
            cost_price / risk_price
        )

        adjusted_R = (
            trade["result_R"] - cost_R
        )

        adjusted_results.append(
            adjusted_R
        )

    return adjusted_results


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 60)
    print("PRUEBA DE COSTES - TODO 2025")
    print("=" * 60)

    print()
    print("ESTRATEGIA CONGELADA:")
    print("EMA 20 / 50 / 200")
    print("SEÑAL 4/4")
    print("INVERSIÓN ACTIVADA")
    print("CONFIRMACIÓN 1 VELA")
    print("FILTRO CUERPO 0.25 ATR")
    print("SL 1.5 ATR")
    print("TP 3 ATR")
    print("UNA OPERACIÓN A LA VEZ")

    all_trades = []

    # ========================================================
    # DESCARGAR Y SIMULAR TODO 2025
    # ========================================================

    for index, (name, end_date) in enumerate(PERIODS):

        print()
        print("=" * 50)
        print(name)
        print("=" * 50)

        df = get_data(end_date)

        print(
            f"VELAS: {len(df)}"
        )

        df = calculate_indicators(df)

        candidates = generate_candidates(df)

        trades = simulate(
            df,
            candidates
        )

        # Guardamos el riesgo real de cada operación
        # para poder convertir costes de pips a R.
        for trade, candidate in zip(
            trades,
            []
        ):
            pass

        # Rehacemos la simulación conservando ATR/riesgo
        # directamente para los costes.
        all_period_trades = []

        next_available_index = 0

        for candidate in candidates:

            entry_index = candidate[
                "entry_index"
            ]

            if entry_index < next_available_index:
                continue

            direction = candidate[
                "direction"
            ]

            atr = candidate["atr"]

            entry_price = df.iloc[
                entry_index
            ]["close"]

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

            for j in range(
                entry_index + 1,
                len(df)
            ):

                high = df.iloc[j]["high"]
                low = df.iloc[j]["low"]

                if direction == "LONG":

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

            all_period_trades.append({
                "result_R": result,
                "risk_price": SL_ATR * atr,
            })

            next_available_index = (
                exit_index + 1
            )

        all_trades.extend(
            all_period_trades
        )

        print(
            f"OPERACIONES: "
            f"{len(all_period_trades)}"
        )

        if index < len(PERIODS) - 1:

            print(
                "Esperando 20 segundos..."
            )

            time.sleep(20)

    # ========================================================
    # RESULTADOS
    # ========================================================

    print()
    print("=" * 60)
    print("RESULTADOS BRUTOS")
    print("=" * 60)

    gross_results = [
        t["result_R"]
        for t in all_trades
    ]

    gross_stats = calculate_stats(
        gross_results
    )

    print(
        f"OPERACIONES: {gross_stats['ops']}"
    )

    print(
        f"WIN RATE: "
        f"{gross_stats['win_rate']:.2f}%"
    )

    print(
        f"R TOTAL: "
        f"{gross_stats['R']:.2f}"
    )

    print(
        f"R MEDIO: "
        f"{gross_stats['avg_R']:.3f}"
    )

    print(
        f"DRAWDOWN: "
        f"{gross_stats['DD']:.2f} R"
    )

    print(
        f"PEOR RACHA: "
        f"{gross_stats['streak']}"
    )

    print(
        f"PROFIT FACTOR: "
        f"{gross_stats['PF']:.3f}"
    )

    # ========================================================
    # COSTES
    # ========================================================

    print()
    print("=" * 60)
    print("SENSIBILIDAD A COSTES")
    print("=" * 60)

    cost_results = []

    for cost_pips in COSTS_PIPS:

        results = []

        for trade in all_trades:

            cost_price = (
                cost_pips * PIP_SIZE
            )

            cost_R = (
                cost_price
                / trade["risk_price"]
            )

            adjusted_R = (
                trade["result_R"]
                - cost_R
            )

            results.append(
                adjusted_R
            )

        stats = calculate_stats(
            results
        )

        cost_results.append({
            "cost_pips": cost_pips,
            "ops": stats["ops"],
            "win_rate": stats["win_rate"],
            "R": stats["R"],
            "avg_R": stats["avg_R"],
            "DD": stats["DD"],
            "streak": stats["streak"],
            "PF": stats["PF"],
        })

        print()
        print(
            f"COSTE: {cost_pips:.1f} pips"
        )

        print(
            f"R TOTAL: {stats['R']:.2f}"
        )

        print(
            f"R MEDIO: {stats['avg_R']:.3f}"
        )

        print(
            f"DRAWDOWN: {stats['DD']:.2f} R"
        )

        print(
            f"PROFIT FACTOR: {stats['PF']:.3f}"
        )

    # ========================================================
    # TABLA FINAL
    # ========================================================

    print()
    print("=" * 60)
    print("TABLA FINAL DE COSTES")
    print("=" * 60)

    df_results = pd.DataFrame(
        cost_results
    )

    print(
        df_results.to_string(
            index=False
        )
    )

    print()
    print("=" * 60)
    print("FIN")
    print("=" * 60)


if __name__ == "__main__":
    main()
