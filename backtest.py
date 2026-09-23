import time
import os
import requests
import pandas as pd


API_KEY = os.getenv("TWELVE_DATA_API_KEY")


PERIODS = [
    ("ENERO 2026", "2026-01-31 23:55:00"),
    ("FEBRERO 2026", "2026-02-28 23:55:00"),
    ("MARZO 2026", "2026-03-31 23:55:00"),
    ("ABRIL 2026", "2026-04-30 23:55:00"),
    ("MAYO 2026", "2026-05-31 23:55:00"),
    ("JUNIO 2026", "2026-06-30 23:55:00"),
    ("JULIO 2026", "2026-07-31 23:55:00"),
    ("AGOSTO 2026", "2026-08-18 23:55:00"),
    ("SEPTIEMBRE 2026", "2026-09-23 18:15:00"),
]


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
                    print(
                        f"Rate limit. Esperando {wait_time} segundos..."
                    )
                    time.sleep(wait_time)
                    continue

                return None

            df = pd.DataFrame(data["values"])

            df["datetime"] = pd.to_datetime(df["datetime"])

            numeric_cols = [
                "open",
                "high",
                "low",
                "close",
            ]

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

    # EMA
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

    # ATR 14
    df["PrevClose"] = df["Close"].shift(1)

    df["TR"] = df.apply(
        lambda row: max(
            row["High"] - row["Low"],
            abs(row["High"] - row["PrevClose"])
            if pd.notna(row["PrevClose"])
            else 0,
            abs(row["Low"] - row["PrevClose"])
            if pd.notna(row["PrevClose"])
            else 0,
        ),
        axis=1,
    )

    df["ATR"] = df["TR"].rolling(14).mean()

    # Cuerpo de vela
    df["Body"] = abs(df["Close"] - df["Open"])

    return df


def get_original_signal(df, index):
    current = df.iloc[index]

    if pd.isna(current["EMA200"]) or pd.isna(current["ATR"]):
        return None

    long_score = 0
    short_score = 0

    # 1. Precio respecto a EMA20
    if current["Close"] > current["EMA20"]:
        long_score += 1

    if current["Close"] < current["EMA20"]:
        short_score += 1

    # 2. EMA20 respecto a EMA50
    if current["EMA20"] > current["EMA50"]:
        long_score += 1

    if current["EMA20"] < current["EMA50"]:
        short_score += 1

    # 3. EMA50 respecto a EMA200
    if current["EMA50"] > current["EMA200"]:
        long_score += 1

    if current["EMA50"] < current["EMA200"]:
        short_score += 1

    # 4. Estructura de una vela
    if current["Close"] > current["Open"]:
        long_score += 1

    if current["Close"] < current["Open"]:
        short_score += 1

    if long_score >= 4:
        return "LONG"

    if short_score >= 4:
        return "SHORT"

    return None


def calculate_max_drawdown(results):
    equity = 0
    peak = 0
    max_drawdown = 0

    for r in results:
        equity += r
        peak = max(peak, equity)
        drawdown = equity - peak
        max_drawdown = min(max_drawdown, drawdown)

    return max_drawdown


def calculate_max_losing_streak(results):
    current_streak = 0
    max_streak = 0

    for r in results:
        if r < 0:
            current_streak += 1
            max_streak = max(max_streak, current_streak)
        else:
            current_streak = 0

    return max_streak


def calculate_profit_factor(results):
    gross_profit = sum(r for r in results if r > 0)
    gross_loss = abs(sum(r for r in results if r < 0))

    if gross_loss == 0:
        return float("inf")

    return gross_profit / gross_loss


def duration_bucket(minutes):
    if minutes < 30:
        return "< 30 min"
    elif minutes < 60:
        return "30-60 min"
    elif minutes < 120:
        return "60-120 min"
    elif minutes < 180:
        return "120-180 min"
    elif minutes < 360:
        return "180-360 min"
    else:
        return "> 360 min"


def run_backtest(df):
    trades = []

    raw_candidates = 0
    confirmation_candidates = 0
    strong_confirmations = 0
    weak_confirmations = 0

    i = 200

    while i < len(df) - 2:

        # -----------------------------------
        # VELA A: señal original
        # -----------------------------------
        original_signal = get_original_signal(df, i)

        if original_signal is None:
            i += 1
            continue

        raw_candidates += 1

        # -----------------------------------
        # VELA B: confirmación
        # -----------------------------------
        confirmation_signal = get_original_signal(df, i + 1)

        if confirmation_signal != original_signal:
            i += 1
            continue

        confirmation_candidates += 1

        confirmation_candle = df.iloc[i + 1]

        body = confirmation_candle["Body"]
        atr = confirmation_candle["ATR"]

        # Seguridad por datos incompletos
        if pd.isna(body) or pd.isna(atr) or atr <= 0:
            i += 1
            continue

        # -----------------------------------
        # NUEVO FILTRO
        # Cuerpo >= 0.5 ATR
        # -----------------------------------
        if body < 0.5 * atr:
            weak_confirmations += 1
            i += 1
            continue

        strong_confirmations += 1

        # -----------------------------------
        # INVERTIMOS LA SEÑAL
        # -----------------------------------
        if original_signal == "LONG":
            signal = "SHORT"
        else:
            signal = "LONG"

        entry_index = i + 1
        current = df.iloc[entry_index]

        entry = current["Close"]
        atr = current["ATR"]

        if pd.isna(atr) or atr <= 0:
            i += 1
            continue

        # -----------------------------------
        # SL / TP
        # -----------------------------------
        risk_distance = 1.5 * atr

        if signal == "LONG":
            stop = entry - risk_distance
            target = entry + 3 * atr

        else:
            stop = entry + risk_distance
            target = entry - 3 * atr

        # -----------------------------------
        # BUSCAR SALIDA
        # -----------------------------------
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

            # Conservador:
            # si toca ambos en la misma vela,
            # contamos pérdida.
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

        # Si no hubo salida, terminamos
        # el backtest del período.
        if result is None:
            break

        entry_time = current["datetime"]
        exit_time = df.iloc[exit_index]["datetime"]

        duration = (
            exit_time - entry_time
        ).total_seconds() / 60

        trades.append(
            {
                "signal": signal,
                "result": result,
                "R": R,
                "duration": duration,
                "bucket": duration_bucket(duration),
            }
        )

        # Solo una operación abierta a la vez
        i = exit_index + 1

    # -----------------------------------
    # MÉTRICAS
    # -----------------------------------
    results = [trade["R"] for trade in trades]

    total_ops = len(trades)
    wins = sum(1 for r in results if r > 0)
    losses = sum(1 for r in results if r < 0)

    if total_ops > 0:
        win_rate = wins / total_ops * 100
        avg_r = sum(results) / total_ops
        avg_duration = sum(
            t["duration"] for t in trades
        ) / total_ops

        sorted_durations = sorted(
            t["duration"] for t in trades
        )

        median_duration = sorted_durations[
            len(sorted_durations) // 2
        ]

    else:
        win_rate = 0
        avg_r = 0
        avg_duration = 0
        median_duration = 0

    total_r = sum(results)

    max_dd = calculate_max_drawdown(results)
    losing_streak = calculate_max_losing_streak(results)
    profit_factor = calculate_profit_factor(results)

    # -----------------------------------
    # DURACIONES
    # -----------------------------------
    buckets = [
        "< 30 min",
        "30-60 min",
        "60-120 min",
        "120-180 min",
        "180-360 min",
        "> 360 min",
    ]

    duration_stats = {}

    for bucket in buckets:

        bucket_trades = [
            t for t in trades
            if t["bucket"] == bucket
        ]

        bucket_results = [
            t["R"] for t in bucket_trades
        ]

        count = len(bucket_trades)
        bucket_wins = sum(
            1 for r in bucket_results if r > 0
        )

        if count > 0:
            bucket_win_rate = (
                bucket_wins / count * 100
            )

            bucket_total_r = sum(bucket_results)

            bucket_avg_r = (
                bucket_total_r / count
            )

            percentage = (
                count / total_ops * 100
                if total_ops > 0
                else 0
            )

        else:
            bucket_win_rate = 0
            bucket_total_r = 0
            bucket_avg_r = 0
            percentage = 0

        duration_stats[bucket] = {
            "operations": count,
            "percentage": percentage,
            "win_rate": bucket_win_rate,
            "R_total": bucket_total_r,
            "R_avg": bucket_avg_r,
        }

    # -----------------------------------
    # LONG / SHORT
    # -----------------------------------
    long_trades = [
        t for t in trades
        if t["signal"] == "LONG"
    ]

    short_trades = [
        t for t in trades
        if t["signal"] == "SHORT"
    ]

    long_results = [t["R"] for t in long_trades]
    short_results = [t["R"] for t in short_trades]

    long_wins = sum(
        1 for r in long_results if r > 0
    )

    short_wins = sum(
        1 for r in short_results if r > 0
    )

    long_win_rate = (
        long_wins / len(long_results) * 100
        if long_results
        else 0
    )

    short_win_rate = (
        short_wins / len(short_results) * 100
        if short_results
        else 0
    )

    return {
        "trades": trades,
        "raw_candidates": raw_candidates,
        "confirmation_candidates": confirmation_candidates,
        "strong_confirmations": strong_confirmations,
        "weak_confirmations": weak_confirmations,
        "total_ops": total_ops,
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,
        "total_r": total_r,
        "avg_r": avg_r,
        "max_dd": max_dd,
        "losing_streak": losing_streak,
        "profit_factor": profit_factor,
        "avg_duration": avg_duration,
        "median_duration": median_duration,
        "duration_stats": duration_stats,
        "long_ops": len(long_results),
        "long_win_rate": long_win_rate,
        "long_r": sum(long_results),
        "short_ops": len(short_results),
        "short_win_rate": short_win_rate,
        "short_r": sum(short_results),
    }


def print_results(period_name, result):

    print("\n=========================")
    print("RESULTADOS")
    print("=========================")

    print(
        f"\nCANDIDATAS ORIGINALES: "
        f"{result['raw_candidates']}"
    )

    print(
        f"SEÑALES CONFIRMADAS: "
        f"{result['confirmation_candidates']}"
    )

    confirmation_rate = (
        result["confirmation_candidates"]
        / result["raw_candidates"]
        * 100
        if result["raw_candidates"] > 0
        else 0
    )

    print(
        f"TASA DE CONFIRMACIÓN: "
        f"{confirmation_rate:.2f}%"
    )

    print(
        f"\nCONFIRMACIONES FUERTES: "
        f"{result['strong_confirmations']}"
    )

    print(
        f"CONFIRMACIONES DÉBILES DESCARTADAS: "
        f"{result['weak_confirmations']}"
    )

    strong_rate = (
        result["strong_confirmations"]
        / result["confirmation_candidates"]
        * 100
        if result["confirmation_candidates"] > 0
        else 0
    )

    print(
        f"TASA DE CONFIRMACIÓN FUERTE: "
        f"{strong_rate:.2f}%"
    )

    print(
        f"\nOPERACIONES: "
        f"{result['total_ops']}"
    )

    print(
        f"GANADORAS: "
        f"{result['wins']}"
    )

    print(
        f"PERDEDORAS: "
        f"{result['losses']}"
    )

    print(
        f"WIN RATE: "
        f"{result['win_rate']:.2f}%"
    )

    print(
        f"RESULTADO: "
        f"{result['total_r']:.2f} R"
    )

    print(
        f"PROMEDIO: "
        f"{result['avg_r']:.3f} R"
    )

    print(
        f"DRAWDOWN MÁXIMO: "
        f"{result['max_dd']:.2f} R"
    )

    print(
        f"PEOR RACHA: "
        f"{result['losing_streak']}"
    )

    if result["profit_factor"] == float("inf"):
        print("PROFIT FACTOR: inf")
    else:
        print(
            f"PROFIT FACTOR: "
            f"{result['profit_factor']:.3f}"
        )

    print(
        f"DURACIÓN MEDIA: "
        f"{result['avg_duration']:.1f} min"
    )

    print(
        f"DURACIÓN MEDIANA: "
        f"{result['median_duration']:.1f} min"
    )

    print("\n========================================")
    print("RESULTADOS POR DURACIÓN")
    print("========================================")

    for bucket, stats in result["duration_stats"].items():

        print(f"\n{bucket}")
        print(
            f"  Operaciones: "
            f"{stats['operations']}"
        )
        print(
            f"  Porcentaje: "
            f"{stats['percentage']:.2f}%"
        )
        print(
            f"  Win rate: "
            f"{stats['win_rate']:.2f}%"
        )
        print(
            f"  R total: "
            f"{stats['R_total']:.2f}"
        )
        print(
            f"  R medio: "
            f"{stats['R_avg']:.3f}"
        )

    print("\n========================================")
    print("LONG / SHORT")
    print("========================================")

    print(
        f"\nLONG: "
        f"{result['long_ops']} ops | "
        f"{result['long_win_rate']:.2f}% | "
        f"{result['long_r']:.2f} R"
    )

    print(
        f"SHORT: "
        f"{result['short_ops']} ops | "
        f"{result['short_win_rate']:.2f}% | "
        f"{result['short_r']:.2f} R"
    )


def main():

    all_trades = []
    summary_rows = []

    for period_name, end_date in PERIODS:

        print("\n")
        print("#" * 40)
        print(period_name)
        print("#" * 40)

        df = download_data(end_date)

        if df is None or df.empty:
            print("No se pudieron descargar los datos.")
            continue

        print(
            f"\nFINAL DE DATOS: "
            f"{df['datetime'].iloc[-1]}"
        )

        print(
            f"\nVELAS OBTENIDAS: "
            f"{len(df)}"
        )

        print(
            f"\nDESDE: "
            f"{df['datetime'].iloc[0]}"
        )

        print(
            f"\nHASTA: "
            f"{df['datetime'].iloc[-1]}"
        )

        df = calculate_indicators(df)

        print(
            "\nINDICADORES CALCULADOS CORRECTAMENTE"
        )

        result = run_backtest(df)

        print_results(period_name, result)

        all_trades.extend(result["trades"])

        confirmation_rate = (
            result["confirmation_candidates"]
            / result["raw_candidates"]
            * 100
            if result["raw_candidates"] > 0
            else 0
        )

        summary_rows.append(
            {
                "periodo": period_name,
                "candidatas": result["raw_candidates"],
                "confirmadas": result[
                    "confirmation_candidates"
                ],
                "tasa_confirmacion":
                    confirmation_rate,
                "confirmaciones_fuertes":
                    result["strong_confirmations"],
                "confirmaciones_debiles":
                    result["weak_confirmations"],
                "operaciones":
                    result["total_ops"],
                "ganadoras":
                    result["wins"],
                "perdedoras":
                    result["losses"],
                "win_rate":
                    result["win_rate"],
                "R":
                    result["total_r"],
                "R_promedio":
                    result["avg_r"],
                "drawdown":
                    result["max_dd"],
                "racha_max":
                    result["losing_streak"],
                "profit_factor":
                    result["profit_factor"],
                "duracion_media":
                    result["avg_duration"],
            }
        )

        print(
            "\nEsperando 20 segundos antes "
            "del siguiente período..."
        )

        time.sleep(20)

    # ========================================
    # AUDITORÍA COMBINADA
    # ========================================

    print("\n")
    print("=" * 40)
    print("AUDITORÍA COMBINADA")
    print("=" * 40)

    combined_results = [
        trade["R"]
        for trade in all_trades
    ]

    total_ops = len(combined_results)

    wins = sum(
        1 for r in combined_results
        if r > 0
    )

    losses = sum(
        1 for r in combined_results
        if r < 0
    )

    total_r = sum(combined_results)

    avg_r = (
        total_r / total_ops
        if total_ops > 0
        else 0
    )

    win_rate = (
        wins / total_ops * 100
        if total_ops > 0
        else 0
    )

    max_dd = calculate_max_drawdown(
        combined_results
    )

    losing_streak = calculate_max_losing_streak(
        combined_results
    )

    profit_factor = calculate_profit_factor(
        combined_results
    )

    durations = [
        trade["duration"]
        for trade in all_trades
    ]

    avg_duration = (
        sum(durations) / len(durations)
        if durations
        else 0
    )

    sorted_durations = sorted(durations)

    median_duration = (
        sorted_durations[len(sorted_durations) // 2]
        if sorted_durations
        else 0
    )

    print(
        f"\nOPERACIONES: {total_ops}"
    )

    print(
        f"\nGANADORAS: {wins}"
    )

    print(
        f"\nPERDEDORAS: {losses}"
    )

    print(
        f"\nWIN RATE: {win_rate:.2f}%"
    )

    print(
        f"\nR TOTAL: {total_r:.2f}"
    )

    print(
        f"\nR MEDIO: {avg_r:.3f}"
    )

    print(
        f"\nDRAWDOWN MÁXIMO: {max_dd:.2f} R"
    )

    print(
        f"\nPEOR RACHA: {losing_streak}"
    )

    if profit_factor == float("inf"):
        print("\nPROFIT FACTOR: inf")
    else:
        print(
            f"\nPROFIT FACTOR: "
            f"{profit_factor:.3f}"
        )

    print(
        f"\nDURACIÓN MEDIA: "
        f"{avg_duration:.1f} min"
    )

    print(
        f"\nDURACIÓN MEDIANA: "
        f"{median_duration:.1f} min"
    )

    # ========================================
    # DURACIÓN COMBINADA
    # ========================================

    print("\n")
    print("=" * 40)
    print("RESULTADOS POR DURACIÓN")
    print("=" * 40)

    buckets = [
        "< 30 min",
        "30-60 min",
        "60-120 min",
        "120-180 min",
        "180-360 min",
        "> 360 min",
    ]

    for bucket in buckets:

        bucket_trades = [
            t for t in all_trades
            if t["bucket"] == bucket
        ]

        bucket_results = [
            t["R"] for t in bucket_trades
        ]

        count = len(bucket_trades)

        bucket_wins = sum(
            1 for r in bucket_results
            if r > 0
        )

        bucket_win_rate = (
            bucket_wins / count * 100
            if count > 0
            else 0
        )

        bucket_total_r = sum(
            bucket_results
        )

        bucket_avg_r = (
            bucket_total_r / count
            if count > 0
            else 0
        )

        percentage = (
            count / total_ops * 100
            if total_ops > 0
            else 0
        )

        print(f"\n{bucket}")
        print(
            f"  Operaciones: {count}"
        )
        print(
            f"  Porcentaje: {percentage:.2f}%"
        )
        print(
            f"  Win rate: {bucket_win_rate:.2f}%"
        )
        print(
            f"  R total: {bucket_total_r:.2f}"
        )
        print(
            f"  R medio: {bucket_avg_r:.3f}"
        )

    # ========================================
    # LONG / SHORT COMBINADO
    # ========================================

    print("\n")
    print("=" * 40)
    print("LONG / SHORT")
    print("=" * 40)

    long_trades = [
        t for t in all_trades
        if t["signal"] == "LONG"
    ]

    short_trades = [
        t for t in all_trades
        if t["signal"] == "SHORT"
    ]

    long_results = [
        t["R"] for t in long_trades
    ]

    short_results = [
        t["R"] for t in short_trades
    ]

    long_wins = sum(
        1 for r in long_results
        if r > 0
    )

    short_wins = sum(
        1 for r in short_results
        if r > 0
    )

    long_win_rate = (
        long_wins / len(long_results) * 100
        if long_results
        else 0
    )

    short_win_rate = (
        short_wins / len(short_results) * 100
        if short_results
        else 0
    )

    print(
        f"\nLONG: "
        f"{len(long_results)} ops | "
        f"{long_win_rate:.2f}% | "
        f"{sum(long_results):.2f} R"
    )

    print(
        f"\nSHORT: "
        f"{len(short_results)} ops | "
        f"{short_win_rate:.2f}% | "
        f"{sum(short_results):.2f} R"
    )

    # ========================================
    # RESUMEN POR PERÍODO
    # ========================================

    print("\n")
    print("=" * 40)
    print("RESUMEN POR PERÍODO")
    print("=" * 40)

    summary_df = pd.DataFrame(summary_rows)

    if not summary_df.empty:
        print(
            summary_df.to_string(
                index=False
            )
        )

    print("\n")
    print("=" * 40)
    print("FIN DEL BACKTEST")
    print("=" * 40)


if __name__ == "__main__":
    main()
