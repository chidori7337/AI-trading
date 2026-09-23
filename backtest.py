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


FILTERS = {
    "SIN FILTRO": 0.0,
    "0.5 ATR": 0.5,
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
            response = requests.get(
                url,
                params=params,
                timeout=30
            )

            data = response.json()

            if "values" not in data:

                message = data.get(
                    "message",
                    "Error desconocido"
                )

                print(
                    f"Error Twelve Data: {message}"
                )

                if (
                    response.status_code == 429
                    or "limit" in message.lower()
                ):

                    wait_time = 20 * (attempt + 1)

                    print(
                        f"Rate limit. "
                        f"Esperando {wait_time} segundos..."
                    )

                    time.sleep(wait_time)
                    continue

                return None

            df = pd.DataFrame(data["values"])

            df["datetime"] = pd.to_datetime(
                df["datetime"]
            )

            numeric_cols = [
                "open",
                "high",
                "low",
                "close",
            ]

            for col in numeric_cols:
                df[col] = pd.to_numeric(
                    df[col],
                    errors="coerce"
                )

            df = df.sort_values(
                "datetime"
            ).reset_index(drop=True)

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

            print(
                f"Error descargando datos: {e}"
            )

            if attempt < max_retries - 1:

                wait_time = 10 * (attempt + 1)

                print(
                    f"Esperando {wait_time} segundos..."
                )

                time.sleep(wait_time)

    return None


def calculate_indicators(df):

    df = df.copy()

    # EMA 20
    df["EMA20"] = (
        df["Close"]
        .ewm(
            span=20,
            adjust=False
        )
        .mean()
    )

    # EMA 50
    df["EMA50"] = (
        df["Close"]
        .ewm(
            span=50,
            adjust=False
        )
        .mean()
    )

    # EMA 200
    df["EMA200"] = (
        df["Close"]
        .ewm(
            span=200,
            adjust=False
        )
        .mean()
    )

    # ATR
    df["PrevClose"] = (
        df["Close"].shift(1)
    )

    df["TR"] = df.apply(
        lambda row: max(
            row["High"] - row["Low"],

            abs(
                row["High"]
                - row["PrevClose"]
            )
            if pd.notna(
                row["PrevClose"]
            )
            else 0,

            abs(
                row["Low"]
                - row["PrevClose"]
            )
            if pd.notna(
                row["PrevClose"]
            )
            else 0,
        ),
        axis=1,
    )

    df["ATR"] = (
        df["TR"]
        .rolling(14)
        .mean()
    )

    # Cuerpo de la vela
    df["Body"] = abs(
        df["Close"]
        - df["Open"]
    )

    return df


def get_original_signal(df, index):

    current = df.iloc[index]

    if (
        pd.isna(current["EMA200"])
        or pd.isna(current["ATR"])
    ):
        return None

    long_score = 0
    short_score = 0

    # 1. Precio vs EMA20
    if current["Close"] > current["EMA20"]:
        long_score += 1

    if current["Close"] < current["EMA20"]:
        short_score += 1

    # 2. EMA20 vs EMA50
    if current["EMA20"] > current["EMA50"]:
        long_score += 1

    if current["EMA20"] < current["EMA50"]:
        short_score += 1

    # 3. EMA50 vs EMA200
    if current["EMA50"] > current["EMA200"]:
        long_score += 1

    if current["EMA50"] < current["EMA200"]:
        short_score += 1

    # 4. Estructura de 1 vela
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

        original_signal = get_original_signal(
            df,
            i
        )

        if original_signal is None:
            i += 1
            continue

        confirmation_signal = (
            get_original_signal(
                df,
                i + 1
            )
        )

        if (
            confirmation_signal
            != original_signal
        ):
            i += 1
            continue

        confirmation_candle = df.iloc[i + 1]

        body = confirmation_candle["Body"]
        atr = confirmation_candle["ATR"]

        if (
            pd.isna(body)
            or pd.isna(atr)
            or atr <= 0
        ):
            i += 1
            continue

        candidates.append(
            {
                "signal_index": i,
                "entry_index": i + 1,
                "original_signal":
                    original_signal,
                "body": body,
                "atr": atr,
                "body_atr_ratio":
                    body / atr,
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

    gross_profit = sum(
        r for r in results
        if r > 0
    )

    gross_loss = abs(
        sum(
            r for r in results
            if r < 0
        )
    )

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


def simulate_filter(
    df,
    candidates,
    threshold
):

    trades = []

    skipped_by_filter = 0

    next_available_index = 0

    for candidate in candidates:

        entry_index = candidate[
            "entry_index"
        ]

        # No podemos abrir otra operación
        # mientras haya una anterior abierta.
        if entry_index < next_available_index:
            continue

        body_atr_ratio = candidate[
            "body_atr_ratio"
        ]

        # Filtro
        if body_atr_ratio < threshold:

            skipped_by_filter += 1

            continue

        original_signal = candidate[
            "original_signal"
        ]

        # Inversión de señal
        if original_signal == "LONG":
            signal = "SHORT"
        else:
            signal = "LONG"

        current = df.iloc[entry_index]

        entry = current["Close"]
        atr = current["ATR"]

        if (
            pd.isna(atr)
            or atr <= 0
        ):
            continue

        # SL / TP
        risk_distance = 1.5 * atr

        if signal == "LONG":

            stop = (
                entry
                - risk_distance
            )

            target = (
                entry
                + 3 * atr
            )

        else:

            stop = (
                entry
                + risk_distance
            )

            target = (
                entry
                - 3 * atr
            )

        result = None
        R = None
        exit_index = None

        j = entry_index + 1

        while j < len(df):

            future = df.iloc[j]

            if signal == "LONG":

                hit_stop = (
                    future["Low"]
                    <= stop
                )

                hit_target = (
                    future["High"]
                    >= target
                )

            else:

                hit_stop = (
                    future["High"]
                    >= stop
                )

                hit_target = (
                    future["Low"]
                    <= target
                )

            # Conservador: si toca ambos
            # en la misma vela, contamos LOSS.
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

        exit_time = (
            df.iloc[
                exit_index
            ]["datetime"]
        )

        duration = (
            exit_time
            - entry_time
        ).total_seconds() / 60

        trades.append(
            {
                "signal": signal,
                "result": result,
                "R": R,
                "duration": duration,
                "bucket":
                    duration_bucket(
                        duration
                    ),
                "body_atr_ratio":
                    body_atr_ratio,
            }
        )

        # La siguiente entrada solo puede
        # ocurrir después de la salida.
        next_available_index = (
            exit_index + 1
        )

    results = [
        trade["R"]
        for trade in trades
    ]

    total_ops = len(trades)

    wins = sum(
        1 for r in results
        if r > 0
    )

    losses = sum(
        1 for r in results
        if r < 0
    )

    win_rate = (
        wins / total_ops * 100
        if total_ops > 0
        else 0
    )

    total_r = sum(results)

    avg_r = (
        total_r / total_ops
        if total_ops > 0
        else 0
    )

    max_dd = calculate_max_drawdown(
        results
    )

    losing_streak = (
        calculate_max_losing_streak(
            results
        )
    )

    profit_factor = (
        calculate_profit_factor(
            results
        )
    )

    durations = [
        trade["duration"]
        for trade in trades
    ]

    avg_duration = (
        sum(durations)
        / len(durations)
        if durations
        else 0
    )

    sorted_durations = sorted(
        durations
    )

    median_duration = (
        sorted_durations[
            len(sorted_durations) // 2
        ]
        if sorted_durations
        else 0
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
        "profit_factor":
            profit_factor,
        "avg_duration":
            avg_duration,
        "median_duration":
            median_duration,
        "skipped_by_filter":
            skipped_by_filter,
    }


def print_test_result(
    name,
    result,
    total_candidates
):

    print(f"\n{name}")
    print("-" * 35)

    print(
        f"OPERACIONES: "
        f"{result['operations']}"
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
        f"R TOTAL: "
        f"{result['R']:.2f}"
    )

    print(
        f"R MEDIO: "
        f"{result['avg_R']:.3f}"
    )

    print(
        f"DRAWDOWN MÁXIMO: "
        f"{result['drawdown']:.2f} R"
    )

    print(
        f"PEOR RACHA: "
        f"{result['losing_streak']}"
    )

    if (
        result["profit_factor"]
        == float("inf")
    ):
        print(
            "PROFIT FACTOR: inf"
        )
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

    if total_candidates > 0:

        skipped_percentage = (
            result["skipped_by_filter"]
            / total_candidates
            * 100
        )

    else:
        skipped_percentage = 0

    print(
        f"DESCARTADAS POR FILTRO: "
        f"{result['skipped_by_filter']}"
    )

    print(
        f"% CANDIDATAS DESCARTADAS: "
        f"{skipped_percentage:.2f}%"
    )


def main():

    combined = {
        name: []
        for name in FILTERS
    }

    period_results = []

    for period_name, end_date in PERIODS:

        print("\n")
        print("#" * 40)
        print(period_name)
        print("#" * 40)

        df = download_data(end_date)

        if df is None or df.empty:

            print(
                "No se pudieron descargar "
                "los datos."
            )

            continue

        print(
            f"\nFINAL DE DATOS: "
            f"{df['datetime'].iloc[-1]}"
        )

        print(
            f"VELAS OBTENIDAS: "
            f"{len(df)}"
        )

        print(
            f"DESDE: "
            f"{df['datetime'].iloc[0]}"
        )

        print(
            f"HASTA: "
            f"{df['datetime'].iloc[-1]}"
        )

        df = calculate_indicators(df)

        print(
            "\nINDICADORES CALCULADOS "
            "CORRECTAMENTE"
        )

        # ==================================
        # GENERAMOS LAS CANDIDATAS UNA SOLA VEZ
        # ==================================

        candidates = generate_candidate_signals(
            df
        )

        total_candidates = len(candidates)

        print(
            f"\nCANDIDATAS ORIGINALES: "
            f"{total_candidates}"
        )

        print(
            "\nLas dos pruebas usan "
            "exactamente estas mismas candidatas."
        )

        results_for_period = {}

        # ==================================
        # A/B
        # ==================================

        for name, threshold in FILTERS.items():

            result = simulate_filter(
                df,
                candidates,
                threshold
            )

            results_for_period[name] = result

            combined[name].extend(
                result["trades"]
            )

            print_test_result(
                name,
                result,
                total_candidates
            )

        # ==================================
        # COMPARACIÓN DIRECTA
        # ==================================

        a = results_for_period[
            "SIN FILTRO"
        ]

        b = results_for_period[
            "0.5 ATR"
        ]

        print("\n")
        print("=" * 40)
        print("CAMBIO DEL FILTRO 0.5 ATR")
        print("=" * 40)

        print(
            f"\nCAMBIO OPERACIONES: "
            f"{b['operations'] - a['operations']:+d}"
        )

        print(
            f"CAMBIO WIN RATE: "
            f"{b['win_rate'] - a['win_rate']:+.2f} puntos"
        )

        print(
            f"CAMBIO R: "
            f"{b['R'] - a['R']:+.2f}"
        )

        print(
            f"CAMBIO R MEDIO: "
            f"{b['avg_R'] - a['avg_R']:+.3f}"
        )

        print(
            f"CAMBIO DRAWDOWN: "
            f"{b['drawdown'] - a['drawdown']:+.2f} R"
        )

        print(
            f"CAMBIO RACHA: "
            f"{b['losing_streak'] - a['losing_streak']:+d}"
        )

        print(
            f"CAMBIO PROFIT FACTOR: "
            f"{b['profit_factor'] - a['profit_factor']:+.3f}"
        )

        period_results.append(
            {
                "periodo": period_name,

                "candidatas":
                    total_candidates,

                "ops_sin_filtro":
                    a["operations"],

                "R_sin_filtro":
                    a["R"],

                "DD_sin_filtro":
                    a["drawdown"],

                "ops_0_5":
                    b["operations"],

                "R_0_5":
                    b["R"],

                "DD_0_5":
                    b["drawdown"],

                "cambio_R":
                    b["R"] - a["R"],
            }
        )

        print(
            "\nEsperando 20 segundos..."
        )

        time.sleep(20)

    # ==================================
    # AUDITORÍA COMBINADA
    # ==================================

    print("\n")
    print("=" * 50)
    print("AUDITORÍA COMBINADA A/B")
    print("=" * 50)

    combined_results = {}

    for name, trades in combined.items():

        results = [
            trade["R"]
            for trade in trades
        ]

        total_ops = len(results)

        wins = sum(
            1 for r in results
            if r > 0
        )

        losses = sum(
            1 for r in results
            if r < 0
        )

        total_r = sum(results)

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
            results
        )

        losing_streak = (
            calculate_max_losing_streak(
                results
            )
        )

        profit_factor = (
            calculate_profit_factor(
                results
            )
        )

        durations = [
            trade["duration"]
            for trade in trades
        ]

        avg_duration = (
            sum(durations)
            / len(durations)
            if durations
            else 0
        )

        combined_results[name] = {
            "operations": total_ops,
            "wins": wins,
            "losses": losses,
            "win_rate": win_rate,
            "R": total_r,
            "avg_R": avg_r,
            "drawdown": max_dd,
            "losing_streak":
                losing_streak,
            "profit_factor":
                profit_factor,
            "avg_duration":
                avg_duration,
        }

        print("\n")
        print(name)
        print("-" * 35)

        print(
            f"OPERACIONES: {total_ops}"
        )

        print(
            f"GANADORAS: {wins}"
        )

        print(
            f"PERDEDORAS: {losses}"
        )

        print(
            f"WIN RATE: "
            f"{win_rate:.2f}%"
        )

        print(
            f"R TOTAL: "
            f"{total_r:.2f}"
        )

        print(
            f"R MEDIO: "
            f"{avg_r:.3f}"
        )

        print(
            f"DRAWDOWN MÁXIMO: "
            f"{max_dd:.2f} R"
        )

        print(
            f"PEOR RACHA: "
            f"{losing_streak}"
        )

        if profit_factor == float("inf"):
            print(
                "PROFIT FACTOR: inf"
            )
        else:
            print(
                f"PROFIT FACTOR: "
                f"{profit_factor:.3f}"
            )

        print(
            f"DURACIÓN MEDIA: "
            f"{avg_duration:.1f} min"
        )

    # ==================================
    # COMPARACIÓN COMBINADA
    # ==================================

    a = combined_results[
        "SIN FILTRO"
    ]

    b = combined_results[
        "0.5 ATR"
    ]

    print("\n")
    print("=" * 50)
    print("COMPARACIÓN COMBINADA")
    print("=" * 50)

    print(
        f"\nCAMBIO OPERACIONES: "
        f"{b['operations'] - a['operations']:+d}"
    )

    print(
        f"CAMBIO WIN RATE: "
        f"{b['win_rate'] - a['win_rate']:+.2f} puntos"
    )

    print(
        f"CAMBIO R TOTAL: "
        f"{b['R'] - a['R']:+.2f} R"
    )

    print(
        f"CAMBIO R MEDIO: "
        f"{b['avg_R'] - a['avg_R']:+.3f}"
    )

    print(
        f"CAMBIO DRAWDOWN: "
        f"{b['drawdown'] - a['drawdown']:+.2f} R"
    )

    print(
        f"CAMBIO PEOR RACHA: "
        f"{b['losing_streak'] - a['losing_streak']:+d}"
    )

    print(
        f"CAMBIO PROFIT FACTOR: "
        f"{b['profit_factor'] - a['profit_factor']:+.3f}"
    )

    # ==================================
    # RESUMEN POR PERÍODO
    # ==================================

    print("\n")
    print("=" * 50)
    print("RESUMEN POR PERÍODO")
    print("=" * 50)

    summary_df = pd.DataFrame(
        period_results
    )

    if not summary_df.empty:

        print(
            summary_df.to_string(
                index=False
            )
        )

    # ==================================
    # CONCLUSIÓN AUTOMÁTICA
    # ==================================

    print("\n")
    print("=" * 50)
    print("LECTURA AUTOMÁTICA")
    print("=" * 50)

    difference_R = (
        b["R"] - a["R"]
    )

    difference_dd = (
        b["drawdown"] - a["drawdown"]
    )

    difference_streak = (
        b["losing_streak"]
        - a["losing_streak"]
    )

    print()

    if difference_R > 0:
        print(
            f"El filtro 0.5 ATR añade "
            f"{difference_R:.2f} R."
        )

    elif difference_R < 0:
        print(
            f"El filtro 0.5 ATR resta "
            f"{abs(difference_R):.2f} R."
        )

    else:
        print(
            "El filtro 0.5 ATR no cambia "
            "el R total."
        )

    if difference_dd > 0:
        print(
            "El drawdown también mejora."
        )

    elif difference_dd < 0:
        print(
            "El drawdown empeora."
        )

    else:
        print(
            "El drawdown queda igual."
        )

    if difference_streak < 0:
        print(
            "La peor racha también mejora."
        )

    elif difference_streak > 0:
        print(
            "La peor racha empeora."
        )

    else:
        print(
            "La peor racha queda igual."
        )

    print("\n")
    print("=" * 50)
    print("FIN DEL BACKTEST")
    print("=" * 50)


if __name__ == "__main__":
    main()
