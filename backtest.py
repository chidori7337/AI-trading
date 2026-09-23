import os
import requests
import pandas as pd


# ============================================================
# CONFIGURACIÓN
# ============================================================

API_KEY = os.getenv("TWELVE_DATA_API_KEY")

SYMBOL = "EUR/USD"

INTERVAL_5M = "5min"
INTERVAL_15M = "15min"

OUTPUT_SIZE_5M = 5000
OUTPUT_SIZE_15M = 2500


# ============================================================
# GESTIÓN DE RIESGO
# ============================================================

ATR_PERIOD = 14

SL_ATR = 1.5
TP_ATR = 3.0


# ============================================================
# ESTRUCTURA
# ============================================================

STRUCTURE_LOOKBACK = 1


# ============================================================
# PERIODOS HISTÓRICOS A VALIDAR
# ============================================================

VALIDATION_PERIODS = [
    "2026-06-30 23:55:00",
    "2026-07-31 23:55:00",
    "2026-08-18 23:55:00"
]


# ============================================================
# COMPROBAR API KEY
# ============================================================

if not API_KEY:
    raise ValueError(
        "No se encontró TWELVE_DATA_API_KEY"
    )


# ============================================================
# DESCARGAR DATOS
# ============================================================

def download_data(
    interval,
    outputsize,
    end_date
):

    url = "https://api.twelvedata.com/time_series"

    params = {
        "symbol": SYMBOL,
        "interval": interval,
        "outputsize": outputsize,
        "end_date": end_date,
        "apikey": API_KEY,
        "format": "JSON"
    }


    response = requests.get(
        url,
        params=params,
        timeout=30
    )


    response.raise_for_status()


    data = response.json()


    if "values" not in data:

        raise ValueError(
            f"Error descargando {interval}: {data}"
        )


    df = pd.DataFrame(
        data["values"]
    )


    df["datetime"] = pd.to_datetime(
        df["datetime"]
    )


    for column in [
        "open",
        "high",
        "low",
        "close"
    ]:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )


    df = (
        df
        .sort_values("datetime")
        .reset_index(drop=True)
    )


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
        subset=[
            "Open",
            "High",
            "Low",
            "Close"
        ],
        inplace=True
    )


    df.reset_index(
        drop=True,
        inplace=True
    )


    return df


# ============================================================
# CALCULAR ATR EN 5M
# ============================================================

def calculate_atr(df):

    previous_close = (
        df["Close"].shift(1)
    )


    tr1 = (
        df["High"]
        -
        df["Low"]
    )


    tr2 = (
        df["High"]
        -
        previous_close
    ).abs()


    tr3 = (
        df["Low"]
        -
        previous_close
    ).abs()


    true_range = pd.concat(
        [
            tr1,
            tr2,
            tr3
        ],
        axis=1
    ).max(axis=1)


    df["ATR"] = true_range.ewm(
        alpha=1 / ATR_PERIOD,
        adjust=False
    ).mean()


    return df


# ============================================================
# BACKTEST DE UN PERÍODO
# ============================================================

def run_backtest(
    df5,
    df15
):

    # ========================================================
    # EMAs 5M
    # ========================================================

    df5["EMA20"] = df5["Close"].ewm(
        span=20,
        adjust=False
    ).mean()


    df5["EMA50"] = df5["Close"].ewm(
        span=50,
        adjust=False
    ).mean()


    df5["EMA200"] = df5["Close"].ewm(
        span=200,
        adjust=False
    ).mean()


    # ========================================================
    # ATR 5M
    # ========================================================

    df5 = calculate_atr(
        df5
    )


    # ========================================================
    # EMAS 15M
    # ========================================================

    df15["EMA50"] = df15["Close"].ewm(
        span=50,
        adjust=False
    ).mean()


    df15["EMA200"] = df15["Close"].ewm(
        span=200,
        adjust=False
    ).mean()


    # ========================================================
    # TENDENCIA 15M
    # ========================================================
    #
    # Muy importante:
    #
    # Una vela de 15 minutos con timestamp 10:00
    # representa aproximadamente 10:00-10:15.
    #
    # Para una vela de 5 minutos que empieza a las 10:10,
    # esa vela de 15m todavía no está cerrada.
    #
    # Por eso usamos la vela de 15m anterior.
    #
    # Ejemplo:
    #
    # 5m = 10:10 -> usamos 15m = 09:45
    # 5m = 10:15 -> podemos usar 15m = 10:00
    #

    df5["completed_15m"] = (
        df5["datetime"].dt.floor("15min")
        -
        pd.Timedelta(minutes=15)
    )


    trend15 = df15[
        [
            "datetime",
            "EMA50",
            "EMA200"
        ]
    ].copy()


    trend15.rename(
        columns={
            "datetime": "completed_15m",
            "EMA50": "EMA50_15M",
            "EMA200": "EMA200_15M"
        },
        inplace=True
    )


    trend15 = trend15.sort_values(
        "completed_15m"
    )


    df5 = df5.sort_values(
        "completed_15m"
    )


    df5 = pd.merge_asof(
        df5,
        trend15,
        on="completed_15m",
        direction="backward"
    )


    df5 = df5.sort_values(
        "datetime"
    ).reset_index(drop=True)


    # ========================================================
    # ELIMINAR FILAS SIN INDICADORES
    # ========================================================

    df5.dropna(
        subset=[
            "EMA20",
            "EMA50",
            "EMA200",
            "ATR",
            "EMA50_15M",
            "EMA200_15M"
        ],
        inplace=True
    )


    df5.reset_index(
        drop=True,
        inplace=True
    )


    # ========================================================
    # BACKTEST
    # ========================================================

    trades = []


    i = 200


    while i < len(df5) - 1:

        current = df5.iloc[i]


        signal = None


        # ====================================================
        # LONG
        # ====================================================

        long_trend_15m = (
            current["EMA50_15M"]
            >
            current["EMA200_15M"]
        )


        long_price = (
            current["Close"]
            >
            current["EMA20"]
        )


        long_ema = (
            current["EMA20"]
            >
            current["EMA50"]
        )


        long_structure = (
            current["High"]
            >
            df5.iloc[
                i - STRUCTURE_LOOKBACK:i
            ]["High"].max()

            and

            current["Low"]
            >
            df5.iloc[
                i - STRUCTURE_LOOKBACK:i
            ]["Low"].min()
        )


        if (
            long_trend_15m
            and
            long_price
            and
            long_ema
            and
            long_structure
        ):

            signal = "LONG"


        # ====================================================
        # SHORT
        # ====================================================

        short_trend_15m = (
            current["EMA50_15M"]
            <
            current["EMA200_15M"]
        )


        short_price = (
            current["Close"]
            <
            current["EMA20"]
        )


        short_ema = (
            current["EMA20"]
            <
            current["EMA50"]
        )


        short_structure = (
            current["High"]
            <
            df5.iloc[
                i - STRUCTURE_LOOKBACK:i
            ]["High"].min()

            and

            current["Low"]
            <
            df5.iloc[
                i - STRUCTURE_LOOKBACK:i
            ]["Low"].max()
        )


        if (
            signal is None
            and
            short_trend_15m
            and
            short_price
            and
            short_ema
            and
            short_structure
        ):

            signal = "SHORT"


        # ====================================================
        # SIN SEÑAL
        # ====================================================

        if signal is None:

            i += 1
            continue


        # ====================================================
        # ENTRADA
        # ====================================================

        entry = current["Close"]

        atr = current["ATR"]


        if atr <= 0:

            i += 1
            continue


        # ====================================================
        # STOP Y TARGET
        # ====================================================

        risk_distance = (
            SL_ATR * atr
        )


        if signal == "LONG":

            stop = (
                entry
                -
                risk_distance
            )


            target = (
                entry
                +
                (TP_ATR * atr)
            )


        else:

            stop = (
                entry
                +
                risk_distance
            )


            target = (
                entry
                -
                (TP_ATR * atr)
            )


        # ====================================================
        # BUSCAR SALIDA
        # ====================================================

        result = None

        result_r = None

        exit_index = None


        j = i + 1


        while j < len(df5):

            future = df5.iloc[j]

            high = future["High"]

            low = future["Low"]


            # =================================================
            # LONG
            # =================================================

            if signal == "LONG":

                hit_stop = (
                    low <= stop
                )


                hit_target = (
                    high >= target
                )


                # Criterio conservador:
                # si toca ambos en la misma vela,
                # contamos STOP primero.

                if hit_stop:

                    result = "LOSS"

                    result_r = -1

                    exit_index = j

                    break


                if hit_target:

                    result = "WIN"

                    result_r = 2

                    exit_index = j

                    break


            # =================================================
            # SHORT
            # =================================================

            else:

                hit_stop = (
                    high >= stop
                )


                hit_target = (
                    low <= target
                )


                if hit_stop:

                    result = "LOSS"

                    result_r = -1

                    exit_index = j

                    break


                if hit_target:

                    result = "WIN"

                    result_r = 2

                    exit_index = j

                    break


            j += 1


        # ====================================================
        # SI NO SE CERRÓ
        # ====================================================

        if result is None:

            break


        # ====================================================
        # GUARDAR OPERACIÓN
        # ====================================================

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


        # ====================================================
        # UNA SOLA OPERACIÓN ABIERTA
        # ====================================================

        i = exit_index + 1


    return pd.DataFrame(
        trades
    )


# ============================================================
# EJECUTAR LOS PERIODOS
# ============================================================

all_results = []


for period_number, end_date in enumerate(
    VALIDATION_PERIODS,
    start=1
):

    print()
    print()
    print(
        "########################################"
    )

    print(
        f"PERIODO HISTÓRICO {period_number}"
    )

    print(
        "########################################"
    )


    print()

    print(
        f"FINAL DE DATOS: {end_date}"
    )


    # ========================================================
    # DESCARGAR 5M
    # ========================================================

    df5 = download_data(
        INTERVAL_5M,
        OUTPUT_SIZE_5M,
        end_date
    )


    print()

    print(
        "VELAS 5M:",
        len(df5)
    )


    print()

    print(
        "DESDE 5M:",
        df5["datetime"].iloc[0]
    )


    print()

    print(
        "HASTA 5M:",
        df5["datetime"].iloc[-1]
    )


    # ========================================================
    # DESCARGAR 15M
    # ========================================================

    df15 = download_data(
        INTERVAL_15M,
        OUTPUT_SIZE_15M,
        end_date
    )


    print()

    print(
        "VELAS 15M:",
        len(df15)
    )


    # ========================================================
    # BACKTEST
    # ========================================================

    results = run_backtest(
        df5,
        df15
    )


    print()

    print(
        "INDICADORES CALCULADOS CORRECTAMENTE"
    )


    # ========================================================
    # RESULTADOS
    # ========================================================

    if results.empty:

        print()

        print(
            "No se encontraron operaciones."
        )

        continue


    total_trades = len(
        results
    )


    winners = (
        results["result"]
        ==
        "WIN"
    ).sum()


    losers = (
        results["result"]
        ==
        "LOSS"
    ).sum()


    win_rate = (
        winners
        /
        total_trades
    ) * 100


    total_r = (
        results["R"]
        .sum()
    )


    average_r = (
        results["R"]
        .mean()
    )


    print()

    print(
        "OPERACIONES:",
        total_trades
    )


    print()

    print(
        "GANADORAS:",
        winners
    )


    print()

    print(
        "PERDEDORAS:",
        losers
    )


    print()

    print(
        f"WIN RATE: {win_rate:.2f}%"
    )


    print()

    print(
        f"RESULTADO: {total_r:.2f} R"
    )


    print()

    print(
        f"PROMEDIO: {average_r:.3f} R"
    )


    # ========================================================
    # RESULTADOS LONG
    # ========================================================

    long_results = results[
        results["signal"] == "LONG"
    ]


    if not long_results.empty:

        long_total = len(
            long_results
        )


        long_wins = (
            long_results["result"]
            ==
            "WIN"
        ).sum()


        long_losses = (
            long_results["result"]
            ==
            "LOSS"
        ).sum()


        long_win_rate = (
            long_wins
            /
            long_total
        ) * 100


        long_r = (
            long_results["R"]
            .sum()
        )


        print()

        print(
            "LONG:"
        )

        print(
            f"  Operaciones: {long_total}"
        )

        print(
            f"  Ganadoras: {long_wins}"
        )

        print(
            f"  Perdedoras: {long_losses}"
        )

        print(
            f"  Win rate: {long_win_rate:.2f}%"
        )

        print(
            f"  Resultado: {long_r:.2f} R"
        )


    # ========================================================
    # RESULTADOS SHORT
    # ========================================================

    short_results = results[
        results["signal"] == "SHORT"
    ]


    if not short_results.empty:

        short_total = len(
            short_results
        )


        short_wins = (
            short_results["result"]
            ==
            "WIN"
        ).sum()


        short_losses = (
            short_results["result"]
            ==
            "LOSS"
        ).sum()


        short_win_rate = (
            short_wins
            /
            short_total
        ) * 100


        short_r = (
            short_results["R"]
            .sum()
        )


        print()

        print(
            "SHORT:"
        )

        print(
            f"  Operaciones: {short_total}"
        )

        print(
            f"  Ganadoras: {short_wins}"
        )

        print(
            f"  Perdedoras: {short_losses}"
        )

        print(
            f"  Win rate: {short_win_rate:.2f}%"
        )

        print(
            f"  Resultado: {short_r:.2f} R"
        )


    # ========================================================
    # GUARDAR RESUMEN
    # ========================================================

    all_results.append(
        {
            "periodo": period_number,
            "final": end_date,
            "operaciones": total_trades,
            "ganadoras": winners,
            "perdedoras": losers,
            "win_rate": win_rate,
            "R": total_r,
            "R_promedio": average_r
        }
    )


# ============================================================
# RESUMEN FINAL
# ============================================================

print()
print()
print(
    "========================================"
)

print(
    "RESUMEN DE TODOS LOS PERIODOS"
)

print(
    "========================================"
)


if not all_results:

    print()

    print(
        "No hubo resultados."
    )

else:

    summary = pd.DataFrame(
        all_results
    )


    print()

    print(
        summary.to_string(
            index=False
        )
    )


    # ========================================================
    # TOTAL COMBINADO
    # ========================================================

    total_operations = (
        summary["operaciones"]
        .sum()
    )


    total_winners = (
        summary["ganadoras"]
        .sum()
    )


    total_losers = (
        summary["perdedoras"]
        .sum()
    )


    total_r = (
        summary["R"]
        .sum()
    )


    combined_win_rate = (
        total_winners
        /
        total_operations
    ) * 100


    combined_average = (
        total_r
        /
        total_operations
    )


    print()

    print(
        "========================================"
    )

    print(
        "TOTAL COMBINADO"
    )

    print(
        "========================================"
    )


    print()

    print(
        "OPERACIONES:",
        total_operations
    )


    print()

    print(
        "GANADORAS:",
        total_winners
    )


    print()

    print(
        "PERDEDORAS:",
        total_losers
    )


    print()

    print(
        f"WIN RATE: {combined_win_rate:.2f}%"
    )


    print()

    print(
        f"RESULTADO TOTAL: {total_r:.2f} R"
    )


    print()

    print(
        f"PROMEDIO POR OPERACIÓN: "
        f"{combined_average:.3f} R"
    )

