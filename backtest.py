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


# ============================================================
# ESTRATEGIA CONGELADA
# ============================================================

SCORE_THRESHOLD = 4

ATR_PERIOD = 14

SL_ATR = 1.5
TP_ATR = 3.0

STRUCTURE_LOOKBACK = 1


# ============================================================
# PERIODOS
# ============================================================

VALIDATION_PERIODS = [
    {
        "name": "ENERO 2026",
        "end_date": "2026-01-31 23:55:00"
    },
    {
        "name": "SEPTIEMBRE 2026 HASTA HOY",
        "end_date": "2026-09-23 23:55:00"
    }
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

def download_data(end_date):

    url = "https://api.twelvedata.com/time_series"

    params = {
        "symbol": SYMBOL,
        "interval": INTERVAL,
        "outputsize": OUTPUT_SIZE,
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
            f"Error descargando datos: {data}"
        )


    df = pd.DataFrame(
        data["values"]
    )


    # ========================================================
    # PREPARAR DATOS
    # ========================================================

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
# CALCULAR INDICADORES
# ============================================================

def calculate_indicators(df):

    # ========================================================
    # EMA
    # ========================================================

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


    # ========================================================
    # ATR
    # ========================================================

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


    # ========================================================
    # ELIMINAR FILAS SIN INDICADORES
    # ========================================================

    df.dropna(
        subset=[
            "EMA20",
            "EMA50",
            "EMA200",
            "ATR"
        ],
        inplace=True
    )


    df.reset_index(
        drop=True,
        inplace=True
    )


    return df


# ============================================================
# CALCULAR DRAWDOWN
# ============================================================

def calculate_drawdown(results):

    if results.empty:

        return 0.0


    cumulative_r = (
        results["R"]
        .cumsum()
    )


    running_max = (
        cumulative_r
        .cummax()
    )


    drawdown = (
        cumulative_r
        -
        running_max
    )


    max_drawdown = (
        drawdown
        .min()
    )


    return max_drawdown


# ============================================================
# CALCULAR PEOR RACHA DE PÉRDIDAS
# ============================================================

def calculate_max_losing_streak(results):

    max_streak = 0

    current_streak = 0


    for value in results["R"]:

        if value < 0:

            current_streak += 1

        else:

            current_streak = 0


        if current_streak > max_streak:

            max_streak = current_streak


    return max_streak


# ============================================================
# PROFIT FACTOR
# ============================================================

def calculate_profit_factor(results):

    gross_profit = (
        results.loc[
            results["R"] > 0,
            "R"
        ].sum()
    )


    gross_loss = (
        results.loc[
            results["R"] < 0,
            "R"
        ].abs().sum()
    )


    if gross_loss == 0:

        return None


    return (
        gross_profit
        /
        gross_loss
    )


# ============================================================
# BACKTEST
# ============================================================

def run_backtest(df):

    trades = []


    i = 200


    while i < len(df) - 1:

        current = df.iloc[i]


        long_score = 0

        short_score = 0


        # ====================================================
        # LONG ORIGINAL
        # ====================================================

        # 1. Precio > EMA20

        if (
            current["Close"]
            >
            current["EMA20"]
        ):

            long_score += 1


        # 2. EMA20 > EMA50

        if (
            current["EMA20"]
            >
            current["EMA50"]
        ):

            long_score += 1


        # 3. EMA50 > EMA200

        if (
            current["EMA50"]
            >
            current["EMA200"]
        ):

            long_score += 1


        # 4. Estructura alcista

        if (
            current["High"]
            >
            df.iloc[
                i - STRUCTURE_LOOKBACK
            ]["High"]

            and

            current["Low"]
            >
            df.iloc[
                i - STRUCTURE_LOOKBACK
            ]["Low"]
        ):

            long_score += 1


        # ====================================================
        # SHORT ORIGINAL
        # ====================================================

        # 1. Precio < EMA20

        if (
            current["Close"]
            <
            current["EMA20"]
        ):

            short_score += 1


        # 2. EMA20 < EMA50

        if (
            current["EMA20"]
            <
            current["EMA50"]
        ):

            short_score += 1


        # 3. EMA50 < EMA200

        if (
            current["EMA50"]
            <
            current["EMA200"]
        ):

            short_score += 1


        # 4. Estructura bajista

        if (
            current["High"]
            <
            df.iloc[
                i - STRUCTURE_LOOKBACK
            ]["High"]

            and

            current["Low"]
            <
            df.iloc[
                i - STRUCTURE_LOOKBACK
            ]["Low"]
        ):

            short_score += 1


        # ====================================================
        # SEÑAL ORIGINAL
        # ====================================================

        original_signal = None


        if (
            long_score
            >=
            SCORE_THRESHOLD
        ):

            original_signal = "LONG"


        elif (
            short_score
            >=
            SCORE_THRESHOLD
        ):

            original_signal = "SHORT"


        # Sin señal

        if original_signal is None:

            i += 1

            continue


        # ====================================================
        # INVERTIR SEÑAL
        # ====================================================

        if original_signal == "LONG":

            signal = "SHORT"

        else:

            signal = "LONG"


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

        if signal == "LONG":

            stop = (
                entry
                -
                (SL_ATR * atr)
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
                (SL_ATR * atr)
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


        while j < len(df):

            future = df.iloc[j]

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
                # si toca ambos, STOP primero.

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
        # GUARDAR
        # ====================================================

        trades.append(
            {
                "datetime":
                    current["datetime"],

                "original_signal":
                    original_signal,

                "signal":
                    signal,

                "entry":
                    entry,

                "stop":
                    stop,

                "target":
                    target,

                "result":
                    result,

                "R":
                    result_r
            }
        )


        # ====================================================
        # UNA SOLA OPERACIÓN
        # ====================================================

        i = exit_index + 1


    return pd.DataFrame(
        trades
    )


# ============================================================
# EJECUTAR PERIODOS
# ============================================================

all_results = []


for period in VALIDATION_PERIODS:

    name = period["name"]

    end_date = period["end_date"]


    print()
    print()

    print(
        "########################################"
    )

    print(
        name
    )

    print(
        "########################################"
    )


    print()

    print(
        f"FINAL DE DATOS: {end_date}"
    )


    # ========================================================
    # DESCARGAR
    # ========================================================

    df = download_data(
        end_date
    )


    print()

    print(
        "VELAS OBTENIDAS:",
        len(df)
    )


    print()

    print(
        "DESDE:",
        df["datetime"].iloc[0]
    )


    print()

    print(
        "HASTA:",
        df["datetime"].iloc[-1]
    )


    # ========================================================
    # INDICADORES
    # ========================================================

    df = calculate_indicators(
        df
    )


    print()

    print(
        "INDICADORES CALCULADOS CORRECTAMENTE"
    )


    # ========================================================
    # BACKTEST
    # ========================================================

    results = run_backtest(
        df
    )


    if results.empty:

        print()

        print(
            "No se encontraron operaciones."
        )

        continue


    # ========================================================
    # ESTADÍSTICAS
    # ========================================================

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


    max_drawdown = (
        calculate_drawdown(
            results
        )
    )


    max_losing_streak = (
        calculate_max_losing_streak(
            results
        )
    )


    profit_factor = (
        calculate_profit_factor(
            results
        )
    )


    # ========================================================
    # MOSTRAR RESULTADOS
    # ========================================================

    print()

    print(
        "========================="
    )

    print(
        "RESULTADOS"
    )

    print(
        "========================="
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
        f"RESULTADO TOTAL: "
        f"{total_r:.2f} R"
    )


    print()

    print(
        f"PROMEDIO POR OPERACIÓN: "
        f"{average_r:.3f} R"
    )


    print()

    print(
        f"DRAWDOWN MÁXIMO: "
        f"{max_drawdown:.2f} R"
    )


    print()

    print(
        f"PEOR RACHA DE PÉRDIDAS: "
        f"{max_losing_streak}"
    )


    if profit_factor is not None:

        print()

        print(
            f"PROFIT FACTOR: "
            f"{profit_factor:.3f}"
        )


    # ========================================================
    # RESULTADOS LONG
    # ========================================================

    long_results = results[
        results["signal"] == "LONG"
    ]


    if not long_results.empty:

        long_wins = (
            long_results["result"]
            ==
            "WIN"
        ).sum()


        long_total = len(
            long_results
        )


        long_r = (
            long_results["R"]
            .sum()
        )


        print()

        print(
            "LONG INVERTIDO:"
        )


        print(
            f"  Operaciones: {long_total}"
        )


        print(
            f"  Ganadoras: {long_wins}"
        )


        print(
            f"  Win rate: "
            f"{(long_wins / long_total) * 100:.2f}%"
        )


        print(
            f"  Resultado: "
            f"{long_r:.2f} R"
        )


    # ========================================================
    # RESULTADOS SHORT
    # ========================================================

    short_results = results[
        results["signal"] == "SHORT"
    ]


    if not short_results.empty:

        short_wins = (
            short_results["result"]
            ==
            "WIN"
        ).sum()


        short_total = len(
            short_results
        )


        short_r = (
            short_results["R"]
            .sum()
        )


        print()

        print(
            "SHORT INVERTIDO:"
        )


        print(
            f"  Operaciones: "
            f"{short_total}"
        )


        print(
            f"  Ganadoras: "
            f"{short_wins}"
        )


        print(
            f"  Win rate: "
            f"{(short_wins / short_total) * 100:.2f}%"
        )


        print(
            f"  Resultado: "
            f"{short_r:.2f} R"
        )


    # ========================================================
    # GUARDAR RESUMEN
    # ========================================================

    all_results.append(
        {
            "periodo":
                name,

            "operaciones":
                total_trades,

            "ganadoras":
                winners,

            "perdedoras":
                losers,

            "win_rate":
                win_rate,

            "R":
                total_r,

            "R_promedio":
                average_r,

            "drawdown_max":
                max_drawdown,

            "racha_perdidas":
                max_losing_streak,

            "profit_factor":
                profit_factor
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
    "RESUMEN FINAL"
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
    # TOTALES
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
        f"WIN RATE: "
        f"{combined_win_rate:.2f}%"
    )


    print()

    print(
        f"RESULTADO TOTAL: "
        f"{total_r:.2f} R"
    )


    print()

    print(
        f"PROMEDIO POR OPERACIÓN: "
        f"{combined_average:.3f} R"
    )


    print()

    print(
        "NOTA:"
    )

    print(
        "El periodo de septiembre NO es una "
        "validación fuera de muestra porque "
        "ese tramo ya se utilizó durante "
        "el proceso de ajuste."
    )
