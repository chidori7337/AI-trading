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

# 4/4 condiciones
SCORE_THRESHOLD = 4

# Gestión de riesgo
ATR_PERIOD = 14
SL_ATR = 1.5
TP_ATR = 3.0

# Estructura de 1 vela
STRUCTURE_LOOKBACK = 1


# ============================================================
# PERIODOS HISTÓRICOS
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
# FUNCIÓN PARA DESCARGAR DATOS
# ============================================================

def download_data(end_date):

    url = "https://api.twelvedata.com/time_series"

    params = {
        "symbol": SYMBOL,
        "interval": INTERVAL,
        "end_date": end_date,
        "outputsize": OUTPUT_SIZE,
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
# FUNCIÓN DE BACKTEST
# ============================================================

def run_backtest(df):

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


    # ========================================================
    # BACKTEST
    # ========================================================

    trades = []

    i = 200


    while i < len(df) - 1:

        current = df.iloc[i]

        long_score = 0
        short_score = 0


        # ====================================================
        # LONG
        # ====================================================

        # 1. Precio > EMA20

        if current["Close"] > current["EMA20"]:

            long_score += 1


        # 2. EMA20 > EMA50

        if current["EMA20"] > current["EMA50"]:

            long_score += 1


        # 3. EMA50 > EMA200

        if current["EMA50"] > current["EMA200"]:

            long_score += 1


        # 4. Estructura alcista 1 vela

        if (
            current["High"]
            >
            df.iloc[i - 1]["High"]

            and

            current["Low"]
            >
            df.iloc[i - 1]["Low"]
        ):

            long_score += 1


        # ====================================================
        # SHORT
        # ====================================================

        # 1. Precio < EMA20

        if current["Close"] < current["EMA20"]:

            short_score += 1


        # 2. EMA20 < EMA50

        if current["EMA20"] < current["EMA50"]:

            short_score += 1


        # 3. EMA50 < EMA200

        if current["EMA50"] < current["EMA200"]:

            short_score += 1


        # 4. Estructura bajista 1 vela

        if (
            current["High"]
            <
            df.iloc[i - 1]["High"]

            and

            current["Low"]
            <
            df.iloc[i - 1]["Low"]
        ):

            short_score += 1


        # ====================================================
        # SEÑAL
        # ====================================================

        signal = None


        if long_score >= SCORE_THRESHOLD:

            signal = "LONG"


        elif short_score >= SCORE_THRESHOLD:

            signal = "SHORT"


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
        # STOP / TARGET
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


                # Criterio conservador
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


                # Criterio conservador
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
                "result": result,
                "R": result_r
            }
        )


        # ====================================================
        # UNA SOLA OPERACIÓN ABIERTA
        # ====================================================

        i = exit_index + 1


    return pd.DataFrame(trades)


# ============================================================
# EJECUTAR LOS TRES PERIODOS
# ============================================================

all_results = []


for period_number, end_date in enumerate(
    VALIDATION_PERIODS,
    start=1
):

    print()
    print()
    print("########################################")
    print(
        f"PERIODO HISTÓRICO {period_number}"
    )
    print("########################################")


    print()
    print(
        f"FINAL DE DATOS: {end_date}"
    )


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


    print()

    print(
        "INDICADORES CALCULADOS CORRECTAMENTE"
    )


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
    # RESULTADOS
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
    # GUARDAR RESULTADO
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
print("========================================")
print("RESUMEN DE LOS TRES PERIODOS")
print("========================================")


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
    # TOTALES COMBINADOS
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


    combined_win_rate = (
        total_winners
        /
        total_operations
    ) * 100


    combined_r = (
        summary["R"]
        .sum()
    )


    combined_average = (
        combined_r
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
        f"RESULTADO TOTAL: {combined_r:.2f} R"
    )


    print()

    print(
        f"PROMEDIO POR OPERACIÓN: "
        f"{combined_average:.3f} R"
    )
