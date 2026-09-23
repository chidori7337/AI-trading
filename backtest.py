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
# COSTES HIPOTÉTICOS
# ============================================================
#
# No son costes reales de un broker concreto.
# Son escenarios para comprobar cuánto margen queda
# si añadimos diferentes costes por operación.
#
# EUR/USD:
# 1 pip = 0.0001
#

PIP_SIZE = 0.0001

COST_SCENARIOS_PIPS = [
    0.5,
    1.0,
    1.5
]


# ============================================================
# PERIODOS
# ============================================================

VALIDATION_PERIODS = [
    {
        "name": "ENERO 2026",
        "end_date": "2026-01-31 23:55:00"
    },
    {
        "name": "FEBRERO 2026",
        "end_date": "2026-02-28 23:55:00"
    },
    {
        "name": "MARZO 2026",
        "end_date": "2026-03-31 23:55:00"
    },
    {
        "name": "ABRIL 2026",
        "end_date": "2026-04-30 23:55:00"
    },
    {
        "name": "MAYO 2026",
        "end_date": "2026-05-31 23:55:00"
    },
    {
        "name": "JUNIO 2026",
        "end_date": "2026-06-30 23:55:00"
    },
    {
        "name": "JULIO 2026",
        "end_date": "2026-07-31 23:55:00"
    },
    {
        "name": "AGOSTO 2026",
        "end_date": "2026-08-18 23:55:00"
    },
    {
        "name": "SEPTIEMBRE 2026",
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
# INDICADORES
# ============================================================

def calculate_indicators(df):

    # --------------------------------------------------------
    # EMA
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # ATR
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # Limpiar
    # --------------------------------------------------------

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
# DRAWDOWN
# ============================================================

def calculate_max_drawdown(results):

    if results.empty:
        return 0.0

    equity = (
        results["R"]
        .cumsum()
    )

    running_max = (
        equity
        .cummax()
    )

    drawdown = (
        equity
        -
        running_max
    )

    return drawdown.min()


# ============================================================
# PEOR RACHA DE PÉRDIDAS
# ============================================================

def calculate_max_losing_streak(results):

    max_streak = 0
    current_streak = 0

    for value in results["R"]:

        if value < 0:

            current_streak += 1

        else:

            current_streak = 0

        max_streak = max(
            max_streak,
            current_streak
        )

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
# ANALIZAR DURACIONES
# ============================================================

def calculate_duration_stats(results):

    if results.empty:

        return {
            "average_minutes": 0,
            "median_minutes": 0,
            "under_1h_pct": 0,
            "between_1_3h_pct": 0,
            "over_3h_pct": 0,
            "max_minutes": 0
        }


    durations = (
        results["duration_minutes"]
    )


    return {

        "average_minutes":
            durations.mean(),

        "median_minutes":
            durations.median(),

        "under_1h_pct":
            (
                durations < 60
            ).mean() * 100,

        "between_1_3h_pct":
            (
                (durations >= 60)
                &
                (durations <= 180)
            ).mean() * 100,

        "over_3h_pct":
            (
                durations > 180
            ).mean() * 100,

        "max_minutes":
            durations.max()
    }


# ============================================================
# COSTES
# ============================================================

def calculate_cost_adjusted_r(
    results,
    cost_pips
):

    if results.empty:
        return 0.0

    cost_price = (
        cost_pips
        *
        PIP_SIZE
    )

    cost_r = (
        cost_price
        /
        results["risk_distance"]
    )

    adjusted = (
        results["R"]
        -
        cost_r
    )

    return adjusted.sum()


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
        # INVERTIR
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


        # Distancia de riesgo

        risk_distance = (
            SL_ATR
            *
            atr
        )


        # ====================================================
        # STOP Y TARGET
        # ====================================================

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
        # DURACIÓN
        # ====================================================

        entry_time = (
            current["datetime"]
        )

        exit_time = (
            df.iloc[exit_index]["datetime"]
        )


        duration_minutes = (
            exit_time
            -
            entry_time
        ).total_seconds() / 60


        # ====================================================
        # GUARDAR
        # ====================================================

        trades.append(
            {
                "datetime":
                    entry_time,

                "exit_datetime":
                    exit_time,

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

                "risk_distance":
                    risk_distance,

                "result":
                    result,

                "R":
                    result_r,

                "duration_minutes":
                    duration_minutes
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
# EJECUTAR TODOS LOS PERIODOS
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
    # ESTADÍSTICAS GENERALES
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
        calculate_max_drawdown(
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


    duration_stats = (
        calculate_duration_stats(
            results
        )
    )


    # ========================================================
    # COSTES
    # ========================================================

    cost_results = {}


    for cost_pips in COST_SCENARIOS_PIPS:

        adjusted_r = (
            calculate_cost_adjusted_r(
                results,
                cost_pips
            )
        )

        cost_results[cost_pips] = (
            adjusted_r
        )


    # ========================================================
    # RESULTADOS
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
        f"RESULTADO BRUTO: "
        f"{total_r:.2f} R"
    )


    print()

    print(
        f"PROMEDIO: "
        f"{average_r:.3f} R"
    )


    print()

    print(
        f"DRAWDOWN MÁXIMO: "
        f"{max_drawdown:.2f} R"
    )


    print()

    print(
        f"PEOR RACHA: "
        f"{max_losing_streak}"
    )


    print()

    if profit_factor is not None:

        print(
            f"PROFIT FACTOR: "
            f"{profit_factor:.3f}"
        )

    else:

        print(
            "PROFIT FACTOR: N/A"
        )


    # ========================================================
    # DURACIÓN
    # ========================================================

    print()

    print(
        "========================="
    )

    print(
        "DURACIÓN DE OPERACIONES"
    )

    print(
        "========================="
    )


    print()

    print(
        f"DURACIÓN MEDIA: "
        f"{duration_stats['average_minutes']:.1f} min"
    )


    print()

    print(
        f"DURACIÓN MEDIANA: "
        f"{duration_stats['median_minutes']:.1f} min"
    )


    print()

    print(
        f"MENOS DE 1 HORA: "
        f"{duration_stats['under_1h_pct']:.2f}%"
    )


    print()

    print(
        f"ENTRE 1 Y 3 HORAS: "
        f"{duration_stats['between_1_3h_pct']:.2f}%"
    )


    print()

    print(
        f"MÁS DE 3 HORAS: "
        f"{duration_stats['over_3h_pct']:.2f}%"
    )


    print()

    print(
        f"MÁXIMA DURACIÓN: "
        f"{duration_stats['max_minutes']:.1f} min"
    )


    # ========================================================
    # COSTES
    # ========================================================

    print()

    print(
        "========================="
    )

    print(
        "ESCENARIOS CON COSTES"
    )

    print(
        "========================="
    )


    for cost_pips in COST_SCENARIOS_PIPS:

        print()

        print(
            f"{cost_pips:.1f} PIPS "
            f"POR OPERACIÓN: "
            f"{cost_results[cost_pips]:.2f} R"
        )


    # ========================================================
    # LONG / SHORT
    # ========================================================

    long_results = results[
        results["signal"] == "LONG"
    ]


    short_results = results[
        results["signal"] == "SHORT"
    ]


    print()

    print(
        "========================="
    )

    print(
        "LONG / SHORT"
    )

    print(
        "========================="
    )


    if not long_results.empty:

        long_wins = (
            long_results["result"]
            ==
            "WIN"
        ).sum()


        print()

        print(
            f"LONG: "
            f"{len(long_results)} ops | "
            f"{(long_wins / len(long_results)) * 100:.2f}% | "
            f"{long_results['R'].sum():.2f} R"
        )


    if not short_results.empty:

        short_wins = (
            short_results["result"]
            ==
            "WIN"
        ).sum()


        print()

        print(
            f"SHORT: "
            f"{len(short_results)} ops | "
            f"{(short_wins / len(short_results)) * 100:.2f}% | "
            f"{short_results['R'].sum():.2f} R"
        )


    # ========================================================
    # GUARDAR RESUMEN
    # ========================================================

    row = {

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

        "R_bruto":
            total_r,

        "R_promedio":
            average_r,

        "drawdown_max":
            max_drawdown,

        "racha_max":
            max_losing_streak,

        "profit_factor":
            profit_factor,

        "duracion_media_min":
            duration_stats[
                "average_minutes"
            ],

        "duracion_mediana_min":
            duration_stats[
                "median_minutes"
            ],

        "pct_1_3h":
            duration_stats[
                "between_1_3h_pct"
            ],

        "pct_mas_3h":
            duration_stats[
                "over_3h_pct"
            ]
    }


    for cost_pips in COST_SCENARIOS_PIPS:

        row[
            f"R_con_{cost_pips}_pips"
        ] = cost_results[
            cost_pips
        ]


    all_results.append(
        row
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
    "AUDITORÍA FINAL"
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


    # ========================================================
    # TABLA RESUMEN
    # ========================================================

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
        summary["R_bruto"]
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
        f"R BRUTO TOTAL: "
        f"{total_r:.2f} R"
    )


    print()

    print(
        f"R PROMEDIO: "
        f"{combined_average:.3f} R"
    )


    # ========================================================
    # COSTES COMBINADOS
    # ========================================================

    print()

    print(
        "========================================"
    )

    print(
        "TOTAL DESPUÉS DE COSTES"
    )

    print(
        "========================================"
    )


    for cost_pips in COST_SCENARIOS_PIPS:

        column = (
            f"R_con_{cost_pips}_pips"
        )


        total_after_costs = (
            summary[column]
            .sum()
        )


        print()

        print(
            f"{cost_pips:.1f} PIPS: "
            f"{total_after_costs:.2f} R"
        )


    # ========================================================
    # DURACIÓN COMBINADA
    # ========================================================

    print()

    print(
        "========================================"
    )

    print(
        "DURACIÓN COMBINADA"
    )

    print(
        "========================================"
    )


    weighted_average_duration = (
        summary["duracion_media_min"]
        *
        summary["operaciones"]
    ).sum() / total_operations


    weighted_pct_1_3h = (
        summary["pct_1_3h"]
        *
        summary["operaciones"]
    ).sum() / total_operations


    weighted_pct_over_3h = (
        summary["pct_mas_3h"]
        *
        summary["operaciones"]
    ).sum() / total_operations


    print()

    print(
        f"DURACIÓN MEDIA: "
        f"{weighted_average_duration:.1f} min"
    )


    print()

    print(
        f"OPERACIONES DE 1-3H: "
        f"{weighted_pct_1_3h:.2f}%"
    )


    print()

    print(
        f"OPERACIONES >3H: "
        f"{weighted_pct_over_3h:.2f}%"
    )


    print()

    print(
        "IMPORTANTE:"
    )

    print(
        "Los costes de 0.5, 1.0 y 1.5 pips "
        "son escenarios hipotéticos, no "
        "costes reales de un broker concreto."
    )

    print()

    print(
        "La ventana de septiembre ya fue "
        "utilizada durante el ajuste y por "
        "tanto no es una validación "
        "independiente."
    )
