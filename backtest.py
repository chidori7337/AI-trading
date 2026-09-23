import os
import time
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
        "end_date": "2026-09-23 18:15:00"
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

    url = (
        "https://api.twelvedata.com/time_series"
    )


    params = {

        "symbol": SYMBOL,

        "interval": INTERVAL,

        "outputsize": OUTPUT_SIZE,

        "end_date": end_date,

        "apikey": API_KEY,

        "format": "JSON"
    }


    max_attempts = 6


    for attempt in range(
        max_attempts
    ):

        try:

            response = requests.get(
                url,
                params=params,
                timeout=30
            )

        except requests.RequestException as error:

            if attempt == max_attempts - 1:

                raise RuntimeError(
                    f"Error conectando con Twelve Data: {error}"
                )


            wait_seconds = (
                20 * (attempt + 1)
            )


            print()

            print(
                "Error de conexión."
            )

            print(
                f"Reintentando en "
                f"{wait_seconds} segundos..."
            )


            time.sleep(
                wait_seconds
            )

            continue


        # ====================================================
        # RATE LIMIT 429
        # ====================================================

        if response.status_code == 429:

            retry_after = (
                response.headers.get(
                    "Retry-After"
                )
            )


            if retry_after:

                try:

                    wait_seconds = int(
                        retry_after
                    )

                except ValueError:

                    wait_seconds = (
                        30 * (attempt + 1)
                    )

            else:

                wait_seconds = (
                    30 * (attempt + 1)
                )


            print()

            print(
                "LÍMITE DE TWELVE DATA ALCANZADO."
            )

            print(
                f"Intento {attempt + 1} "
                f"de {max_attempts}"
            )

            print(
                f"Esperando "
                f"{wait_seconds} segundos..."
            )


            time.sleep(
                wait_seconds
            )


            continue


        response.raise_for_status()


        try:

            data = response.json()

        except ValueError:

            raise RuntimeError(
                "Twelve Data devolvió una respuesta "
                "no válida."
            )


        if "values" not in data:

            raise ValueError(
                f"Error descargando datos: {data}"
            )


        df = pd.DataFrame(
            data["values"]
        )


        if df.empty:

            raise ValueError(
                "Twelve Data devolvió cero velas."
            )


        # ====================================================
        # PREPARAR DATOS
        # ====================================================

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


    raise RuntimeError(
        "No se pudieron descargar los datos "
        "después de varios intentos."
    )


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
    # LIMPIAR
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
# OBTENER SEÑAL ORIGINAL 4/4
# ============================================================

def get_original_signal(
    df,
    index
):

    current = df.iloc[index]

    long_score = 0
    short_score = 0


    # ========================================================
    # LONG
    # ========================================================

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


    # 4. Estructura alcista de 1 vela

    if (
        current["High"]
        >
        df.iloc[
            index - STRUCTURE_LOOKBACK
        ]["High"]

        and

        current["Low"]
        >
        df.iloc[
            index - STRUCTURE_LOOKBACK
        ]["Low"]
    ):

        long_score += 1


    # ========================================================
    # SHORT
    # ========================================================

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


    # 4. Estructura bajista de 1 vela

    if (
        current["High"]
        <
        df.iloc[
            index - STRUCTURE_LOOKBACK
        ]["High"]

        and

        current["Low"]
        <
        df.iloc[
            index - STRUCTURE_LOOKBACK
        ]["Low"]
    ):

        short_score += 1


    # ========================================================
    # DEVOLVER SEÑAL
    # ========================================================

    if (
        long_score
        >=
        SCORE_THRESHOLD
    ):

        return "LONG"


    if (
        short_score
        >=
        SCORE_THRESHOLD
    ):

        return "SHORT"


    return None


# ============================================================
# DRAWDOWN
# ============================================================

def calculate_max_drawdown(
    results
):

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
# PEOR RACHA
# ============================================================

def calculate_max_losing_streak(
    results
):

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

def calculate_profit_factor(
    results
):

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
# ANÁLISIS DE DURACIÓN
# ============================================================

def duration_bucket(
    minutes
):

    if minutes < 30:

        return "< 30 min"


    if minutes < 60:

        return "30-60 min"


    if minutes < 120:

        return "60-120 min"


    if minutes <= 180:

        return "120-180 min"


    if minutes <= 360:

        return "180-360 min"


    return "> 360 min"


DURATION_ORDER = [

    "< 30 min",

    "30-60 min",

    "60-120 min",

    "120-180 min",

    "180-360 min",

    "> 360 min"
]


def print_duration_analysis(
    results
):

    if results.empty:

        return


    data = (
        results
        .copy()
    )


    data["duration_group"] = (
        data["duration_minutes"]
        .apply(
            duration_bucket
        )
    )


    print()

    print(
        "========================================"
    )

    print(
        "RESULTADOS POR DURACIÓN"
    )

    print(
        "========================================"
    )


    total_operations = len(
        data
    )


    for group in DURATION_ORDER:

        group_results = data[
            data["duration_group"]
            ==
            group
        ]


        if group_results.empty:

            print()

            print(
                f"{group}: "
                "0 operaciones"
            )

            continue


        operations = len(
            group_results
        )


        winners = (
            group_results["result"]
            ==
            "WIN"
        ).sum()


        losers = (
            group_results["result"]
            ==
            "LOSS"
        ).sum()


        win_rate = (
            winners
            /
            operations
        ) * 100


        total_r = (
            group_results["R"]
            .sum()
        )


        average_r = (
            group_results["R"]
            .mean()
        )


        percentage = (
            operations
            /
            total_operations
        ) * 100


        print()

        print(
            f"{group}"
        )

        print(
            f"  Operaciones: {operations}"
        )

        print(
            f"  Porcentaje: {percentage:.2f}%"
        )

        print(
            f"  Win rate: {win_rate:.2f}%"
        )

        print(
            f"  R total: {total_r:.2f}"
        )

        print(
            f"  R medio: {average_r:.3f}"
        )


# ============================================================
# BACKTEST CON CONFIRMACIÓN
# ============================================================

def run_backtest(
    df
):

    trades = []

    raw_candidates = 0

    confirmed_candidates = 0


    i = 200


    while i < len(df) - 2:

        # ====================================================
        # PRIMERA VELA
        # ====================================================

        original_signal = (
            get_original_signal(
                df,
                i
            )
        )


        if original_signal is None:

            i += 1

            continue


        raw_candidates += 1


        # ====================================================
        # SEGUNDA VELA
        # ====================================================

        confirmation_signal = (
            get_original_signal(
                df,
                i + 1
            )
        )


        # La señal debe mantenerse
        if (
            confirmation_signal
            !=
            original_signal
        ):

            i += 1

            continue


        confirmed_candidates += 1


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

        entry_index = i + 1

        current = df.iloc[
            entry_index
        ]


        entry = current["Close"]

        atr = current["ATR"]


        if atr <= 0:

            i += 1

            continue


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
                (
                    TP_ATR
                    *
                    atr
                )
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
                (
                    TP_ATR
                    *
                    atr
                )
            )


        # ====================================================
        # BUSCAR SALIDA
        # ====================================================

        result = None

        result_r = None

        exit_index = None


        j = entry_index + 1


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
        # DURACIÓN
        # ====================================================

        entry_time = (
            current["datetime"]
        )


        exit_time = (
            df.iloc[
                exit_index
            ]["datetime"]
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
                    result_r,

                "duration_minutes":
                    duration_minutes
            }

        )


        # ====================================================
        # UNA SOLA OPERACIÓN
        # ====================================================

        i = exit_index + 1


    return (
        pd.DataFrame(trades),
        raw_candidates,
        confirmed_candidates
    )


# ============================================================
# EJECUTAR PERIODOS
# ============================================================

all_results = []

all_trades = []


for period_index, period in enumerate(
    VALIDATION_PERIODS,
    start=1
):

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

    (
        results,
        raw_candidates,
        confirmed_candidates

    ) = run_backtest(
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


    confirmation_rate = (

        confirmed_candidates
        /
        raw_candidates
        *
        100

        if raw_candidates > 0
        else 0
    )


    # ========================================================
    # MOSTRAR
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
        "CANDIDATAS ORIGINALES:",
        raw_candidates
    )


    print()

    print(
        "SEÑALES CONFIRMADAS:",
        confirmed_candidates
    )


    print()

    print(
        f"TASA DE CONFIRMACIÓN: "
        f"{confirmation_rate:.2f}%"
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
        f"WIN RATE: "
        f"{win_rate:.2f}%"
    )


    print()

    print(
        f"RESULTADO: "
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


    print()

    print(
        f"DURACIÓN MEDIA: "
        f"{results['duration_minutes'].mean():.1f} min"
    )


    print()

    print(
        f"DURACIÓN MEDIANA: "
        f"{results['duration_minutes'].median():.1f} min"
    )


    # ========================================================
    # ANÁLISIS DURACIÓN
    # ========================================================

    print_duration_analysis(
        results
    )


    # ========================================================
    # GUARDAR
    # ========================================================

    period_results = (
        results
        .copy()
    )


    period_results["periodo"] = name


    all_trades.append(
        period_results
    )


    all_results.append(
        {

            "periodo":
                name,

            "candidatas":
                raw_candidates,

            "confirmadas":
                confirmed_candidates,

            "tasa_confirmacion":
                confirmation_rate,

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

            "drawdown":
                max_drawdown,

            "racha_max":
                max_losing_streak,

            "profit_factor":
                profit_factor,

            "duracion_media":
                results[
                    "duration_minutes"
                ].mean()
        }
    )


    # ========================================================
    # PAUSA
    # ========================================================

    if (
        period_index
        <
        len(
            VALIDATION_PERIODS
        )
    ):

        print()

        print(
            "Esperando 20 segundos "
            "antes del siguiente período..."
        )

        time.sleep(
            20
        )


# ============================================================
# AUDITORÍA COMBINADA
# ============================================================

print()

print()

print(
    "========================================"
)

print(
    "AUDITORÍA COMBINADA"
)

print(
    "========================================"
)


if not all_trades:

    print()

    print(
        "No hubo operaciones."
    )

else:

    combined_results = pd.concat(
        all_trades,
        ignore_index=True
    )


    # ========================================================
    # ESTADÍSTICAS
    # ========================================================

    total_operations = len(
        combined_results
    )


    total_winners = (
        combined_results["result"]
        ==
        "WIN"
    ).sum()


    total_losers = (
        combined_results["result"]
        ==
        "LOSS"
    ).sum()


    total_r = (
        combined_results["R"]
        .sum()
    )


    win_rate = (
        total_winners
        /
        total_operations
    ) * 100


    average_r = (
        total_r
        /
        total_operations
    )


    max_drawdown = (
        calculate_max_drawdown(
            combined_results
        )
    )


    max_losing_streak = (
        calculate_max_losing_streak(
            combined_results
        )
    )


    profit_factor = (
        calculate_profit_factor(
            combined_results
        )


    )


    average_duration = (
        combined_results[
            "duration_minutes"
        ].mean()
    )


    median_duration = (
        combined_results[
            "duration_minutes"
        ].median()
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
        f"{win_rate:.2f}%"
    )


    print()

    print(
        f"R TOTAL: "
        f"{total_r:.2f}"
    )


    print()

    print(
        f"R MEDIO: "
        f"{average_r:.3f}"
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


    print()

    print(
        f"DURACIÓN MEDIA: "
        f"{average_duration:.1f} min"
    )


    print()

    print(
        f"DURACIÓN MEDIANA: "
        f"{median_duration:.1f} min"
    )


    # ========================================================
    # DURACIÓN COMBINADA
    # ========================================================

    print_duration_analysis(
        combined_results
    )


    # ========================================================
    # LONG / SHORT
    # ========================================================

    print()

    print(
        "========================================"
    )

    print(
        "LONG / SHORT"
    )

    print(
        "========================================"
    )


    long_results = combined_results[
        combined_results["signal"]
        ==
        "LONG"
    ]


    short_results = combined_results[
        combined_results["signal"]
        ==
        "SHORT"
    ]


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
    # RESUMEN POR PERÍODO
    # ========================================================

    print()

    print(
        "========================================"
    )

    print(
        "RESUMEN POR PERÍODO"
    )

    print(
        "========================================"
    )


    summary = pd.DataFrame(
        all_results
    )


    print()

    print(
        summary.to_string(
            index=False
        )
    )


# ============================================================
# FIN
# ============================================================

print()

print(
    "========================================"
)

print(
    "FIN DEL BACKTEST"
)

print(
    "========================================"
)
