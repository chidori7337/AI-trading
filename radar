import os
import requests

API_KEY = os.environ["TWELVE_DATA_API_KEY"]

url = "https://api.twelvedata.com/time_series"

params = {
    "symbol": "EUR/USD",
    "interval": "5min",
    "outputsize": 10,
    "apikey": API_KEY
}

response = requests.get(url, params=params)
data = response.json()

if "values" not in data:
    print("ERROR:")
    print(data)
    raise SystemExit(1)

print("EUR/USD - últimas velas de 5 minutos\n")

for candle in data["values"]:
    print(
        f"{candle['datetime']} | "
        f"O: {candle['open']} "
        f"H: {candle['high']} "
        f"L: {candle['low']} "
        f"C: {candle['close']}"
    )
