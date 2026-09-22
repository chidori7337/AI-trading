import os
import requests
import pandas as pd

API_KEY = os.environ["TWELVE_DATA_API_KEY"]

SYMBOL = "EUR/USD"
INTERVAL = "5min"

url = "https://api.twelvedata.com/time_series"

params = {
    "symbol": SYMBOL,
    "interval": INTERVAL,
    "outputsize": 5000,
    "apikey": API_KEY
}

response = requests.get(url, params=params)
data = response.json()

if "values" not in data:
    print("ERROR:")
    print(data)
    raise SystemExit(1)

df = pd.DataFrame(data["values"])

df["datetime"] = pd.to_datetime(df["datetime"])

for column in ["open", "high", "low", "close"]:
    df[column] = pd.to_numeric(df[column])

df = df.sort_values("datetime").reset_index(drop=True)

print("VELAS OBTENIDAS:", len(df))
print("DESDE:", df["datetime"].iloc[0])
print("HASTA:", df["datetime"].iloc[-1])
