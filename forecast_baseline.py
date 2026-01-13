import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

# --- 1. Daten laden ---
df = pd.read_csv("cashflow_weekly.csv")

# Prophet erwartet Spalten: ds = Datum, y = Wert
df_prophet = df.rename(columns={
    "week_start": "ds",
    "net_cash": "y"
})

# Datumsformat sicherstellen
df_prophet["ds"] = pd.to_datetime(df_prophet["ds"])

# --- 2. Modell erstellen ---
model = Prophet(
    weekly_seasonality=True,
    daily_seasonality=False,
    yearly_seasonality=False
)
model.fit(df_prophet)

# --- 3. Zukunftsdaten für 13 Wochen erzeugen ---
future = model.make_future_dataframe(periods=13, freq="W-MON")

# --- 4. Forecast berechnen ---
forecast = model.predict(future)

# Datumsformat sicherstellen
forecast["ds"] = pd.to_datetime(forecast["ds"])

# --- 5. Export der letzten 13 Wochen ---
forecast_tail = forecast.tail(13)
forecast_tail.to_csv("forecast_13weeks.csv", index=False)

# --- 6. Plot ---
plt.figure(figsize=(12, 6))

plt.plot(df_prophet["ds"], df_prophet["y"], label="Historische IST-Werte", linewidth=2)
plt.plot(forecast["ds"], forecast["yhat"], label="Forecast (yhat)", linestyle="--")
plt.fill_between(
    forecast["ds"],
    forecast["yhat_lower"],
    forecast["yhat_upper"],
    alpha=0.2,
    label="Unsicherheitsintervall"
)

plt.title("Baseline 13-Wochen Cashflow-Forecast (Prophet-Modell)")
plt.xlabel("Datum / Kalenderwochen")
plt.ylabel("Netto-Cashflow (€)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
