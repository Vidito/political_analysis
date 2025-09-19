import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_absolute_error

# Load data
df = pd.read_csv("frp_poll_data.csv", parse_dates=["date"])

# Prophet expects columns: ds (date), y (value)
df = df.rename(columns={"date": "ds", "FrP": "y"})

# Train/test split (last 12 months = test)
train_data = df.iloc[:-12]
test_data = df.iloc[-12:]

# Define Prophet model with custom election-cycle seasonality
model = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=False,
    daily_seasonality=False,
    changepoint_prior_scale=0.3,  # more flexible trend changes
)

# Add 4-year (48 months) election cycle seasonality
model.add_seasonality(name="election_cycle", period=48, fourier_order=5)

# Fit model
model.fit(train_data)

# Forecast for training + test + next 12 months
future = model.make_future_dataframe(periods=12 + 12, freq="MS")  # month start
forecast = model.predict(future)

# Align forecasts with test dates
forecast_test = pd.merge_asof(
    test_data.sort_values("ds"),
    forecast[["ds", "yhat"]].sort_values("ds"),
    on="ds",
    direction="nearest",
)

# Evaluate forecast on test set
mae = mean_absolute_error(forecast_test["y"], forecast_test["yhat"])
print(f"\nMean Absolute Error (MAE): {mae:.2f}")

# Extract future-only forecasts
future_forecast = forecast[forecast["ds"] > df["ds"].max()][
    ["ds", "yhat", "yhat_lower", "yhat_upper"]
]
print("\nForecast for the next 12 months:")
print(future_forecast.to_string(index=False))

# Plot results
plt.figure(figsize=(15, 7))
plt.plot(train_data["ds"], train_data["y"], label="Training Data", color="blue")
plt.plot(
    test_data["ds"],
    test_data["y"],
    label="Actual Poll Results",
    color="green",
    marker="o",
)
plt.plot(
    forecast["ds"],
    forecast["yhat"],
    label="Prophet Forecast",
    color="red",
    linestyle="--",
)

# Prophet's uncertainty intervals
plt.fill_between(
    forecast["ds"],
    forecast["yhat_lower"],
    forecast["yhat_upper"],
    color="pink",
    alpha=0.3,
    label="Confidence Interval",
)

plt.title(
    "FrP Popularity: Historical Data, Test, and Future Forecast (with 4-year cycle)",
    fontsize=16,
)
plt.xlabel("Date", fontsize=12)
plt.ylabel("Popularity (%)", fontsize=12)
plt.legend()
plt.grid(True, linestyle="--", linewidth=0.5)
plt.tight_layout()
plt.show()
