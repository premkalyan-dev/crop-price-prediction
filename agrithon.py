import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Load the dataset
df = pd.read_csv("Dummy_Cropdataset_20Years.csv")

# Convert the 'Date' column to datetime format
df["Date"] = pd.to_datetime(df["Date"])

# Set 'Date' as the index
df.set_index("Date", inplace=True)

# Get unique commodities
commodities = df["Category"].unique()

# Forecast horizon (e.g., 12 months after 2024)
forecast_steps = 12

# Dictionary to store forecasts
forecasts = {}

for commodity in commodities:
    print(f"Processing {commodity}...")
    
    # Filter data for the commodity
    df_commodity = df[df["Category"] == commodity]
    
    # Resample to monthly frequency
    df_monthly = df_commodity["Price"].resample("M").mean()
    
    # Train ARIMA model (adjust order as needed)
    model = ARIMA(df_monthly, order=(2, 1, 2))
    model_fit = model.fit()
    
    # Forecast future prices
    forecast = model_fit.forecast(steps=forecast_steps)
    
    # Create forecasted date range
    forecast_dates = pd.date_range(start=df_monthly.index[-1], periods=forecast_steps + 1, freq="M")[1:]

    # Store forecast
    forecasts[commodity] = pd.DataFrame({"Date": forecast_dates, "Price": forecast.values})

    # Plot actual data until 2024 and predictions beyond
    plt.figure(figsize=(12, 6))
    plt.plot(df_monthly, label="Actual Prices", color="blue")
    plt.plot(forecast_dates, forecast, linestyle="dashed", color="red", label="Forecasted Prices")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.title(f"{commodity} Price Forecast Beyond 2024")
    plt.legend()
    plt.show()
