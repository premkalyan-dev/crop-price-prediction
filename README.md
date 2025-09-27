
# 📈 ARIMA-based Commodity Price Forecasting

This Python project implements a time series analysis pipeline using the **ARIMA (Autoregressive Integrated Moving Average)** model to forecast future prices for various agricultural commodities. The project processes historical data, applies the ARIMA model to each commodity individually, and visualizes the actual data against the predictions.

## ✨ Features

  * **Data Preparation:** Loads a 20-year dummy crop dataset, converts the date column to a datetime index, and filters data by commodity.
  * **Time Series Resampling:** Resamples the daily/irregular data into a consistent **monthly frequency** (`"M"`), averaging the prices for stability.
  * **ARIMA Modeling:** Applies an `ARIMA(2, 1, 2)` model (Autoregressive order 2, Integrated order 1, Moving Average order 2) to the historical price data for each unique commodity.
  * **Future Forecasting:** Generates a 12-month price forecast for each commodity beyond the last date in the dataset.
  * **Visualization:** Creates a separate plot for each commodity, showing the historical price trend and the future forecasted trend.
  * **Structured Output:** Stores all forecasts in a Python dictionary for easy access and potential further use.

## 🛠️ Prerequisites

### 1\. Libraries

This project requires Python and the following data science and time series libraries:

```bash
pip install pandas numpy matplotlib statsmodels
```

### 2\. Dataset

  * **File Name:** `Dummy_Cropdataset_20Years.csv`
  * **Format:** A standard CSV file that must contain at least the following columns:
    | Column Name | Data Type | Description |
    | :--- | :--- | :--- |
    | `Date` | Date/String | The date of the price record (e.g., 'YYYY-MM-DD'). |
    | `Category` | String | The name of the commodity (e.g., 'Wheat', 'Rice'). |
    | `Price` | Numeric | The recorded price value. |

## 🚀 How to Run

1.  **File Setup:** Place the Python script and the `Dummy_Cropdataset_20Years.csv` file in the same directory.

2.  **Execute the Script:** Open your terminal and run the script:

    ```bash
    python commodity_forecast.py
    ```

3.  **View Output:** The script will print the processing status for each commodity to the console and display a series of plots (one for each unique commodity) showing the actual and forecasted prices.

## 💻 Code Structure Highlights

### 1\. Data Loading and Setup

The script loads the CSV, ensures the 'Date' column is a proper index, and identifies all unique categories to loop through.

```python
df = pd.read_csv("Dummy_Cropdataset_20Years.csv")
df["Date"] = pd.to_datetime(df["Date"])
df.set_index("Date", inplace=True)
commodities = df["Category"].unique()
```

### 2\. Core Forecasting Loop

The script iterates over each commodity, prepares the data, fits the ARIMA model, and generates the forecast.

```python
for commodity in commodities:
    # Resample to monthly mean
    df_monthly = df[df["Category"] == commodity]["Price"].resample("M").mean()

    # ARIMA Model Fitting
    model = ARIMA(df_monthly, order=(2, 1, 2))
    model_fit = model.fit()

    # Generate Forecast
    forecast = model_fit.forecast(steps=forecast_steps)
    # ... Store result ...
```

### 3\. Visualization

Uses `matplotlib` to plot the historical monthly trend (`df_monthly`) and the new forecasted points (`forecast_dates`, `forecast`).

```python
plt.figure(figsize=(12, 6))
plt.plot(df_monthly, label="Actual Prices", color="blue")
plt.plot(forecast_dates, forecast, linestyle="dashed", color="red", label="Forecasted Prices")
# ... set titles and labels ...
plt.show()
```

## ⚠️ Customization and Model Refinement

The current script uses a fixed ARIMA order of `(2, 1, 2)`. This order was likely determined through preliminary analysis (like examining ACF and PACF plots).

**To refine the model for better accuracy:**

1.  **Visualize ACF/PACF:** Before training, plot the Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF) of the differenced time series (`df_monthly.diff().dropna()`) to help identify optimal values for the **p** (AR) and **q** (MA) orders.
2.  **Order Selection:** Use automated tools like `pmdarima.auto_arima` or a grid search to find the best `(p, d, q)` combination based on evaluation metrics like AIC/BIC.
3.  **Seasonality:** If the data exhibits yearly patterns, consider switching to the **SARIMA** (Seasonal ARIMA) model.
