from flask import Flask, jsonify, render_template
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')  # Render the index.html file

@app.route('/forecast/<commodity>', methods=['GET'])  # Use the commodity parameter in the URL
def forecast(commodity):
    # Load and preprocess your dataset
    df = pd.read_csv("Dummy_Cropdataset_20Years.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)

    if commodity not in df["Category"].unique():
        return jsonify({"error": "Commodity not found"})

    # Filter data for the selected commodity
    df_commodity = df[df["Category"] == commodity]
    df_monthly = df_commodity["Price"].resample("M").mean()

    forecast_steps = 12
    model = ARIMA(df_monthly, order=(2, 1, 2))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=forecast_steps)
    forecast_dates = pd.date_range(start=df_monthly.index[-1], periods=forecast_steps + 1, freq="M")[1:]

    forecast_data = {
        "Date": forecast_dates.strftime('%a, %d %b %Y').tolist(),  # Ensure dates are formatted as strings
        "Price": forecast.tolist()
    }

    return jsonify(forecast_data)

if __name__ == '__main__':
    app.run(debug=True)
