from flask import Flask, request, jsonify
import pandas as pd
import yfinance as yf
from prophet import Prophet
import io
import base64
from flask_cors import CORS
import numpy as np
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
import numpy as np
import yfinance as yf
import pandas as pd

def remove_outliers_from_forecast(forecast, threshold=1):
    """
    Removes outliers from the forecast data by replacing any value in 'yhat', 'yhat_lower',
    and 'yhat_upper' that deviates significantly from its 5 neighboring values (2 before, 2 after).
    """
    # Handle each of the columns: 'yhat', 'yhat_lower', 'yhat_upper'
    for col in ['yhat', 'yhat_lower', 'yhat_upper']:
        values = forecast[col].values
        new_values = np.copy(values)

        # Iterate over all values except the first two and last two (since they don't have enough neighbors)
        for i in range(2, len(values) - 2):
            # Take 2 previous and 2 next neighbors to calculate the mean and standard deviation
            neighbors = [values[i - 2], values[i - 1], values[i + 1], values[i + 2]]
            mean_neighbors = np.mean(neighbors)
            std_dev = np.std(neighbors + [values[i]])
            current_val = values[i]

            # Check if the current value deviates from the mean by a factor of the threshold
            print(current_val - mean_neighbors, threshold * std_dev)
            if abs(values[i] - mean_neighbors) > threshold * std_dev:
                new_values[i] = mean_neighbors  # Replace the outlier with the mean of its neighbors

        forecast[col] = new_values  # Update the DataFrame with cleaned values for the current column

    return forecast



def make_forecast(ticker, periods, hist='max', return_forecast=True , display_chart=True, remove_outliers=True):
    # Fetch stock data using yfinance
    stock_data = yf.Ticker(ticker)
    hist_data = stock_data.history(period=hist, auto_adjust=True, interval="1d")

    # Prepare the DataFrame for Prophet
    df = pd.DataFrame()
    df['ds'] = hist_data.index.values  # Prophet expects 'ds' as the date column
    df['y'] = hist_data['Close'].values  # 'y' is the target variable (Closing price)

    # Initialize and fit the Prophet model
    m = Prophet(daily_seasonality=False)  # Adjust Prophet's configuration if needed
    m.fit(df)

    # Create a DataFrame for future predictions
    future = m.make_future_dataframe(periods=periods, freq='D')
    forecast = m.predict(future)

    # Optionally, remove outliers from the forecast results
    forecast = remove_outliers_from_forecast(forecast)

        # Return the forecasted data as a dictionary
    return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_dict(orient='records')




@app.route('/forecast', methods=['GET'])
def forecast():
    ticker = request.args.get('ticker')
    periods = int(request.args.get('periods', 30))
    hist = request.args.get('hist', 'max')

    display_chart = request.args.get('display_chart', 'true').lower() == 'true'
    return_forecast = request.args.get('return_forecast', 'false').lower() == 'true'

    forecast_data = make_forecast(ticker, periods, hist, return_forecast, display_chart)
    return jsonify(forecast_data)

if __name__ == '__main__':
    app.run(debug=True)
