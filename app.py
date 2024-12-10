import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import streamlit as st

# Function to fetch stock data from Yahoo Finance
def fetch_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data['Close']

# ARIMA Model
def arima_model(train_data):
    model = ARIMA(train_data, order=(5, 1, 0))  # p=5, d=1, q=0 (you can fine-tune these values)
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=30)  # Forecasting the next 30 days
    return forecast

# LSTM Model
def lstm_model(train_data, future_days=30):
    # Data scaling
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train_data.values.reshape(-1, 1))

    # Prepare data for LSTM
    X_train = []
    y_train = []
    for i in range(60, len(train_scaled)):  # 60 is the lookback period
        X_train.append(train_scaled[i-60:i, 0])
        y_train.append(train_scaled[i, 0])
    
    X_train, y_train = np.array(X_train), np.array(y_train)
    
    # Reshaping the input for LSTM [samples, timesteps, features]
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)

    # Build LSTM Model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dense(units=1))  # Prediction for the next day

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=10, batch_size=32)

    # Predict the next 30 days
    test_data = train_scaled[-60:].reshape(1, -1)
    test_data = test_data.reshape((test_data.shape[0], test_data.shape[1], 1))
    prediction = model.predict(test_data)
    prediction = scaler.inverse_transform(prediction)

    return prediction

# Streamlit Web Application
def main():
    st.title('Stock Price Prediction')
    
    ticker = st.text_input("Enter Stock Symbol (e.g., AAPL, MSFT)", "AAPL")
    start_date = st.date_input("Start Date", pd.to_datetime('2020-01-01'))
    end_date = st.date_input("End Date", pd.to_datetime('2023-01-01'))

    if st.button("Predict"):
        st.write(f"Fetching stock data for {ticker}...")
        stock_data = fetch_stock_data(ticker, start_date, end_date)

        # Display stock data
        st.subheader(f"Stock Data for {ticker}")
        st.write(stock_data.tail())

        # Train-test split
        train_data = stock_data[:int(0.8 * len(stock_data))]
        test_data = stock_data[int(0.8 * len(stock_data)):]

        # ARIMA Prediction
        arima_forecast = arima_model(train_data)
        
        # LSTM Prediction
        lstm_forecast = lstm_model(train_data)

        # Create the date range for the next 30 days
        forecast_dates = pd.date_range(start=stock_data.index[-1] + pd.Timedelta(days=1), periods=30, freq='D')

        # Plot Results
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        ax.plot(stock_data.index, stock_data.values, label='Historical Prices', color='blue')  # Actual data
        ax.plot(forecast_dates, arima_forecast, label='ARIMA Prediction', color='green')  # ARIMA forecast
        ax.plot(forecast_dates, lstm_forecast.flatten(), label='LSTM Prediction', color='red')  # LSTM forecast

        ax.set_title(f'{ticker} Price Prediction')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price (USD)')
        ax.legend()
        st.pyplot(fig)

        # Display Performance Metrics
        arima_rmse = np.sqrt(np.mean((arima_forecast - test_data.values[:30])**2))
        lstm_rmse = np.sqrt(np.mean((lstm_forecast.flatten() - test_data.values[:30])**2))

        st.write(f"ARIMA RMSE: {arima_rmse}")
        st.write(f"LSTM RMSE: {lstm_rmse}")

if __name__ == '__main__':
    main()
