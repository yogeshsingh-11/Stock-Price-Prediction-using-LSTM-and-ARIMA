import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

def fetch_and_preprocess_data(ticker, start_date, end_date):
    # Fetch data
    data = yf.download(ticker, start=start_date, end=end_date)
    data = data[['Close']].dropna()
    data.reset_index(inplace=True)
    data.rename(columns={'Close': 'Price'}, inplace=True)
    
    # Plot stock prices
    plt.figure(figsize=(10, 6))
    plt.plot(data['Date'], data['Price'], label=f"{ticker} Prices")
    plt.title("Stock Prices Over Time")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid()
    plt.show()
    
    return data

# Example usage
data = fetch_and_preprocess_data('AAPL', '2020-01-01', '2023-01-01')
data.to_csv('stock_prices.csv', index=False)
