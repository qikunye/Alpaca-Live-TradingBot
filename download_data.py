import yfinance as yf

symbols = ["SPY", "AAPL", "MSFT"]
start_date = "2019-10-01"
end_date = "2024-12-31"

for symbol in symbols:
    df = yf.download(symbol, start=start_date, end=end_date)
    df.to_csv(f"{symbol}_ohlcv.csv")
