# ML Trading Bot

A sophisticated algorithmic trading system that combines multiple machine learning models, sentiment analysis, and Thompson sampling for automated stock trading using the Alpaca API.

## 🚀 Features

- **Multi-Model ML Approach**: Combines Transformer models and XGBoost for price prediction
- **Sentiment Analysis**: Uses FinBERT and NewsAPI for market sentiment evaluation
- **Thompson Sampling**: Multi-armed bandit approach for asset selection optimization
- **Risk Management**: Dynamic position sizing with stop-loss and take-profit orders
- **Backtesting Support**: Historical strategy validation using Yahoo Finance data
- **Real-time Trading**: Live trading capabilities through Alpaca API

## 🛠 Tech Stack

- **Python 3.8+**
- **Trading Framework**: Lumibot
- **ML Models**: 
  - Hugging Face Transformers (TimeSeriesTransformerForPrediction)
  - XGBoost
  - FinBERT for sentiment analysis
- **Data Sources**: 
  - Alpaca API (live trading)
  - Yahoo Finance (backtesting)
  - NewsAPI (sentiment data)
- **Key Libraries**: pandas, numpy, torch, scikit-learn, yfinance

## 📋 Prerequisites

### API Keys Required

1. **Alpaca Trading Account**
   - Get API keys from [Alpaca Markets](https://alpaca.markets/)
   - Paper trading supported for testing

2. **NewsAPI Key** 
   - Sign up at [NewsAPI](https://newsapi.org/)
   - Free tier available

### Required Files

Ensure these model files are in your project directory:
- `transformer_price_model/` - Pre-trained Transformer model directory
- `transformer_scaler.pkl` - Feature scaler for the Transformer model
- `xgb_bollinger_model.json` - Trained XGBoost model
- `xgb_features.pkl` - Feature names for XGBoost model
- `config.yml` - Configuration file with API credentials

## ⚙️ Installation

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd ml-trading-bot
```

2. **Install dependencies**
```bash
pip install lumibot alpaca-trade-api transformers torch xgboost pandas numpy yfinance requests joblib pyyaml
```

3. **Set up configuration file**
Create `config.yml`:
```yaml
alpaca:
  API_KEY: "your_alpaca_api_key"
  API_SECRET: "your_alpaca_secret_key"
  BASE_URL: "https://paper-api.alpaca.markets/v2"  # Use paper trading URL

newsapi:
  API_KEY: "your_newsapi_key"
```

4. **Add your NewsAPI key**
In `tradingbot.py`, replace:
```python
NEWSAPI_KEY = NEWS_API_KEY  # Replace with your actual key
```

## 🎯 Usage

### Backtesting Mode
```python
# Configure symbols and date range
symbols = [Asset("SPY"), Asset("AAPL")]
start_date = pd.Timestamp("2019-10-01").tz_localize(None)
end_date = pd.Timestamp("2024-12-31").tz_localize(None)

# Run backtest
strategy.backtest(YahooDataBacktesting, start_date, end_date)
```

### Live Trading Mode
```python
# Uncomment these lines in the code:
trader = Trader()
trader.add_strategy(strategy)
trader.run_all()
```

### Key Parameters
- `symbols`: List of assets to trade (default: ["SPY", "AAPL"])
- `cash_at_risk`: Portion of available cash to risk per trade (default: 0.5)
- `sleeptime`: Time between trading iterations (default: "24H")

## 🧠 How It Works

### 1. Asset Selection (Thompson Sampling)
Uses multi-armed bandit approach to optimize asset selection based on historical win/loss ratios.

### 2. Price Prediction
- **Transformer Model**: Predicts next-day closing price using OHLCV data and time features
- **XGBoost Model**: Predicts returns using Bollinger Bands and technical indicators

### 3. Sentiment Analysis
- Fetches recent news headlines for each symbol via NewsAPI
- Uses FinBERT model to classify sentiment (positive/negative/neutral)
- Adjusts trading signals based on sentiment probability

### 4. Signal Generation
Combines predictions with weighted scoring:
```python
combined_score = transformer_weight * price_change_ratio + xgb_weight * predicted_return
adjusted_score = combined_score * (1 + 0.1 * sentiment_factor)
```

### 5. Risk Management
- Dynamic position sizing based on available cash
- Volatility-adjusted stop-loss and take-profit levels
- Maximum drawdown protection (30% limit)
- Trade cooldown periods (5 days between trades per symbol)

## 📊 Model Details

### Transformer Model
- **Purpose**: Price prediction using time series data
- **Input Features**: OHLCV + day_of_week, day_of_month, month
- **Architecture**: TimeSeriesTransformerForPrediction from Hugging Face
- **Context Length**: Configurable based on model config

### XGBoost Model  
- **Purpose**: Return prediction using technical indicators
- **Features**: Bollinger Bands, volume ratios, price returns
- **Output**: Expected return percentage

### FinBERT Sentiment
- **Purpose**: News sentiment classification
- **Input**: Recent news headlines (3-day window)
- **Output**: Sentiment probability and classification

## ⚠️ Risk Disclaimers

- **This is experimental software**: Use paper trading first
- **Past performance ≠ Future results**: Models may not work in all market conditions
- **Market Risk**: You can lose money trading stocks
- **Model Risk**: ML predictions can be wrong
- **API Risk**: Ensure stable internet and API connections

## 🔧 Configuration Options

### Strategy Parameters
```python
MLTrader(
    symbols=["SPY", "AAPL", "QQQ"],  # Assets to trade
    cash_at_risk=0.3,                # Risk 30% of cash per trade
)
```

### Model Thresholds
- **Volatility Factor**: 1.5 (high confidence) or 2.5 (normal)
- **Score Threshold**: Dynamic based on market volatility
- **Sentiment Threshold**: 0.55 (positive), 0.45 (negative)

## 📈 Performance Monitoring

The bot logs:
- Portfolio value changes
- Individual trade decisions and reasoning
- Model predictions vs actual outcomes
- Sentiment analysis results
- Risk management actions

## 🐛 Troubleshooting

### Common Issues

1. **"Insufficient data" errors**
   - Ensure symbols have enough historical data
   - Check internet connection for data fetching

2. **Model loading failures**
   - Verify all model files are in the correct directories
   - Check file permissions

3. **API connection errors**
   - Validate API keys in config.yml
   - Check API rate limits

4. **Sentiment analysis failures**
   - Verify NewsAPI key and quota
   - Check if headlines contain relevant keywords

## 📝 License

This project is for educational purposes. Use at your own risk for live trading.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable  
5. Submit a pull request

## 📞 Support

For issues and questions:
- Check the troubleshooting section
- Review Lumibot documentation
- Verify API credentials and permissions

---

**⚠️ Important**: Always test with paper trading before using real money. This bot is experimental and may not be profitable in all market conditions.

