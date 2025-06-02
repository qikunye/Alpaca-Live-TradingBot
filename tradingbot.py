from lumibot.brokers import Alpaca
from lumibot.backtesting import YahooDataBacktesting
from lumibot.data_sources import YahooData
from lumibot.strategies.strategy import Strategy
from lumibot.traders import Trader
from datetime import datetime 
from alpaca_trade_api import REST 
from pandas import Timedelta
from csv_backtesting import CSVDataBacktesting
from finbert_utils import estimate_sentiment
from transformers import TimeSeriesTransformerConfig
import joblib
import numpy as np  
from numpy.random import beta
import xgboost as xgb
import pandas as pd
import yaml
from alpaca_trade_api import REST
import yfinance as yf
import requests
from transformers import TimeSeriesTransformerForPrediction
import torch
import os
from pathlib import Path

class Asset:
    def __init__(self, symbol):
        self.symbol = symbol
    def __str__(self):
        return self.symbol
    def __repr__(self):
        return f"Asset('{self.symbol}')"
    def __eq__(self, other):
        if isinstance(other, Asset):
            return self.symbol == other.symbol
        return False
    def __hash__(self):
        return hash(self.symbol)



with open("/Users/qikunye/Documents/Machine Learning Udemy/tradingbot/config.yml", "r") as file:
    config = yaml.safe_load(file)

API_KEY = config["alpaca"]["API_KEY"]
API_SECRET = config["alpaca"]["API_SECRET"]
BASE_URL = config["alpaca"].get("BASE_URL", "https://paper-api.alpaca.markets/v2")

ALPACA_CREDS = {
    "API_KEY": API_KEY,
    "API_SECRET": API_SECRET,
    "PAPER": True
}
# this is for live data from alpaca
# api = REST(key_id=API_KEY, secret_key=API_SECRET, base_url=BASE_URL)


NEWSAPI_KEY = "ec92ed3d709141bab0eee6bd36e4070a"  # replace with your actual key


from transformers import TimeSeriesTransformerForPrediction

model = TimeSeriesTransformerForPrediction.from_pretrained("./transformer_price_model")





news_cache = {}

class MLTrader(Strategy): 
    def initialize(self, symbols=["SPY", "AAPL"], cash_at_risk:float=0.5): 
        self.symbols = symbols
        self.sleeptime = "24H" 
        self.last_trade = {}
        self.cash_at_risk = cash_at_risk
        self.api = REST(base_url=BASE_URL, key_id=API_KEY, secret_key=API_SECRET)
        self.lookback = 30
        
        
        model_path = Path("tradingbot/transformer_price_model")
        self.transformer_model = TimeSeriesTransformerForPrediction.from_pretrained(str(model_path))
        print("Loading model from:", str(model_path))


        self.transformer_config = self.transformer_model.config
        scaler_path = os.path.join(os.path.dirname(__file__), "transformer_scaler.pkl")
        self.scaler = joblib.load(scaler_path)





   


    #Thompson Sampling setup
        self.symbol_index = {symbol: i for i, symbol in enumerate(self.symbols)}
        self.wins = [1] * len(self.symbols)
        self.losses = [1] * len(self.symbols)
        self.trade_outcomes = {}  # to store outcome tracking
    #xgboost
        self.xgb_model = xgb.XGBRegressor()
        self.xgb_model.load_model("tradingbot/xgb_bollinger_model.json")
        self.xgb_features = joblib.load("tradingbot/xgb_features.pkl")

    def position_sizing(self, symbol): 
        last_price = self.get_last_price(str(symbol))
        if last_price is None:
            self.log_message(f"⚠️ Warning: No last price available for {symbol}")
            return 0, None, 0

        position = self.get_position(symbol)
        position_value = 0
        if position is not None:
            position_value = position.quantity * last_price

        available_cash = self.get_cash() 
        if available_cash <= 0:
            return 0, last_price, 0

        quantity = round(available_cash * self.cash_at_risk / last_price, 0)
        if quantity < 1:  # Add validation
            return 0, last_price, 0
        return available_cash, last_price, quantity


    def get_dates(self): 
        today = self.get_datetime()
        three_days_prior = today - Timedelta(days=3)
        return today.strftime('%Y-%m-%d'), three_days_prior.strftime('%Y-%m-%d')
    
    def get_sentiment(self, symbol):
        try:
            today = pd.Timestamp.now().normalize() - pd.Timedelta(days=1)

            three_days_prior = today - pd.Timedelta(days=3)

            # Enforce 30-day limit on start date
            earliest_allowed = today - pd.Timedelta(days=30)
            start_date = max(three_days_prior, earliest_allowed)

            cache_key = f"{symbol}_{today.strftime('%Y-%m-%d')}"
            if cache_key in news_cache:
                headlines = news_cache[cache_key]
            else:
                # Prepare NewsAPI request
                url = "https://newsapi.org/v2/everything"
                params = {
                    "q": str(symbol),
                    "from": start_date.strftime('%Y-%m-%d'),
                    "to": today.strftime('%Y-%m-%d'),
                    "language": "en",
                    "sortBy": "relevancy",
                    "pageSize": 20,  # max results to fetch
                    "apiKey": NEWSAPI_KEY
                }
                response = requests.get(url, params=params)
                response.raise_for_status()
                articles = response.json().get("articles", [])
                headlines = [article["title"] for article in articles]
                news_cache[cache_key] = headlines
                if not headlines:
                    print(f"[{symbol}] No headlines found. Returning neutral sentiment.")
                    return 0.5, "neutral"


            #Optional: keyword filtering to reduce noise
            keywords = [symbol.lower(), "stock", "price", "earnings", "revenue", "ceo", "lawsuit", 
            "investigation", "buy", "sell", "downgrade", "upgrade", "target", "forecast", "regulation"]

            filtered_headlines = [h for h in headlines if any(kw in h.lower() for kw in keywords)]
            if not filtered_headlines and headlines:
                filtered_headlines = headlines[:5]

            probability, sentiment = estimate_sentiment(filtered_headlines)
            if probability > 0.55:
                sentiment = "positive"
            elif probability < 0.45:
                sentiment = "negative"
            else:
                sentiment = "neutral"
            #logging sentiment for debugging
            print(f"[{symbol}] Headlines used for sentiment analysis:")
            for headline in headlines:
                print(f" - {headline}")
            print(f"[{symbol}] Sentiment result: {sentiment}, Probability: {probability:.3f}")
            return probability, sentiment

        except Exception as e:
            print(f"[{symbol}] Sentiment fetch error: {e}")
            return 0.5, "neutral"

    
 

    def get_historical_data(self, symbol, limit=60):
        try:
            is_backtesting = isinstance(self.broker, YahooDataBacktesting)

            if is_backtesting:
                bars = self.get_historical_prices(symbol, timeframe="1d", length=limit, feed='iex')
                if len(bars) < self.lookback:
                    print(f"[{symbol}] Insufficient data: {len(bars)} < {self.lookback}")
                    return None
                if bars is None or bars.empty:
                    print(f"[{symbol}] Warning: No bars returned in backtest mode.")
                    return None
            else:
                # Yahoo Finance live fetch replacement
                data = yf.download(str(symbol), period=f"{limit}d", interval="1d")
                if data is None or data.empty:
                    print(f"[{symbol}] Warning: No bars returned from Yahoo Finance.")
                    return None
                data.index = pd.to_datetime(data.index)
                bars = data.rename(columns={
                    "Open": "open", 
                    "High": "high", 
                    "Low": "low", 
                    "Close": "close", 
                    "Volume": "volume"
                })
                bars = bars[['open', 'high', 'low', 'close', 'volume']]  # keep expected cols
            
            bars.index = pd.to_datetime(bars.index)
            bars = bars.tz_localize(None)
            return bars

        except Exception as e:
            print(f"[{symbol}] Error fetching bars: {e}")
            return None




    

    
    def choose_asset_thompson(self):
        ranked_assets = []
        for symbol in self.symbols:
            # Sample from Beta posterior (you can keep or simplify this logic)
            a = self.wins[self.symbol_index[symbol]]
            b = self.losses[self.symbol_index[symbol]]

            sampled_value = np.random.beta(a, b)
            ranked_assets.append((symbol, sampled_value))
        # Sort descending by sampled value
        ranked_assets.sort(key=lambda x: x[1], reverse=True)
        return [s for s, _ in ranked_assets]

    
    def record_trade_outcome(self, symbol, outcome):
        i = self.symbol_index[symbol]
        if outcome == 1:
            self.wins[i] += 1
        else:
            self.losses[i] += 1

     #transformer 
    def get_transformer_prediction(self, bars):
        df = bars[['open', 'high', 'low', 'close', 'volume']].tail(self.lookback)
        if len(df) < self.lookback:
            return None

        scaled = self.transformer_scaler.transform(df)
        input_tensor = torch.tensor([scaled], dtype=torch.float32)  # shape (1, 30, 5)

        with torch.no_grad():
            output = self.transformer_model(past_values=input_tensor).logits
        pred_scaled = output[0].item()

        last_row = scaled[-1]
        last_row[3] = pred_scaled
        full_reconstructed = np.array([last_row])
        pred_price = self.transformer_scaler.inverse_transform(full_reconstructed)[0][3]
        return float(pred_price)

   
    
    def get_xgb_prediction(self, bars):
        df = pd.DataFrame(bars)

        # Flatten multi-index columns
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
        print(f"Input df columns: {df.columns.tolist()}, shape: {df.shape}")

        # Check required columns
        if not all(col in df.columns for col in ['close', 'volume']):
            raise ValueError(f"Missing required columns for XGB features. Got: {df.columns.tolist()}")

        # Feature engineering
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['std_20'] = df['close'].rolling(window=20).std()
        df['upper_band'] = df['sma_20'] + 2 * df['std_20']
        df['lower_band'] = df['sma_20'] - 2 * df['std_20']
        df['avg_vol_15'] = df['volume'].rolling(window=15).mean()
        df['return_1d'] = df['close'].pct_change()
        df['close_to_upper'] = (df['close'] - df['upper_band']) / df['upper_band']
        df['close_to_lower'] = (df['close'] - df['lower_band']) / df['lower_band']
        df['vol_to_avgvol'] = df['volume'] / df['avg_vol_15']

        print(f"Shape before dropna: {df.shape}")
        df.dropna(inplace=True)
        print(f"Shape after dropna: {df.shape}")

        if df.empty:
            print("DataFrame is empty after dropna, no prediction made.")
            return None

        try:
            latest_features = df[self.xgb_features].iloc[-1]
            prediction = self.xgb_model.predict(latest_features.values.reshape(1, -1))[0]
            print(f"Prediction: {prediction}")
            return prediction
        except Exception as e:
            print(f"Error during prediction: {e}")
            return None







    def on_trading_iteration(self):
        ranked_symbols = self.choose_asset_thompson()
        now = self.get_datetime()
        
        for symbol in ranked_symbols:
            cash, last_price, quantity = self.position_sizing(symbol)
            if last_price is None or quantity == 0:
                continue 

            # Skip if already in a position
            position = self.get_position(symbol)
            if position is not None and position.quantity > 0:
                print(f"[{symbol}] Already in position. Skipping.")
                continue

            # Cooldown check
            last_trade_time = self.last_trade.get(f"{symbol}_time", now - Timedelta(days=10))
            if (now - last_trade_time).days < 5:
                print(f"[{symbol}] In cooldown period. Skipping.")
                continue

            probability, sentiment = self.get_sentiment(str(symbol))
            sentiment_factor = (probability - 0.5) * 2  # from -1 to 1

            # Fetch recent bars
            bars = self.get_historical_data(symbol, limit=60)
            if bars is None or len(bars) < 50:
                print(f"[{symbol}] Skipping - not enough bars (have {0 if bars is None else len(bars)})")
                continue

            # Predict XGB return
            xgb_return = self.get_xgb_prediction(bars)
            if xgb_return is None:
                print(f"[{symbol}] Skipping - XGB prediction failed")
                continue

            # Predict price using LSTM
            predicted_price = self.get_transformer_prediction(bars)
            if predicted_price is None:
                print(f"[{symbol}] Skipping - LSTM prediction failed")
                continue

            print(f"[{symbol}] Sentiment: {sentiment}, Prob: {probability:.3f}, XGB: {xgb_return:.4f}, LSTM: {predicted_price:.2f}, Last: {last_price:.2f}")

            transformer_price = self.get_transformer_prediction(bars)

            # Initialize trade history if not already
            if symbol not in self.last_trade:
                self.last_trade[symbol] = None
            
            # Calculate LSTM and XGB signal strengths
            transformer_score = (transformer_price - last_price) / last_price
            xgb_strength = abs(xgb_return)

           
            # 4️⃣ Normalize weights
            total = transformer_score + xgb_strength
            if total > 0:
                transformer_weight = transformer_score / total
                xgb_weight = xgb_strength / total
            else:
                transformer_weight = xgb_weight = 0.5

            


            # 5️⃣ Calculate final score
            combined_score = transformer_weight * ((predicted_price - last_price) / last_price) + \
                            xgb_weight * xgb_return
            
            adjusted_score = combined_score * (1 + 0.1 * sentiment_factor)

            print(f"[{symbol}] LSTM Weight: {transformer_weight:.2f}, XGB Weight: {xgb_weight:.2f}, Combined Score: {combined_score:.4f}")



            # Entry logic
            if cash > last_price:

                #debug manual overide
                debug_override = False  # set False to disable manual sentiment override
                if debug_override and symbol == "AAPL":
                    sentiment = "negative"
                    print(f"[OVERRIDE] Forced sentiment for {symbol} to 'negative' for testing.")


                action = None
            
           # Define returns first
            returns = bars["close"].pct_change().dropna()

            # Calculate volatility from returns
            volatility_factor = 1.5 if probability > 0.7 else 2.5
            volatility = returns.rolling(window=10).std().iloc[-1]
            if isinstance(volatility, pd.Series):
                volatility = volatility.item()  # flatten to scalar if needed
            volatility = min(float(volatility), 0.03)  # ✅ this is safe

            threshold = max(0.001, 0.3 * volatility)
            if adjusted_score > threshold:
                action = "buy"
            elif adjusted_score < -threshold:
                action = "sell"

            if action == "buy":
                returns = bars["close"].pct_change().dropna()

                volatility_factor = 1.5 if probability > 0.7 else 2.5
                volatility = returns.rolling(window=10).std().iloc[-1]
                if isinstance(volatility, pd.Series):
                    volatility = volatility.item()  # flatten to scalar if needed
                volatility = min(float(volatility), 0.03)  # ✅ this is safe

                tp = last_price * (1 + volatility_factor * volatility)
                sl = last_price * (1 - volatility_factor * volatility)

                order = self.create_order(
                    str(symbol),
                    quantity,
                    "buy",
                    type="bracket",
                    take_profit_price=tp,
                    stop_loss_price=sl
                )
                self.submit_order(order)
        
                self.last_trade[f"{symbol}_time"] = now
                self.trade_outcomes[symbol] = order

            elif action == "sell" and position is not None and position.quantity > 0:
                returns = bars["close"].pct_change().dropna()
                volatility_factor = 1.5 if probability > 0.7 else 2.5
                volatility = returns.rolling(window=10).std().iloc[-1]
                if isinstance(volatility, pd.Series):
                    volatility = volatility.item()  # flatten to scalar if needed
                volatility = min(float(volatility), 0.03)  # ✅ this is safe

                tp = last_price * (1 + volatility_factor * volatility)
                sl = last_price * (1 - volatility_factor * volatility)

                order = self.create_order(
                    str(symbol),
                    quantity,
                    "sell",
                    type="bracket",
                    take_profit_price=tp,
                    stop_loss_price=sl
                )
                self.submit_order(order)
          
                self.last_trade[f"{symbol}_time"] = now
                self.trade_outcomes[symbol] = order

            else:
                print(f"[{symbol}] Skipped - combined_score not significant.")
                print(f"[DEBUG] Sentiment: {sentiment}, Prob: {probability:.3f}, Score: {combined_score:.4f}")


        # ✅ Evaluate active positions for early exit if sentiment/model reverses
        for symbol in self.symbols:
            position = self.get_position(symbol)
            if position is not None and position.quantity > 0:
                bars = self.get_historical_data(symbol)
                if bars is None or len(bars) < 50:
                    continue

                predicted_price = self.get_transformer_prediction(bars)
                if predicted_price is None:
                    continue

                current_price = bars["close"].iloc[-1]
                if isinstance(current_price, pd.Series):
                    current_price = current_price.item() 
                change = float((predicted_price - current_price) / current_price)

                if change < -0.03 and sentiment == "negative":  # Predicting >=3% drop
                    print(f"[{symbol}] 🚨 LSTM predicts drop. Exiting early.")
                    self.submit_order(
                        self.create_order(symbol, position.quantity, "sell", type="market")
                    )
                    self.last_trade[f"{symbol}_time"] = self.get_datetime()

        # ✅ Print final portfolio value AFTER all symbols processed
        portfolio_value = self.get_portfolio_value()
        if not hasattr(self, 'initial_cash'):
            self.initial_cash = portfolio_value

        drawdown = (portfolio_value - self.initial_cash) / self.initial_cash
        if drawdown < -0.30:
            self.log_message("❌ Max drawdown exceeded. Stopping trading.")
            self.stop_trading()
            return

        print(f"✅ Final portfolio value: {portfolio_value:.2f}")
        

   

       

symbols = [Asset("SPY"), Asset("AAPL")]

start_date = pd.Timestamp("2019-10-01").tz_localize(None)
end_date = pd.Timestamp("2024-12-31").tz_localize(None)


data_source = YahooDataBacktesting(
    datetime_start=start_date,
    datetime_end=end_date
)

broker = Alpaca(ALPACA_CREDS, data_source=data_source)

strategy = MLTrader(
    name='mlstrat',
    broker=broker,
    parameters={"symbols": symbols, "cash_at_risk": 0.5}
)

strategy.backtest(
    YahooDataBacktesting,
    start_date,
    end_date
)









# trader = Trader()
# trader.add_strategy(strategy)
# trader.run_all()



# ##thompson sampling
# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
# dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

# import random
# N = 10000
# d = 10
# ads_selected = []
# numbers_of_rewards_1 = [0] * d
# numbers_of_rewards_0 = [0] * d
# total_reward = 0
# for n in range(0, N):
#   ad = 0
#   max_random = 0
#   for i in range(0, d):
#     random_beta = random.betavariate(numbers_of_rewards_1[i] + 1, numbers_of_rewards_0[i] + 1)
#     if (random_beta > max_random):
#       max_random = random_beta
#       ad = i
#   ads_selected.append(ad)
#   reward = dataset.values[n, ad]
#   if reward == 1:
#     numbers_of_rewards_1[ad] = numbers_of_rewards_1[ad] + 1
#   else:
#     numbers_of_rewards_0[ad] = numbers_of_rewards_0[ad] + 1
#   total_reward = total_reward + reward


#    # Optional: use this for live trading or post-backtest evaluation
#     def on_order_filled(self, order):
#         symbol = order.symbol
#         if order.take_profit_hit:
#             self.record_trade_outcome(symbol, 1)
#         elif order.stop_loss_hit:
#             self.record_trade_outcome(symbol, 0)


#tensorflow implementation
# import numpy as np
# import pandas as pd
# import tensorflow as tf

# dataset = pd.read_csv('Churn_Modelling.csv')
# X = dataset.iloc[:, 3:-1].values
# y = dataset.iloc[:, -1].values

# from sklearn.preprocessing import LabelEncoder
# le = LabelEncoder()
# X[:, 2] = le.fit_transform(X[:, 2])

# from sklearn.compose import ColumnTransformer
# from sklearn.preprocessing import OneHotEncoder
# ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
# X = np.array(ct.fit_transform(X))

# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# from sklearn.preprocessing import StandardScaler
# sc = StandardScaler()
# X_train = sc.fit_transform(X_train)
# X_test = sc.transform(X_test)
# # If some parts of the data are really big numbers (like salaries: 100,000), and others are small numbers (like age: 25), the robot might think the big numbers are more important, just because they're bigger.
# # But that’s not true — they're just measured differently.
# # Feature scaling is like telling the robot:
# # "Hey, don’t judge importance by size — let’s shrink everything to the same range so you treat all features fairly."

# ann = tf.keras.models.Sequential()
# ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
# ## how many units/neurons is just anyhow experiment and check accuracy
# ## activation refers to activation function
# ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
# ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
# ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
# ##optimizer - adam is stoachastic gradient descent
# ##loss function  - binary is binary_crossentropy if non binary is categorical_crossentropy and activation not signmoid must be softmax

# ann.fit(X_train, y_train, batch_size = 32, epochs = 100)
# ## default batch size is 32 for batch learning
# print(ann.predict(sc.transform([[1, 0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])) > 0.5)
# ##change the input value to the encoded value eg geography=france change to 1,0,0 and apply feature scaling

# y_pred = ann.predict(X_test)
# y_pred = (y_pred > 0.5)
# print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

# from sklearn.metrics import confusion_matrix, accuracy_score
# cm = confusion_matrix(y_test, y_pred)
# print(cm)
# accuracy_score(y_test, y_pred)

# xgboost implementation
# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
# dataset = pd.read_csv('Data.csv')
# X = dataset.iloc[:, :-1].values
# y = dataset.iloc[:, -1].values
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
# from xgboost import XGBClassifier
# classifier = XGBClassifier()
# classifier.fit(X_train, y_train)
# ##if using for regression models import XGBCRegressor
# from sklearn.metrics import confusion_matrix, accuracy_score
# y_pred = classifier.predict(X_test)
# cm = confusion_matrix(y_test, y_pred)
# print(cm)
# accuracy_score(y_test, y_pred)
# from sklearn.model_selection import cross_val_score
# accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
# print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
# print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))