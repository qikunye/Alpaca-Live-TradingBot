from lumibot.brokers import Alpaca
from lumibot.backtesting import YahooDataBacktesting
from lumibot.strategies.strategy import Strategy
from lumibot.traders import Trader
from datetime import datetime 
from alpaca_trade_api import REST 
from pandas import Timedelta
from csv_backtesting import CSVDataBacktesting
from finbert_utils import estimate_sentiment
from tensorflow.keras.models import load_model
import joblib
import numpy as np  
from numpy.random import beta
import xgboost as xgb
import pandas as pd
import yaml
from alpaca_trade_api import REST

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



with open("config.yml", "r") as file:
    config = yaml.safe_load(file)

API_KEY = config["alpaca"]["API_KEY"]
API_SECRET = config["alpaca"]["API_SECRET"]
BASE_URL = config["alpaca"].get("BASE_URL", "https://paper-api.alpaca.markets/v2")

ALPACA_CREDS = {
    "API_KEY": API_KEY,
    "API_SECRET": API_SECRET,
    "PAPER": True
}
api = REST(key_id=API_KEY, secret_key=API_SECRET, base_url=BASE_URL)





class MLTrader(Strategy): 
    def initialize(self, symbols=["SPY", "AAPL", "MSFT"], cash_at_risk:float=0.5): 
        self.symbols = symbols
        self.sleeptime = "24H" 
        self.last_trade = {}
        self.cash_at_risk = cash_at_risk
        self.api = REST(base_url=BASE_URL, key_id=API_KEY, secret_key=API_SECRET)

    #LSTM_price prediction
        self.model = load_model("lstm_price_predictor.keras")
        self.scaler = joblib.load("lstm_scaler.pkl")
        self.lookback = 30

     # Thompson Sampling setup
        self.symbol_index = {symbol: i for i, symbol in enumerate(self.symbols)}
        self.wins = [1] * len(self.symbols)
        self.losses = [1] * len(self.symbols)
        self.trade_outcomes = {}  # to store outcome tracking
    #xgboost
        self.xgb_model = xgb.XGBRegressor()
        self.xgb_model.load_model("xgb_bollinger_model.json")
        self.xgb_features = joblib.load("xgb_features.pkl")

    def position_sizing(self,symbol): 
        cash = self.get_cash() 
        last_price = self.get_last_price(symbol)
        if last_price is None:
            self.log_message(f"⚠️ Warning: No last price available for {symbol}")
            return 0, None, 0  # Return something safe to avoid crashing
        quantity = round(cash * self.cash_at_risk / last_price,0)
        return cash, last_price, quantity

    def get_dates(self): 
        today = self.get_datetime()
        three_days_prior = today - Timedelta(days=3)
        return today.strftime('%Y-%m-%d'), three_days_prior.strftime('%Y-%m-%d')

    def get_sentiment(self, symbol): 
        try:
            today, three_days_prior = self.get_dates()
            news = self.api.get_news(symbol=symbol, start=three_days_prior, end=today) 
            news = [ev.__dict__["_raw"]["headline"] for ev in news]
            probability, sentiment = estimate_sentiment(news)
            return probability, sentiment 
        except Exception as e:
            print(f"[{symbol}] Sentiment fetch error: {e}")
            return 0.5, "neutral"

    
    def get_historical_data(self, symbol, limit=60):
        try:
            # In backtesting, use built-in get_historical_data
            if not self.broker.paper_trading:
                # Backtesting
                bars = self.get_historical_prices(symbol, bar_size="1d", length=limit)
                if bars.empty:
                    print(f"[{symbol}] Warning: No bars returned from Alpaca API.")
                    return None
            else:
                bars = self.api.get_bars(symbol, timeframe='1Day', limit=limit, feed='iex').df
                if bars.empty:
                    print(f"[{symbol}] Warning: No bars returned from backtest broker.")
                    return None

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

    def get_lstm_prediction(self, bars):
        df = pd.DataFrame(bars)
        if not all(col in df.columns for col in ['open', 'high', 'low', 'close', 'volume']):
            raise ValueError(f"LSTM input missing required columns. Got: {df.columns.tolist()}")

        df = df[['open', 'high', 'low', 'close', 'volume']].tail(self.lookback)
        if len(df) < self.lookback:
            return None

        scaled = self.scaler.transform(df)
        X_input = np.expand_dims(scaled, axis=0)
        predicted_scaled = self.model.predict(X_input)[0][0]

        reconstructed = np.zeros((1, 5))
        reconstructed[0, 3] = predicted_scaled  # Only filling the 'close' column
        pred_price = self.scaler.inverse_transform(reconstructed)[0][3]
        return pred_price

    
    def get_xgb_prediction(self, bars):
        df = pd.DataFrame(bars)
        print("XGB - DataFrame Columns:", df.columns)

        if not all(col in df.columns for col in ['close', 'volume']):
            raise ValueError(f"Missing required columns for XGB features. Got: {df.columns.tolist()}")

        # Feature engineering (must match training logic)
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['std_20'] = df['close'].rolling(window=20).std()
        df['upper_band'] = df['sma_20'] + 2 * df['std_20']
        df['lower_band'] = df['sma_20'] - 2 * df['std_20']
        df['avg_vol_15'] = df['volume'].rolling(window=15).mean()
        df['return_1d'] = df['close'].pct_change()
        df['close_to_upper'] = (df['close'] - df['upper_band']) / df['upper_band']
        df['close_to_lower'] = (df['close'] - df['lower_band']) / df['lower_band']
        df['vol_to_avgvol'] = df['volume'] / df['avg_vol_15']
        df.dropna(inplace=True)

        if df.empty:
            return None

        latest_features = df[self.xgb_features].iloc[-1]
        prediction = self.xgb_model.predict([latest_features])[0]
        return prediction  # Next-day return





    def on_trading_iteration(self):
        ranked_symbols = self.choose_asset_thompson()
        
        for symbol in ranked_symbols:
            cash, last_price, quantity = self.position_sizing(symbol)
            if last_price is None or quantity == 0:
                continue 
            probability, sentiment = self.get_sentiment(symbol)

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
            predicted_price = self.get_lstm_prediction(bars)
            if predicted_price is None:
                print(f"[{symbol}] Skipping - LSTM prediction failed")
                continue

            print(f"[{symbol}] Sentiment: {sentiment}, Prob: {probability:.3f}, XGB: {xgb_return:.4f}, LSTM: {predicted_price:.2f}, Last: {last_price:.2f}")

          

            # Initialize trade history if not already
            if symbol not in self.last_trade:
                self.last_trade[symbol] = None

            # Entry logic
            if cash > last_price:
                # LONG trade condition
                if (sentiment == "positive" and probability > 0.6 and 
                    xgb_return > 0.001 and predicted_price > last_price * 1.001):
                    
                    if self.last_trade[symbol] == "sell":
                        self.sell_all()

                    order = self.create_order(
                        symbol,
                        quantity,
                        "buy",
                        type="bracket",
                        take_profit_price=last_price * 1.20,
                        stop_loss_price=last_price * 0.95
                    )
                    self.submit_order(order)
                    self.last_trade[symbol] = "buy"
                    self.trade_outcomes[symbol] = order

                # SHORT trade condition
                elif (sentiment == "negative" and probability > 0.6 and 
                    xgb_return < -0.001 and predicted_price < last_price * 0.999):

                    if self.last_trade[symbol] == "buy":
                        self.sell_all()

                    order = self.create_order(
                        symbol,
                        quantity,
                        "sell",
                        type="bracket",
                        take_profit_price=last_price * 0.80,
                        stop_loss_price=last_price * 1.05
                    )
                    self.submit_order(order)
                    self.last_trade[symbol] = "sell"
                    self.trade_outcomes[symbol] = order

symbols = [Asset("SPY"), Asset("AAPL"), Asset("MSFT")]

start_date = pd.Timestamp("2019-10-01").tz_localize(None)
end_date = pd.Timestamp("2024-12-31").tz_localize(None)



broker = Alpaca(ALPACA_CREDS)
strategy = MLTrader(
    name='mlstrat',
    broker=broker,
    parameters={"symbols": symbols, "cash_at_risk": 0.5}
)


# ✅ Pass it into the backtest



strategy.backtest(
    # 🔸 Pass the class itself (not an instance)
    datasource_class=CSVDataBacktesting,

    # 🔸 Pass init arguments using datasource_kwargs
    datasource_kwargs={
    "symbols": [Asset("SPY"), Asset("AAPL"), Asset("MSFT")],
    "datetime_start": datetime(2019, 10, 1),
    "datetime_end": datetime(2024, 12, 31),
    },

    # Other strategy parameters...
    use_cache=False,
    show_progress=True,
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