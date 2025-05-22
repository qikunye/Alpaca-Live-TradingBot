from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd

# Load and preprocess historical data
data = pd.read_csv('Download Data - FUND_US_ARCX_SPY.csv')  # contains ['open', 'high', 'low', 'close', 'volume']
# Rename columns to lowercase for consistency (optional, but helps with code readability)
data.rename(columns={
    'Open': 'open',
    'High': 'high',
    'Low': 'low',
    'Close': 'close',
    'Volume': 'volume'
}, inplace=True)

# Remove thousands separators and convert 'volume' to integer
data['volume'] = data['volume'].str.replace(',', '').astype(int)

# Convert price columns to float (if they're not already)
for col in ['open', 'high', 'low', 'close']:
    data[col] = data[col].astype(float)

# Select only the required columns
data = data[['open', 'high', 'low', 'close', 'volume']]

# (Optional) Reverse the DataFrame to ensure oldest data is first (if needed)
data = data.iloc[::-1].reset_index(drop=True)

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# Create input sequences
def create_dataset(data, lookback=30):
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i-lookback:i])
        y.append(data[i, 3])  # Close price is at index 3
    return np.array(X), np.array(y)

X, y = create_dataset(scaled_data)

# Train-test split
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Define LSTM model
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    LSTM(64),
    Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=20, batch_size=32)

# Save model and scaler
model.save('lstm_price_predictor.keras')
import joblib
joblib.dump(scaler, 'lstm_scaler.pkl')
