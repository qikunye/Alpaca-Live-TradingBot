import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_squared_error, make_scorer
import xgboost as xgb
import joblib

# Load and preprocess data
data = pd.read_csv('Download Data - FUND_US_ARCX_SPY.csv')
data.rename(columns={
    'Open': 'open',
    'High': 'high',
    'Low': 'low',
    'Close': 'close',
    'Volume': 'volume'
}, inplace=True)
data['volume'] = data['volume'].str.replace(',', '').astype(int)
for col in ['open', 'high', 'low', 'close']:
    data[col] = data[col].astype(float)
data = data.iloc[::-1].reset_index(drop=True)

# Feature engineering
data['sma_20'] = data['close'].rolling(window=20).mean()
data['std_20'] = data['close'].rolling(window=20).std()
data['upper_band'] = data['sma_20'] + 2 * data['std_20']
data['lower_band'] = data['sma_20'] - 2 * data['std_20']
data['avg_vol_15'] = data['volume'].rolling(window=15).mean()
data['return_1d'] = data['close'].pct_change()
data['close_to_upper'] = (data['close'] - data['upper_band']) / data['upper_band']
data['close_to_lower'] = (data['close'] - data['lower_band']) / data['lower_band']
data['vol_to_avgvol'] = data['volume'] / data['avg_vol_15']
data.dropna(inplace=True)
data['target_return_1d'] = data['return_1d'].shift(-1)
data.dropna(inplace=True)

features = ['close', 'volume', 'sma_20', 'upper_band', 'lower_band', 'avg_vol_15',
            'return_1d', 'close_to_upper', 'close_to_lower', 'vol_to_avgvol']

X = data[features]
y = data['target_return_1d']

# Train-test split (no shuffle for time series)
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

# Define XGBRegressor with default parameters
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)

# Train model
xgb_model.fit(X_train, y_train)

# Predict on test set
y_pred = xgb_model.predict(X_test)
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Test RMSE: {test_rmse:.6f}")

# K-Fold Cross-validation (no shuffle)
kf = KFold(n_splits=10, shuffle=False)
rmse_scorer = make_scorer(mean_squared_error, greater_is_better=False, squared=False)
cv_scores = cross_val_score(xgb_model, X, y, scoring=rmse_scorer, cv=kf, n_jobs=-1)
cv_rmse_scores = -cv_scores  # convert from negative RMSE to positive

print(f"10-Fold CV RMSE Mean: {cv_rmse_scores.mean():.6f}")
print(f"10-Fold CV RMSE Std: {cv_rmse_scores.std():.6f}")

# Save model and features for later use
xgb_model.save_model('xgb_bollinger_model.json')
joblib.dump(features, 'xgb_features.pkl')
