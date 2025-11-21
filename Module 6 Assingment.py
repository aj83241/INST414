# Ayush Jain
# INST 414
# Section 0101
# Module 6 Assignment

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# collecting and cleaning data
ticker = 'AAPL'
df = yf.download(ticker, start='2023-01-01', end='2025-11-14', progress=False, auto_adjust=True)

df['SMA_5'] = df['Close'].rolling(5).mean()
df['SMA_20'] = df['Close'].rolling(20).mean()
df['Momentum'] = df['Close'].diff()
df['Volatility'] = df['Close'].pct_change().rolling(20).std()
df['Volume_Ratio'] = df['Volume'] / df['Volume'].rolling(5).mean()

df['Target'] = df['Close'].shift(-1)

df = df.dropna()

features = ['SMA_5', 'SMA_20', 'Momentum', 'Volatility', 'Volume_Ratio']
X = df[features]
y = df['Target']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# training and testing
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, shuffle=False, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)

print(f"R² Score: {r2:.4f}")
print(f"RMSE: ${rmse:.2f}")
print(f"MAE: ${mae:.2f}")

# 5 worst predictions
errors = np.abs(y_test.values - y_pred)
worst_idx = np.argsort(errors)[-5:][::-1]

print("\n5 Worst Predictions:")
worst_df = pd.DataFrame({
    'Date': df.index[-len(y_test):][worst_idx],
    'Actual': y_test.values[worst_idx],
    'Predicted': y_pred[worst_idx],
    'Error': y_test.values[worst_idx] - y_pred[worst_idx],
    'Error_Pct': ((y_test.values[worst_idx] - y_pred[worst_idx]) / y_test.values[worst_idx] * 100)
})
print(worst_df)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# actual vs predicted scatter plot
ax1.scatter(y_test, y_pred, alpha=0.6)
ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
ax1.set_xlabel('Actual Price ($)')
ax1.set_ylabel('Predicted Price ($)')
ax1.set_title('Actual vs Predicted Stock Prices')
ax1.grid(True, alpha=0.3)

# feature importance chart
feature_importance = model.feature_importances_
ax2.barh(features, feature_importance)
ax2.set_xlabel('Importance')
ax2.set_title('Feature Importance')
ax2.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.show()