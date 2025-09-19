# Ayush Jain
# INST 414
# Section 0101
# Module 1 Assignment

import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np


""" 
Downloding SPY (S&P 500 ETF) & VIX data for the past 10 years and 
calculating the daily returns for SPY within that time frame
"""
spy = yf.download("SPY", start="2000-01-01", end="2025-01-01", auto_adjust=False)
vix = yf.download("^VIX", start="2000-01-01", end="2025-01-01", auto_adjust=False)

spy['Daily_Return'] = spy['Adj Close'].pct_change()

# Merging the 2 data sets and getting rid of nulls
data = pd.merge(spy[['Daily_Return']], vix[['Adj Close']], 
                left_index=True, right_index=True, how='inner')
data.rename(columns={'Adj Close':'VIX'}, inplace=True)
data.dropna(inplace=True)

# Printing the first few rows of data and the correlation matrix between the two tickers
print(data.head())
print(data.corr())

# Scatterplot
plt.scatter(data['VIX'], data['Daily_Return'], alpha=0.5)
plt.xlabel("VIX")
plt.ylabel("SPY Daily Return")
plt.title("VIX vs SPY Daily Returns")
plt.grid(True, linestyle="--", alpha=0.5)
plt.show()