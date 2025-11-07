# Ayush Jain
# INST 414
# Section 0101
# Module 4 Assignment

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt 
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

sp500 = pd.read_csv('/Users/ayushjain/Downloads/S&P500 Stocks - Sheet1.csv')

tickers = sp500['Ticker'].tolist()
# replace '.' with '-' for tickers like BRK.B to accommodate yfinance format
tickers = [t.replace('.', '-') for t in tickers]
ticker_to_name = dict(zip(sp500['Ticker'], sp500['Company']))

data = []

# getting data and checking for errors
for t in tickers:
    try:
        ticker_obj = yf.Ticker(t)
        info = ticker_obj.info
        market_cap = info.get('marketCap')
        beta = info.get('beta')
        dividend_yield = info.get('dividendYield', 0)  # default 0 if missing
        
        # one year return
        hist = ticker_obj.history(period="1y")
        if len(hist) > 0:
            start_price = hist['Close'].iloc[0]
            end_price = hist['Close'].iloc[-1]
            one_year_return = (end_price - start_price) / start_price
        else:
            one_year_return = 0
        
        data.append({
            'Ticker': t,
            'Company': ticker_to_name.get(t.replace('-', '.'), None),
            'MarketCap': market_cap,
            'Beta': beta,
            'DividendYield': dividend_yield,
            'OneYearReturn': one_year_return
        })
        
    except Exception as e:
        print(f"Error getting data for {t}: {e}")


df = pd.DataFrame(data)

# cleaning data
df['DividendYield'] = df['DividendYield'].fillna(0)
df.dropna(inplace=True)

df.reset_index(drop=True, inplace=True)

features = df[['MarketCap', 'Beta', 'DividendYield', 'OneYearReturn']]

scaler = StandardScaler()
scaled_data = scaler.fit_transform(features)

# elbow method
inertia = []
K_range = range(1, 11)

for k in K_range:
    model = KMeans(n_clusters=k, random_state=42)
    model.fit(scaled_data)
    inertia.append(model.inertia_)

plt.figure(figsize=(8,5))
plt.plot(K_range, inertia, marker='o')
plt.title('Elbow Method for k Selection')
plt.xlabel('k - Number of Clusters')
plt.ylabel('Inertia')
plt.show()


# kmeans and clustering
k = 4 
kmeans = KMeans(n_clusters=k, random_state=42)
df['Cluster'] = kmeans.fit_predict(scaled_data)

cluster_summary = df.groupby('Cluster')[['MarketCap', 'Beta', 'DividendYield', 'OneYearReturn']].mean()
print("Cluster Summary:\n", cluster_summary)

for cluster in range(k):
    print(f"\nCluster {cluster} examples:")
    examples = df[df['Cluster']==cluster][['Ticker','Company','MarketCap', 'Beta', 'DividendYield', 'OneYearReturn']].head(5)
    print(examples)