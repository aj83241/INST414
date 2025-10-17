# Ayush Jain
# INST 414
# Section 0101
# Module 3 Assignment

import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import DistanceMetric, pairwise_distances
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

sp500 = pd.read_csv('/Users/ayushjain/Downloads/S&P500 Stocks - Sheet1.csv')

tickers = sp500['Ticker'].tolist()
# replace '.' with '-' for tickers like BRK.B to accommodate yfinance format
tickers = [t.replace('.', '-') for t in tickers]
ticker_to_name = dict(zip(sp500['Ticker'], sp500['Company']))

data = []

# getting data and checking for errors
for t in tickers:
    try:
        info = yf.Ticker(t).info
        data.append({
            'Ticker': t,
            'Company': ticker_to_name.get(t, None),
            'Sector': info.get('sector'),
            'MarketCap': info.get('marketCap'),
            'TrailingPE': info.get('trailingPE'),
            'ForwardPE': info.get('forwardPE'),
            'ROE': info.get('returnOnEquity'),
            'RevenueGrowth': info.get('revenueGrowth')
        })
    except Exception as e:
        print(f"Error downloading data for {t}: {e}")

df = pd.DataFrame(data)
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

numeric_features = ['MarketCap','TrailingPE','ForwardPE','ROE','RevenueGrowth']

# scaling and getting rid of nulls
X = df[numeric_features].replace([np.inf, -np.inf], np.nan).dropna()
X_scaled = StandardScaler().fit_transform(X)

# cosine similarity matrix
similarity_matrix = cosine_similarity(X_scaled)
sim_df = pd.DataFrame(similarity_matrix, index=df['Ticker'], columns=df['Ticker'])

# getting top 10 similar companies
def get_top_similar(ticker, n=10):
    sims = sim_df[ticker].sort_values(ascending=False).iloc[1:n+1]
    result = pd.DataFrame({
        'Ticker': sims.index,
        'Similarity Score': sims.values
    })
    result = result.merge(df[['Ticker', 'Company']], on='Ticker', how='left')
    return result[['Ticker', 'Company', 'Similarity Score']]

# company with highest Market Cap
query_marketcap = df.loc[df['MarketCap'].idxmax(), 'Ticker']
query_marketcap_name = df.loc[df['MarketCap'].idxmax(), 'Company']
print(f"\nTop 10 similar companies to the company with the highest Market Cap: {query_marketcap} - {query_marketcap_name}")
print(get_top_similar(query_marketcap).to_string(index=False))

# company with highest Revenue Growth
query_revenue = df.loc[df['RevenueGrowth'].idxmax(), 'Ticker']
query_revenue_name = df.loc[df['RevenueGrowth'].idxmax(), 'Company']
print(f"\nTop 10 similar companies to the company with the highest Revenue Growth: {query_revenue} - {query_revenue_name}")
print(get_top_similar(query_revenue).to_string(index=False))

# company with highest Trailing PE
query_trailing_pe = df.loc[df['TrailingPE'].idxmax(), 'Ticker']
query_trailing_pe_name = df.loc[df['TrailingPE'].idxmax(), 'Company']
print(f"\nTop 10 similar companies to the company with the highest Trailing PE: {query_trailing_pe} - {query_trailing_pe_name}")
print(get_top_similar(query_trailing_pe).to_string(index=False))

# specific companies - can be any company ticker decided by user
query_specific = 'AAPL'
query_specific_name = df.loc[df['Ticker'] == query_specific, 'Company'].values[0]
print(f"\nTop 10 similar companies to: {query_specific} - {query_specific_name}")
print(get_top_similar(query_specific).to_string(index=False))


print("\n")