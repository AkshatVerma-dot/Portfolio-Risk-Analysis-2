# -*- coding: utf-8 -*-
"""
Created on Tue Jan 20 10:18:30 2026

@author: Lenovo
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
print('Environment Ready')
import os
os.makedirs("data",exist_ok=True)
os.makedirs("outputs",exist_ok=True)
os.makedirs("charts",exist_ok=True)

tickers={"US_Equity":"SPY","HK_Equity":"^HSI","Bonds":"TLT","Gold":"GLD","USD_HKD":"USDHKD=X"}
start_date= '2019-01-01'
end_date='2025-01-01'
prices= yf.download(list(tickers.values()),start=start_date,end=end_date,auto_adjust=False)
prices=prices["Adj Close"]
prices.columns=tickers.keys()
prices.to_csv("data/raw_prices.csv")
print(prices.head())
prices=prices.fillna(method="ffill")
prices=prices.dropna()
print("\nMissing Values After Cleaning:")
print(prices.isna().sum())
returns=prices.pct_change().dropna()
returns.to_csv("data/daily_returns.csv")
print("\nDaily returns preview:")
print(returns.head())
summary= pd.DataFrame({"Mean Return":returns.mean(),"Volatility":returns.std()})
print("\nReturn Summary")
print(summary)

weights=pd.Series({"US_equity":0.35,"HK_Equity":0.25,"Bonds":0.20,"Gold":0.10,"USD_HKD":0.10})
weights=weights/weights.sum()
print("\nPortfolio Weights:")
print(weights)
print("Sum of Weights:",weights.sum())
weights=weights.reindex(returns.columns).fillna(0)
portfolio_returns= returns@weights
portfolio_returns_clean=portfolio_returns.dropna()
print("\nPortfolio daily returns:")
print(portfolio_returns.head())

cov_matrix= returns.cov()
print("\nCovariance Matrix:")
print(cov_matrix)
portfolio_volatility=np.sqrt(np.dot(weights.T,np.dot(cov_matrix,weights)))
print("\Portfolio Volatility:")
print(portfolio_volatility)
corr=returns.corr()

plt.figure(figsize=(8,6))
plt.imshow(corr,cmap="coolwarm")
plt.colorbar()
plt.xticks(range(len(corr)),corr.columns,rotation=45)
plt.yticks(range(len(corr)),corr.columns)
plt.title("Asset Correlation Mtrix")
plt.tight_layout()
plt.show()


confidence_level=0.95
historical_var= np.percentile(portfolio_returns,(1-confidence_level))*100
print("Historical 95% VaR:",historical_var)

cumilative_returns=(1+portfolio_returns_clean).cumprod()
rolling_max=portfolio_returns_clean.cummax()
drawdown=(cumilative_returns-rolling_max)/rolling_max
max_drawdown=drawdown.min()
print("Maximum Drawdown:",max_drawdown)

plt.figure(figsize=(10,5))
plt.plot(drawdown,label="Drawdown")
plt.axhline(0,color='black',linewidth=0.8)
plt.title("Portfolio Drawdown Over Time")
plt.ylabel("Drawdown")
plt.xlabel("Date")
plt.legend()
plt.tight_layout()
plt.show()

shocks={"Normal":0.5,"-5% Shock":0.05,"-10% Shock":0.10}
stress_results=[]
for scenario, shock in shocks.items():
    stressed_returns= portfolio_returns_clean*(1+shock)
    
    var_95=np.percentile(stressed_returns,5)
    mean_ret=stressed_returns.mean()
    
    stress_results.append({"Scenario":scenario,"Mean Return":mean_ret,"VaR 95%":var_95})
    stress_df=pd.DataFrame(stress_results)
print(stress_df)    
stress_df.to_excel("VaR_scenarios.xlsx",index=False)

