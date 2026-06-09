from fredapi import Fred
import pandas as pd
import numpy as np
import torch
import yfinance as yf
import os

fred = Fred(api_key = '6dac8927ae66be817978bd55e16a9241')
sp500 = yf.download('^GSPC', start = '1950-01-01')['Close'].squeeze()


data = {
    'unemp': fred.get_series('UNRATE'),
    'cpi': fred.get_series('CPIAUCSL'),
    'gdp': fred.get_series('GDP'),
    'spread': fred.get_series('T10Y2Y'),
    'sp500': fred.get_series('SP500'),
    'vix': fred.get_series('VIXCLS'),
    'baa': fred.get_series('BAA'),
    'aaa': fred.get_series('AAA'),
}


unemployment_threshold = 0.3
baa_threshold = 0.05
baa_log = np.log(data['baa'] / data['baa'].shift(1))

unemp_monthly = data['unemp']
unemp_flag = (unemp_monthly.diff(1).abs() >= unemployment_threshold).astype(float).fillna(0.0)

tickers = ["AAPL", "AMZN", "TSLA", "MSFT", "META"]
df = yf.download(tickers, start = "1950-01-01", auto_adjust=True)["Close"]
log_ret = np.log(df / df.shift(1)).dropna()


baa_monthly = data['baa']
baa_flag = (baa_log.abs() >= baa_threshold).astype(float).fillna(0.0)


df['unemp'] = unemp_monthly.reindex(df.index, method = 'ffill')
df['unemp_flag'] = unemp_flag.reindex(df.index, method = 'ffill')

df["cpi"] = data["cpi"].reindex(df.index, method = 'ffill')

df['baa'] = baa_log.reindex(df.index, method = 'ffill')
df['baa_flag'] = baa_flag.reindex(df.index, method = 'ffill')


df_out = pd.DataFrame({
    "unemp":     df["unemp"],
    "unemp_flag": df["unemp_flag"],
    "baa":        df["baa"],
    "baa_flag":   df["baa_flag"],
    "cpi":        df["cpi"],
    "AAPL":       log_ret["AAPL"],
    "AMZN":       log_ret["AMZN"],
    "TSLA":       log_ret["TSLA"],
    "MSFT":       log_ret["MSFT"],
    "META":       log_ret["META"],
})

df_out = df_out.dropna()
df_out.to_csv("explore/macro_data_new.csv", index_label="Date")


print(len(df[df['unemp_flag'] == 1]))
print(len(df[df['baa_flag'] == 1]))
print(len(df))
print(df['unemp'].std())