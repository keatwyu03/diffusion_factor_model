from fredapi import Fred
import pandas as pd
import numpy as np
import torch
import yfinance as yf
import os

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config as cfg

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


baa_threshold = 0.05
baa_log = np.log(data['baa'] / data['baa'].shift(1))

cond_key = cfg.COND_EVENT.lower()
cond_series = data[cond_key]
cond_diff = cond_series.diff(1)
cond_flag = (cond_diff.abs() >= cfg.H_EVENT_THRESHOLD).astype(float).fillna(0.0)

tickers = ["AAPL", "ORCL", "MSFT", "IBM"]
df = yf.download(tickers, start = "1950-01-01", auto_adjust=True)["Close"]
log_ret = np.log(df / df.shift(1)).dropna()

baa_flag = (baa_log.abs() >= baa_threshold).astype(float).fillna(0.0)

df[cfg.COND_EVENT]            = cond_diff.reindex(df.index, method='ffill')
df[f"{cfg.COND_EVENT}_flag"]  = cond_flag.reindex(df.index, method='ffill')
df['baa']                     = baa_log.reindex(df.index, method='ffill')
df['baa_flag']                = baa_flag.reindex(df.index, method='ffill')

df_out = pd.DataFrame({
    cfg.COND_EVENT:            df[cfg.COND_EVENT],
    f"{cfg.COND_EVENT}_flag":  df[f"{cfg.COND_EVENT}_flag"],
    "baa":                     df["baa"],
    "baa_flag":                df["baa_flag"],
    "AAPL":                    log_ret["AAPL"],
    "ORCL":                    log_ret["ORCL"],
    "MSFT":                    log_ret["MSFT"],
    "IBM":                     log_ret["IBM"],
})

df_out = df_out.dropna()
df_out.to_csv("explore/macro_data_new.csv", index_label="Date")

print(f"{cfg.COND_EVENT} flag count (|change| > {cfg.H_EVENT_THRESHOLD}): {int(df[f'{cfg.COND_EVENT}_flag'].sum())}")
print(f"baa flag count: {int(df['baa_flag'].sum())}")
print(f"total rows: {len(df)}")

# for tick in tickers: 
#     print(tick, len(df[tick]))
#     print(df[tick].isnull().sum())

# print(len(df_out))
# print(df_out.isnull().sum())
