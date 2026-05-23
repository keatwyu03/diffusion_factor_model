from fredapi import Fred
import pandas as pd
import numpy as np
import torch
import yfinance as yf

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


sp500_log = np.log(sp500 / sp500.shift(1))
baa_log = np.log(data['baa'] / data['baa'].shift(1))


unemp_monthly = data['unemp']
unemp_flag = (unemp_monthly.diff(1).abs() >= unemployment_threshold).astype(float).fillna(0.0)

baa_monthly = data['baa']
baa_flag = (baa_log.abs() >= baa_threshold).astype(float).fillna(0.0)

df = pd.DataFrame({"sp500" : sp500_log,
})


df['unemp'] = unemp_monthly.reindex(df.index, method = 'ffill')
df['unemp_flag'] = unemp_flag.reindex(df.index, method = 'ffill')
df['baa'] = baa_log.reindex(df.index, method = 'ffill')
df['baa_flag'] = baa_flag.reindex(df.index, method = 'ffill')

df = df[['unemp', 'unemp_flag', 'sp500', 'baa', 'baa_flag']]
df = df.dropna()

df.to_csv("explore/macro_data_new.csv", index_label = "Date")

print(len(df[df['unemp_flag'] == 1]))
print(len(df[df['baa_flag'] == 1]))
print(len(df))
print(df['unemp'].std())


# df_interp = pd.DataFrame({"sp500": sp500_log})
# df_interp['unemp'] = unemp_monthly.reindex(df_interp.index).interpolate(method='spline', order  = 3)
# df_interp['unemp_flag'] = unemp_flag.reindex(df_interp.index).interpolate(method='spline', order = 3)
# df_interp['baa'] = baa_log.reindex(df_interp.index).interpolate(method='spline', order = 3)
# df_interp['baa_flag'] = baa_flag.reindex(df_interp.index).interpolate(method='spline', order = 3)
# df_interp = df_interp[['unemp', 'unemp_flag', 'sp500', 'baa', 'baa_flag']]
# df_interp = df_interp.dropna()

# df_interp.to_csv("explore/macro_data_interp.csv", index_label = "Date")

# print("\n--- Interpolated version ---")
# print(len(df_interp[df_interp['unemp_flag'] >= 0.5]))
# print(len(df_interp[df_interp['baa_flag'] >= 0.5]))
# print(len(df_interp))
# print(df_interp['unemp'].std())