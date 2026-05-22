import pandas as pd
import yfinance as yf
import pandas_market_calendars as mcal
from datetime import datetime
import pytz
import time
import os
import sys

ny_tz = pytz.timezone('America/New_York')
ny_time = datetime.now(ny_tz)
is_dst = ny_time.dst().seconds != 0

if not is_dst:
    time.sleep(3600)

base_dir = os.path.dirname(os.path.abspath(__file__))
output_filename = os.path.join(base_dir, 'master_regular_close.csv')
today_date = ny_time.strftime('%Y-%m-%d')

if os.path.exists(output_filename):
    existing_df = pd.read_csv(output_filename, index_col=0)
    if today_date in existing_df.index and not existing_df.loc[today_date].isnull().all():
        sys.exit(0)

tickers = ['QQQ', 'TQQQ', 'SOXL', 'TECL', 'SGOV']
df_list = []

for ticker in tickers:
    for attempt in range(3):
        try:
            temp_data = yf.download(ticker, period="max", auto_adjust=False, ignore_tz=True)['Close']
            if not temp_data.empty:
                temp_data.name = ticker
                df_list.append(temp_data)
                break
        except:
            time.sleep(5)
    else:
        sys.exit(1)

data = pd.concat(df_list, axis=1)
data = data.loc['2010-01-01':]
data = data.ffill().round(2)
data.index = pd.to_datetime(data.index).normalize()
data.dropna(how='all', inplace=True)
data = data[tickers]

if data.iloc[-1].isnull().any():
    sys.exit(1)

nyse = mcal.get_calendar('NYSE')
today = pd.Timestamp(ny_time.date())
end_date_for_cal = today + pd.Timedelta(days=100)

valid_days = nyse.valid_days(start_date='2010-01-01', end_date=end_date_for_cal)
valid_days = pd.to_datetime(valid_days).tz_localize(None).normalize()

past_idx = valid_days[valid_days <= today]
future_idx = valid_days[valid_days > today][:50]
full_idx = past_idx.union(future_idx)

data = data.reindex(full_idx)
data.index = data.index.strftime('%Y-%m-%d')

usdkrw_data = yf.download('KRW=X', period='1d', progress=False)
current_rate = round(float(usdkrw_data['Close'].iloc[-1]), 2)

data.index.name = str(current_rate)
data.to_csv(output_filename)
