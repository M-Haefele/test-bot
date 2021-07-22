#SellHighBuyLow
#Algorithmic practice for trading based on market data trends


import os  
import pandas_datareader as pdr 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas_datareader import data

start_date = '2020-01-01'
end_date = '2021-01-01'


def query_financials(start_date, end_date):
    return data.DataReader('BTC-USD', 'yahoo', start_date, end_date)

def market_signal(google_data):
    google_data_signal = pd.DataFrame(index=google_data.index)
    google_data_signal['price'] = google_data['Adj Close']
    google_data_signal['signal'] = 0.0
    google_data_signal['daily_difference'] = google_data_signal['price'].diff()
    google_data_signal['signal'] = np.where(google_data_signal['daily_difference'] < 0 , -1.0, 1) 
    return google_data_signal

def visualize(google_data_signal):
    fig = plt.figure()
    ax1 = fig.add_subplot(111, ylabel='BTC-USD price in $')
    google_data_signal['price'].plot(ax=ax1, color='b', lw=2)
    ax1.plot(google_data_signal.loc[google_data_signal.signal == 1.0].index, google_data_signal.price[google_data_signal.signal == 1.0], '^', markersize=5, color='m')
    ax1.plot(google_data_signal.loc[google_data_signal.signal == -1.0].index, google_data_signal.price[google_data_signal.signal == -1.0], 'v', markersize=5, color='k')
    plt.show()

google_data = query_financials(start_date, end_date)
google_data_signal = market_signal(google_data)
visualize(google_data_signal)

