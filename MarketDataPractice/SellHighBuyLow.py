#SellHighBuyLow
#Algorithmic practice for trading based on market data trends


import os  
import pandas_datareader as pdr 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statistics as stat
from sklearn.linear_model import LinearRegression

from pandas_datareader import data
from math import sqrt

start_date = '2020-01-01'
end_date = '2021-01-01'

price_points = []
local_min = []
local_mix = []
support = []
resistance = []

def query_financials(start_date, end_date):
    return data.DataReader('BTC-USD', 'yahoo', start_date, end_date)

def market_signal(google_data):
    global price_points
    google_data_signal = pd.DataFrame(index=google_data.index)
    google_data_signal['price'] = google_data['Adj Close']
    google_data_signal['signal'] = 0.0
    google_data_signal['daily_difference'] = google_data_signal['price'].diff()
    google_data_signal['signal'] = np.where(google_data_signal['daily_difference'] < 0 , -1.0, 1)
    for row in google_data_signal['price']:
        price_points.append(row)
    return google_data_signal

def visualize(google_data_signal, support, resistance):
    fig = plt.figure()
    ax1 = fig.add_subplot(111, ylabel='BTC-USD price in $')
    google_data_signal['price'].plot(ax=ax1, color='b', lw=2)
    ax1.plot(google_data_signal.loc[google_data_signal.signal == 1.0].index, google_data_signal.price[google_data_signal.signal == 1.0], '^', markersize=5, color='m')
    ax1.plot(google_data_signal.loc[google_data_signal.signal == -1.0].index, google_data_signal.price[google_data_signal.signal == -1.0], 'v', markersize=5, color='k')
    plt.show()


def init():
    #dataframe creation
    google_data = query_financials(start_date, end_date)
    google_data_signal = market_signal(google_data)
    #technical analysis generation
    local_min_max(price_points)
    print(local_min)
    print(local_max)
    local_min_slope, local_min_int = regression_ceof(local_min)
    local_max_slope, local_max_int = regression_ceof(local_max)
    #visualize data
    #y=mx+b incorporates x as a series of financial data...
    visualize(google_data_signal, support, resistance)

#https://python.plainenglish.io/estimate-support-and-resistance-of-a-stock-with-python-beginner-algorithm-f1ae1508b66d 
#tutorial on local min and max	
def pythag(pt1, pt2):
    a_sq = (pt2[0] - pt1[0]) ** 2
    b_sq = (pt2[1] - pt1[1]) ** 2
    return sqrt(a_sq + b_sq)

def regression_ceof(pts):
    X = np.array([pt[0] for pt in pts]).reshape(-1, 1)
    y = np.array([pt[1] for pt in pts])
    model = LinearRegression()
    model.fit(X, y)
    return model.coef_[0], model.intercept_


def local_min_max(pts):
    global local_min
    global local_max
    local_min = []
    local_max = []
    prev_pts = [(0, pts[0]), (1, pts[1])]
    for i in range(1, len(pts) - 1):
        append_to = ''
        if pts[i-1] > pts[i] < pts[i+1]:
            append_to = 'min'
        elif pts[i-1] < pts[i] > pts[i+1]:
            append_to = 'max'
        if append_to:
            if local_min or local_max:
                prev_distance = pythag(prev_pts[0], prev_pts[1]) * 0.5
                curr_distance = pythag(prev_pts[1], (i, pts[i]))
                if curr_distance >= prev_distance:
                    prev_pts[0] = prev_pts[1]
                    prev_pts[1] = (i, pts[i])
                    if append_to == 'min':
                        local_min.append((i, pts[i]))
                    else:
                        local_max.append((i, pts[i]))
            else:
                prev_pts[0] = prev_pts[1]
                prev_pts[1] = (i, pts[i])
                if append_to == 'min':
                    local_min.append((i, pts[i]))
                else:
                    local_max.append((i, pts[i]))

init()

