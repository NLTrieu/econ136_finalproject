import numpy as np
import pandas as pd
import datetime
from datetime import date
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import seaborn as sns
import os

for stocksym in ['SPY', 'MSFT', 'INTC', 'AMZN', 'AAPL', 'AMD', 'TSLA']:
    # get data
    df_close = pd.DataFrame()
    df_temp = pd.read_json('https://api.iextrading.com/1.0/stock/'+stocksym+'/chart/5y')
    # transform data to look more like it did from quandl to make reuse of old code easier
    stock = df_temp[['date', 'volume','close']]
    stock = stock.rename(index=str, columns={'date': 'Date', 'volume': 'Volume', 'close': 'Close'})

    # lets try simple momentum strategy. if it goes up 2 previous days we buy
    two_ago = float(stock.Close[-255])
    one_ago = float(stock.Close[-254])
    todaysClose = float(stock.Close[-253])
    buy = one_ago < two_ago and todaysClose < one_ago
    profit = 0.0
    returns = [0]
    baseline = [0]
    for i in reversed(range(1, 252)):
        newClose = float(stock.Close[-i])
        if buy:
            profit += newClose - todaysClose
        returns.append(profit)
        baseline.append(newClose - stock.Close[-252])

        # adjust variables appropriately
        two_ago = one_ago
        one_ago = todaysClose
        todaysClose = newClose
        buy = one_ago > two_ago and todaysClose > one_ago

    print("Strategy profit: ", profit)
    print("Buy and hold: ", float(stock.Close[-1] - stock.Close[-252]))

    plt.plot([i for i in range(len(returns))], returns)
    plt.plot([i for i in range(len(baseline))], baseline)
    plt.xlabel('Days')
    plt.ylabel('Return')
    plt.title(stocksym)
    plt.show()
