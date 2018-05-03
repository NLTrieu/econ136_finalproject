# lets try modeling for a year
import numpy as np
import pandas as pd
import datetime
from datetime import date
import matplotlib.pyplot as plt
import random
import econ_cnn
import json

# get data
stocksym = 'SPY'
df_temp = pd.read_json('https://api.iextrading.com/1.0/stock/'+stocksym+'/chart/5y')
#change changeOverTime  changePercent   close   date        high    label       low     open    unadjustedVolume    volume  vwap
#0.162491   0.000000    0.800        20.4828    2013-04-30  20.5255 Apr 30, 13  20.2177 20.2861 41468253        41468253    20.4012
col_names = ["change","cOT","cP","close","date","high","label","low","open","uVol","vol","vwap"]
full_data = df_temp.values

df_temp_short = df_temp[['change','changeOverTime', 'changePercent','close','high','low','open','vwap']]
short_data = df_temp_short.values

days_past = 32
data = [{"id":full_data[i][6],"features":short_data[i:i+days_past],"label":1 if short_data[i+days_past+1][0]>0 else 0} for i in range(len(short_data)-days_past-1)]
#print(data[0]["features"])
data_tup = [[x["features"],x["label"],x["id"]] for x in data]
#random.shuffle(data)
train = data_tup[:int(len(data_tup)/2)]
test = data_tup[int(len(data_tup)/2):]
probs,train_accs,test_accs,up_accs,down_accs = econ_cnn.cnn_traintest(data_tuple={"train":train,"test":test},specs={"num_epochs":500},verbose=True)
#print(results)

plt.plot(train_accs,test_accs)
plt.title('Model Training Over Epochs')
plt.xlabel('Training Accuracy')
plt.ylabel('Testing Accuracy')
plt.savefig("./econ136_SPY_traintest" + str(days_past) + ".png")

# with open("econ136results.json",'w') as outfile:

#     json.dump(results,outfile)


# transform data to look more like it did from quandl to make reuse of old code easier
# stock = df_temp[['date', 'volume','close']]
# stock = stock.rename(index=str, columns={'date': 'Date', 'volume': 'Volume', 'close': 'Close'})
# for i in reversed(range(1, 252)):
#     todaysClose = float(stock.Close[-i])
#     daysToExpiry = int(30) # ??? what to set to
#     strikePrice = float(250) # ??? what to set to
#     print(str(stock.Date[-i]).split(" ")[0], " BSP: ", 
#           bsp(days=daysToExpiry, stopr=todaysClose, strike=strikePrice, 
#               rfir=float(0.0250), dayvol=float(calc_volatility_252(endDayOffset=i)))[0])