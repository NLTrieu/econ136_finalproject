import numpy as np
import pandas as pd
import datetime
from datetime import date
import matplotlib.pyplot as plt
import random
import econ_cnn
import json

# get data
stocksym = 'INTC'
df_temp = pd.read_json('https://api.iextrading.com/1.0/stock/'+stocksym+'/chart/5y')

# Sample of features
#change changeOverTime  changePercent   close   date        high    label       low     open    unadjustedVolume    volume  vwap
#0.162491   0.000000    0.800        20.4828    2013-04-30  20.5255 Apr 30, 13  20.2177 20.2861 41468253        41468253    20.4012

col_names = ["change","cOT","cP","close","date","high","label","low","open","uVol","vol","vwap"]
full_data = df_temp.values

df_temp_short = df_temp[['change','changeOverTime', 'changePercent','close','high','low','open','vwap']]
short_data = df_temp_short.values

for days_past in [8,32]:
	data = [{"id":full_data[i][6],"features":short_data[i:i+days_past],"label":1 if short_data[i+days_past+1][0]>0 else 0} for i in range(len(short_data)-days_past-1)]
	#print(data[0]["features"])
	data_tup = [[x["features"],x["label"],x["id"]] for x in data]
	#random.shuffle(data)
	train = data_tup[:int(len(data_tup)/2)]
	test = data_tup[int(len(data_tup)/2):]
	probs,train_accs,test_accs,up_accs,down_accs = econ_cnn.cnn_traintest(data_tuple={"train":train,"test":test},specs={"num_epochs":500,"days_past":days_past},verbose=True)
	#print(results)

	plt.plot(train_accs,test_accs)
	plt.title('Model Training Over Epochs')
	plt.xlabel('Training Accuracy')
	plt.ylabel('Testing Accuracy')
	plt.savefig("./econ136_SPY_traintest" + str(days_past) + ".png")