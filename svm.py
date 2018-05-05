import quandl
import pandas as pd
import numpy as np
import datetime

from sklearn import metrics
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

sym = "INTC"
data = pd.read_json('https://api.iextrading.com/1.0/stock/'+sym+'/chart/5y')

data1 = pd.DataFrame()

# Shift values by one, so any given days features will consist of values from previous day
# So previous days values can be used to predict current price
data1[['volume']] = data[['volume']].shift(1).fillna(0)
data1[['close']] = data[['close']].shift(1).fillna(0)
data1[['change']] = data[['change']].shift(1).fillna(0)

# Create the labels array. The label is 1 if price is greaeter then previous day and 0 otherwise
y = np.where(data["close"] > data["close"].shift(1), 1, 0)


num_days_back = 8
# Boolean value indicating if stock price was higher or lower i days ago
for i in range(1,6):
    data1["p_" + str(i)] = np.where(data1["close"] > data1["close"].shift(i), 1, 0) # i: number of look back days


X = np.array(data1)

change = np.asarray(data1[['change']])
change = change.reshape(change.shape[0])

close = np.asarray(data1[['close']])
close = close.reshape(close.shape[0])

# Add rsi and stock oscillator features
rsi_arr=[]
stock_osci = []
for i in range(14,len(change)):
	temp = change[i-14:i]
	gain = np.average(temp[temp > 0])
	loss = np.average(temp[temp < 0])
	rsi = 100 - (100.0/(1+(gain/loss)))
	rsi_arr.append(rsi)

	max14 = np.max(temp)
	min14 = np.min(temp)
	osci = 100* ((change[i]-min14)/(max14-min14))
	stock_osci.append(osci)

# Remove first 14 entries
X = X[14:]
y = y[14:]
X = np.column_stack((X,rsi_arr))
X = np.column_stack((X,stock_osci))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, shuffle=False)

# Training
clf = SVC()
clf.fit(X_train,y_train)

# Testing
y_pred = clf.predict(X_test)
y_predtrain = clf.predict(X_train)

print("Test Accuracy: ", metrics.accuracy_score(y_test, y_pred))
print("Train Accuracy: ",metrics.accuracy_score(y_train, y_predtrain))

