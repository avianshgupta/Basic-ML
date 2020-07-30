import pandas as pd
import quandl, math, datetime
import numpy as np
from sklearn import preprocessing,svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import pickle

style.use('ggplot')
# command to display graph in xming  ' export DISPLAY=localhost:0.0 '

df = quandl.get('WIKI/GOOGL')
df = df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume']]
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100.0
df['PCT_Change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0

df = df[['Adj. Close','HL_PCT','PCT_Change','Adj. Volume']]     #Using only the required features

forecast_col = 'Adj. Close'
df.fillna(-99999,inplace = True)                                #replacing the NaN values with -99999

forecast_out = int(math.ceil(0.0029*len(df)))                     #creating the shift value for predicting future prices
print(forecast_out)
df['label'] = df[forecast_col].shift(-forecast_out)             #creating the label column and shifting it up so it shows future values of Adj. Close
#df.dropna(inplace = True)

X = np.array(df.drop(['label'],1))
#print(X)
#y = np.array(df['label'])
#print(y)
X = preprocessing.scale(X)
print('\n--------------------------------------------------------------------------------------------------\nData Frames:\n')
print(df)
print('\n--------------------------------------------------------------------------------------------------\nX before:\n')
print(X)
print('\n--------------------------------------------------------------------------------------------------\n\n')
X_lately = X[-forecast_out:]
X = X[:-forecast_out]
print('\n--------------------------------------------------------------------------------------------------\nX:\n')
print(X)
print('\n--------------------------------------------------------------------------------------------------\nX_lately:\n')
print(X_lately)
print('\n--------------------------------------------------------------------------------------------------\n')
df.dropna(inplace = True)
y = np.array(df['label'])

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2)

#clf = LinearRegression()                                                    ######################################
#clf.fit(X_train,y_train)                                                    # Training the classifier and saving #
#with open('linearregression.pickle','wb') as f:                             # it to file linearregression.pickle #
#    pickle.dump(clf,f)                                                      ######################################

pickle_in = open('linearregression.pickle','rb')                             # opening the trained classifier and
clf = pickle.load(pickle_in)                                                 # loading it to clf.
accuracy = clf.score(X_test,y_test)
#print(accuracy)

forecast_set = clf.predict(X_lately)
print(forecast_set,"\n",accuracy,"\n",forecast_out)
df['Forecast'] = np.nan

last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]
print('----------------------------------------------')
print(df.head)
print('----------------------------------------------')
print(df.tail)
print('----------------------------------------------')

df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc = 4)
plt.xlabel('Date')
plt.ylabel('Price')
#plt.savefig('graph.png')
plt.show()