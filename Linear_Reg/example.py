from sklearn import preprocessing,svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style

X = [[1,2],[2,3],[3,4],[4,5],[5,6],[6,7]]

y = [0,1,2,3,4,5]

X_train,X_test,y_train,y_test = train_test_split(X,y)
print(X_train)
print(y_train)
print(X_test)
print(y_test)
clf = LinearRegression()
clf.fit(X_train,y_train)
accuracy = clf.score(X_test,y_test)
print(accuracy)
