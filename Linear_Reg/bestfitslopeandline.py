from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import random

style.use('fivethirtyeight')

#xs = np.array([1,2,3,4,5,6], dtype=np.float64)
#ys = np.array([5,4,6,5,6,7], dtype=np.float64)
#print(xs)
#print(ys)

def create_dataset(hm, varience, step=2, correlation=False):
    val = 1
    ys = []
    for i in range(hm):
        y = val + random.randrange(-varience,varience)
        ys.append(y)
        if correlation and correlation == 'pos':
            val += step
        elif correlation and correlation == 'neg':
            val -= step
    xs = [i for i in range(len(ys))]
    return np.array(xs,dtype=np.float64),np.array(ys,dtype=np.float64)


def best_fit_slope_and_intercept(xs,ys):
    m = ( ((mean(xs) * mean(ys)) - mean(xs * ys)) / 
            ((mean(xs) * mean(xs)) - mean(xs*xs)) )
    c = (mean(ys) - (m * mean(xs)))
    return m,c

def squared_error(y_orig,y_line):
    return sum((y_line - y_orig)**2)

def coeff_of_determination(y_orig,y_line):
    y_mean_line = [mean(y_orig) for y in y_orig]
    #print(y_mean_line)
    squared_err_reg = squared_error(y_orig,y_line)
    squared_err_mean = squared_error(y_orig,y_mean_line)
    return 1 - (squared_err_reg / squared_err_mean)

xs,ys = create_dataset(40,10,2,correlation='pos')
for i in range(len(xs)):
    print(xs[i],ys[i])
m,c = best_fit_slope_and_intercept(xs,ys)
print(m,c)

regression_line = [(m*x) + c for x in xs]

r_squared = coeff_of_determination(ys,regression_line)
print("rsq: ",r_squared)
predict_x = 8
predict_y = (m * predict_x) + c

plt.scatter(xs,ys)
#plt.scatter(predict_x,predict_y, color='g')
plt.plot(xs,regression_line)
plt.show()
