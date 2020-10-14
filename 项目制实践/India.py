import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.metrics import mean_squared_error
from scipy.optimize import curve_fit
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
# %matplotlib inline

url = "data3.csv"

df = pd.read_csv(url)
df = df.loc[:, ['date', 'tot_num']]
FMT = '%Y/%m/%d'
date = df['date']
df['date'] = date.map(lambda x: (datetime.strptime(x, FMT) - datetime.strptime("2020/01/01", FMT)).days)


def logistic_model(x, a, b, c):
    return c / (1 + np.exp(-(x - b) / a))

x = list(df.iloc[:, 0])
y = list(df.iloc[:, 1])

fit = curve_fit(logistic_model, x, y, p0=[2, 100, 20000])
errors = [np.sqrt(fit[1][i][i]) for i in [0, 1, 2]]
print("logistic_model")
print([fit[0][i] for i in [0, 1, 2]])
print([errors[i] for i in [0, 1, 2]])

sol = int(fsolve(lambda x: logistic_model(x, fit[0][0], fit[0][1], fit[0][2]) - int(fit[0][2]), fit[0][1]))
print(sol)


def exponential_model(x, a, b, c):
    return a * np.exp(b * (x - c))
    '''
    if np.all((b * ( x - c ) ) >= 0):  # 对sigmoid函数优化，避免出现极大的数据溢出
        return a*np.exp(b * ( x - c))
    else:
        return a/np.exp( -b * ( x - c ))
    '''
exp_fit = curve_fit(exponential_model, x, y, p0=[1, 1, 1], maxfev=5000)

errors2 = [np.sqrt(exp_fit[1][i][i]) for i in [0, 1, 2]]
print("exponential_model")
print([exp_fit[0][i] for i in [0, 1, 2]])
print([errors2[i] for i in [0, 1, 2]])

pred_x = list(range(max(x), sol))
plt.rcParams['figure.figsize'] = [7, 7]
plt.rc('font', size=14)
# Real data
plt.scatter(x, y, label="Real data", color="red")

# Predicted logistic curve
plt.plot(x + pred_x, [logistic_model(i, fit[0][0], fit[0][1], fit[0][2])
                      for i in x + pred_x], label="Logistic model")

# Predicted exponential curve
plt.plot(x + pred_x, [exponential_model(i, exp_fit[0][0], exp_fit[0][1], exp_fit[0][2]) for i in x + pred_x],
         label="Exponential model")
plt.legend()
plt.xlabel("Days since 1 January 2020")
plt.ylabel("Total number of infected people")
plt.xlim(0,300)
plt.ylim((min(y) * 0.9, max(y) * 1.1))
plt.show();

y_pred_logistic = [logistic_model(i, fit[0][0], fit[0][1], fit[0][2]) for i in x]
y_pred_exp = [exponential_model(i, exp_fit[0][0], exp_fit[0][1], exp_fit[0][2]) for i in x]
mse1 = mean_squared_error(y, y_pred_logistic)
mse2 = mean_squared_error(y, y_pred_exp)

print("pred_logistic")
print(mse1)
print("pred_exp")
print(mse2)