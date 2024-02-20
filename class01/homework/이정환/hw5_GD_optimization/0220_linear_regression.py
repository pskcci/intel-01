#여러가지 linear regression 실습
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

x = [5, 7, 8, 7, 2, 17, 2, 9, 4, 11, 12, 9, 6]
y = [99, 86, 87, 88, 111, 86, 103, 87, 94, 78, 77, 85, 86]

slope, intercept, r, p, std_err = stats.linregress(x, y)

def myfunc(x):
    return slope * x + intercept

xx = np.linspace(0, 18, 180)
yy = slope * xx + intercept

plt.scatter(x, y)
plt.plot(xx, yy)
plt.show()
