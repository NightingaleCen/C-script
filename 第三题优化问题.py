import numpy as np
from scipy.optimize import minimize
import statsmodels.api as sm
from sko.GA import GA


model = sm.load('./y2_model.pickle')


def fun(x): return - \
    (model.predict({"x1": x[0], 'x2': x[1], 'x3': x[2]}).values[0])


x1m = sm.load('./x1_model.pickle')
x2m = sm.load('./x2_model.pickle')
x3m = sm.load('./x3_model.pickle')

bd = [(0, None), (0, None), (0, None), (20, 40), (800, 1200)]
con = ({'type': 'eq',
        'fun': lambda x: -x[0] - 2.9791 - 0.0193 * x[3] + 0.0083 * x[4] + 6.253e-06 * x[3] * x[4] + 0.0011 * x[3] * x[3] - 3.325e-06 * x[4] * x[4]},
       {'type': 'eq',
        'fun': lambda x: -x[1] + 75.0643 + 0.352 * x[3] + 0.026 * x[4] - 0.0002 * x[3] * x[4] - 0.0015 * x[3] * x[3] - 8.891e-06 * x[4] * x[4]},
       {'type': 'eq',
        'fun': lambda x: -x[2] + 43.9577 + 1.1539 * x[3] + 0.0588 * x[4] + 2.066e-05 * x[3] * x[4] - 0.0207 * x[3] * x[3] - 3.108e-05 * x[4] * x[4]})
res = minimize(fun, np.random.randn(5), constraints=con, bounds=bd)
print(res)
with open("第三问最优化结果.txt", 'w') as f:
    f.write(str(res)+'\n')
    f.write('x[3]:接收距离, x[4]:热风速度, x[0]:厚度, x[1]:孔隙率, x[2]:缩回弹性率')
