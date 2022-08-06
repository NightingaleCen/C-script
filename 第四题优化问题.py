import numpy as np
from scipy.optimize import minimize
import statsmodels.api as sm

y1model = sm.load('./y1_model.pickle')
y2model = sm.load('./y2_model.pickle')
y3model = sm.load('./y3_model.pickle')


def funE(x): return - \
    (y2model.predict({"x1": x[0], 'x2': x[1], 'x3': x[2]}).values[0])


def funf(x): return (y1model.predict(
    {"x1": x[0], 'x2': x[1], 'x3': x[2]}).values[0])


bd = [(0, None), (0, None), (0, None), (0, 100), (0, 2000),
      (0, None), (0, None), (0, None), (0, None)]
con = ({'type': 'eq',
        'fun': lambda x: -x[0] - 2.9791 - 0.0193 * x[3] + 0.0083 * x[4] + 6.253e-06 * x[3] * x[4] + 0.0011 * x[3] * x[3] - 3.325e-06 * x[4] * x[4]},
       {'type': 'eq',
        'fun': lambda x: -x[1] + 75.0643 + 0.352 * x[3] + 0.026 * x[4] - 0.0002 * x[3] * x[4] - 0.0015 * x[3] * x[3] - 8.891e-06 * x[4] * x[4]},
       {'type': 'eq',
        'fun': lambda x: -x[2] + 43.9577 + 1.1539 * x[3] + 0.0588 * x[4] + 2.066e-05 * x[3] * x[4] - 0.0207 * x[3] * x[3] - 3.108e-05 * x[4] * x[4]},
       {'type': 'eq',
        'fun': lambda x: - 2.9791 - 0.0193 * x[3] + 0.0083 * x[4] + 6.253e-06 * x[3] * x[4] + 0.0011 * x[3] * x[3] - 3.325e-06 * x[4] * x[4] - x[5] + x[6] - 3},
       {'type': 'eq',
        'fun': lambda x: 43.9577 + 1.1539 * x[3] + 0.0588 * x[4] + 2.066e-05 * x[3] * x[4] - 0.0207 * x[3] * x[3] - 3.108e-05 * x[4] * x[4] - x[7] + x[8] - 85})

P1 = 1000000
P2 = 3.5
P3 = 1


def fun(x):
    targets = [funf(x), funE(x), x[5], x[8]]
    return P3 * (targets[0]) + P2 * (targets[1]) + P1 * \
        (targets[2]) + P1 * (targets[3])


res = minimize(
    fun,
    np.random.randn(9),
    constraints=con,
    bounds=bd,
    method='trust-constr')

y1 = y1model.predict(
    {"x1": res.x[0], 'x2': res.x[1], 'x3': res.x[2]}).values[0]
y2 = y2model.predict(
    {"x1": res.x[0], 'x2': res.x[1], 'x3': res.x[2]}).values[0]
y3 = y3model.predict(
    {"x1": res.x[0], 'x2': res.x[1], 'x3': res.x[2]}).values[0]

with open("第四题最优化结果.txt", 'w') as f:
    #f.write('x[3]:接收距离d, x[4]:热风速度v, x[0]:厚度h, x[1]:孔隙率p, x[2]:缩回弹性率c, x[5]:dh+, x[6]:dh-, x[7]:dc+, x[8]:dc-')
    f.write('达到满意解时优先级P1:{0}, P2:{1}, P3:{2}\n'.format(P1, P2, P3))
    f.write('满意解：(d, v) = ({0}, {1})\n'.format(res.x[3], res.x[4]))
    f.write(
        '结构变量:(h, p, c) = ({0}, {1}, {2})\n'.format(
            res.x[0],
            res.x[1],
            res.x[2]))
    f.write(
        '产品性能:(f, E, B) = ({0}, {1}, {2})\n'.format(
            y1,
            y2,
            y3))
