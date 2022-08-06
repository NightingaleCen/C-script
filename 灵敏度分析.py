import numpy as np
from scipy.optimize import minimize
import statsmodels.api as sm
import matplotlib.pyplot as plt
import pandas as pd


x1m = sm.load('./x1_model.pickle')
x2m = sm.load('./x2_model.pickle')
x3m = sm.load('./x3_model.pickle')
y1model = sm.load('./y1_model.pickle')
y2model = sm.load('./y2_model.pickle')
y3model = sm.load('./y3_model.pickle')

x10 = 30
x20 = 1000
x1 = np.linspace(x10 * 0.95, x10 * 1.05, 30)
x2 = np.linspace(x20 * 0.95, x20 * 1.05, 30)

y10 = x1m.predict({'x1': x10, 'x2': x20}).values
y20 = x2m.predict({'x1': x10, 'x2': x20}).values
y30 = x3m.predict({'x1': x10, 'x2': x20}).values
z10 = y1model.predict({'x1': y10, 'x2': y20, 'x3': y30}).values
z20 = y2model.predict({'x1': y10, 'x2': y20, 'x3': y30}).values
z30 = y3model.predict({'x1': y10, 'x2': y20, 'x3': y30}).values

x10 = np.tile(30, 30)
x20 = np.tile(1000, 30)

y11 = x1m.predict({'x1': x1, 'x2': x20}).values
y21 = x2m.predict({'x1': x1, 'x2': x20}).values
y31 = x3m.predict({'x1': x1, 'x2': x20}).values
z11 = y1model.predict({'x1': y11, 'x2': y21, 'x3': y31}).values
z21 = y2model.predict({'x1': y11, 'x2': y21, 'x3': y31}).values
z31 = y3model.predict({'x1': y11, 'x2': y21, 'x3': y31}).values
y12 = x1m.predict({'x1': x10, 'x2': x2}).values
y22 = x2m.predict({'x1': x10, 'x2': x2}).values
y32 = x3m.predict({'x1': x10, 'x2': x2}).values
z12 = y1model.predict({'x1': y12, 'x2': y22, 'x3': y32}).values
z22 = y2model.predict({'x1': y12, 'x2': y22, 'x3': y32}).values
z32 = y3model.predict({'x1': y12, 'x2': y22, 'x3': y32}).values

z1_df = pd.DataFrame(
    {'过滤阻力Pa': z11, '过滤效率（%）': z21, '透气性 mm/s': z31}, index=x1)
z1_df.index.name = '接收距离(cm)'
z1_df.to_excel('./接收距离灵敏度分析.xlsx')
z2_df = pd.DataFrame(
    {'过滤阻力Pa': z12, '过滤效率（%）': z22, '透气性 mm/s': z32}, index=x2)
z2_df.index.name = '热风速度(r/min)'
z2_df.to_excel('./热风速度灵敏度分析.xlsx')

plt.rc('font', size=14)
plt.rc('font', family="SimHei")
plt.rc('axes', unicode_minus=False)
plt.rc('figure', figsize=(12.8, 3.6))
plt.subplots_adjust(wspace=0.4, hspace=0.5)
plt.subplot(131)
plt.plot(x1, z11)
plt.xlabel('接收距离(cm)')
plt.ylabel('过滤阻力Pa')
plt.subplot(132)
plt.plot(x1, z21)
plt.xlabel('接收距离(cm)')
plt.ylabel('过滤效率（%）')
plt.subplot(133)
plt.plot(x1, z31)
plt.xlabel('接收距离(cm)')
plt.ylabel('透气性 mm/s')
plt.savefig("./接收距离灵敏度分析.png", bbox_inches='tight')
plt.clf()
plt.subplot(131)
plt.plot(x2, z12)
plt.xlabel('热风速度(r/min)')
plt.ylabel('过滤阻力Pa')
plt.subplot(132)
plt.plot(x2, z22)
plt.xlabel('热风速度(r/min)')
plt.ylabel('过滤效率（%）')
plt.subplot(133)
plt.plot(x2, z32)
plt.xlabel('热风速度(r/min)')
plt.ylabel('透气性 mm/s')
plt.savefig("./热风速度灵敏度分析.png", bbox_inches='tight')
plt.clf()

with open("灵敏度分析结论.txt", 'w') as f:
    f.write('接收距离，热风速度初始值为:30, 1000\n')
    f.write(
        '当接收距离上下改变10%时，过滤阻力改变约{0}，过滤效率改变约{1}，透气性改变约{2}\n'.format(
            (z11.max() - z11.min()) / z11.min(),
            (z21.max() - z21.min()) / z21.min(),
            (z31.max() - z31.min()) / z31.min()))
    f.write(
        '当热风速度上下改变10%时，过滤阻力改变约{0}，过滤效率改变约{1}，透气性改变约{2}\n'.format(
            (z12.max() - z12.min()) / z12.min(),
            (z22.max() - z22.min()) / z22.min(),
            (z32.max() - z32.min()) / z32.min()))
