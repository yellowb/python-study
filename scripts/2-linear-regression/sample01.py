import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from sklearn.linear_model import LinearRegression
import numpy as np

font = FontProperties(fname=r"c:\windows\fonts\msyh.ttc", size=10)


def runplt():
    plt.figure()
    plt.title('匹萨价格与直径数据', fontproperties=font)
    plt.xlabel('直径（英寸）', fontproperties=font)
    plt.ylabel('价格（美元）', fontproperties=font)
    plt.axis([0, 25, 0, 25])
    plt.grid(True)
    return plt


plt = runplt()
X = [[6], [8], [10], [14], [18]]
y = [[7], [9], [13], [17.5], [18]]
plt.plot(X, y, 'k.')


model = LinearRegression()
X2 = [[0], [12], [20], [25]]
model.fit(X, y)
y2 = model.predict(X2)
plt.plot(X2, y2, 'g-')

# print('预测一张12英寸匹萨价格：', model.predict([[12], [20]]))

# 残差预测值
yr = model.predict(X)
for idx, x in enumerate(X):
    plt.plot([x, x], [y[idx], yr[idx]], 'r-')

# plt.show()

print(model.predict(X) - y)
print('残差平方和: %.2f' % np.mean((model.predict(X) - y) ** 2))

print('OK!')
