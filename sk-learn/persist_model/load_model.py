from sklearn.externals import joblib

# 数据集
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

# 评估器
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

from pandas import DataFrame

# Load models
scaler = joblib.load('Scaler.pkl')
regressor = joblib.load('GBDT_regressor.pkl')

# 数据集3/4作为训练集, 其余作为测试集
data = load_boston()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.25, random_state=3)

# 标准化测试数据
X_test = scaler.transform(X_test)

print('测试集R方值:', regressor.score(X_test, y_test))

# 对比预测数据与真实数据
y_predict = regressor.predict(X_test);
df = DataFrame()
df['predict'] = y_predict
df['real'] = y_test
df['diff'] = y_predict - y_test
print(df.head(20))

print('MAE =  ', mean_absolute_error(y_test, y_predict))
print('MSE =  ', mean_squared_error(y_test, y_predict))
print('R2 =  ', r2_score(y_test, y_predict))


