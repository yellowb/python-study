import numpy as np

# 交叉验证器
from sklearn.model_selection import cross_val_score

# 模型
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import BaggingRegressor

# 评估器
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

from pandas import DataFrame
import pandas as pd


def do():
    train_data = pd.read_csv('D:/testFiles/for_excute_folder/activity_blFreight_2017_5_train_input.csv')
    test_data = pd.read_csv('D:/testFiles/for_excute_folder/activity_blFreight_2017_5_test_input.csv')

    train_data = train_data.drop(['IDX'], axis=1)
    test_data = test_data.drop(['IDX'], axis=1)

    train_data = train_data[train_data["TIME_USED"] <= 1500]
    test_data = test_data[test_data["TIME_USED"] <= 1500]

    train_data['TIME_USED'] = train_data['TIME_USED'] / 60
    test_data['TIME_USED'] = test_data['TIME_USED'] / 60




    print(train_data.head())

    y_train = train_data['TIME_USED'].values.tolist()
    X_train = train_data.drop(['TIME_USED'], axis=1).values.tolist()

    y_test = test_data['TIME_USED'].values.tolist()
    X_test = test_data.drop(['TIME_USED'], axis=1).values.tolist()

    # 选一个模型

    # regressor = SGDRegressor(l1_ratio=0.1)
    # regressor = Ridge()
    # regressor = SVR()
    regressor = RandomForestRegressor()
    # regressor = AdaBoostRegressor()
    # regressor = GradientBoostingRegressor()
    # regressor = BaggingRegressor()

    # 用训练集做交叉验证
    # scores = cross_val_score(regressor, X_train, y_train, cv=5, scoring='r2', n_jobs=-1)

    # print('交叉验证R方值:', scores)
    # print('交叉验证R方均值:', np.mean([scores]))

    # 用训练集训练模型
    regressor.fit(X_train, y_train)
    # 用模型预测测试集, 打分方法也是r2
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


if __name__ == '__main__':
    do()
