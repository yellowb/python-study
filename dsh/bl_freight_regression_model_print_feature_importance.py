import numpy as np
import matplotlib.pyplot as plt

# 交叉验证器
from sklearn.model_selection import cross_val_score

# 模型
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
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
    train_data = pd.read_csv('D:/testFiles/for_excute_folder-full data/activity_blFreight_2017_5_train_input.csv')
    test_data = pd.read_csv('D:/testFiles/for_excute_folder-full data/activity_blFreight_2017_5_test_input.csv')

    drop_col_names = ['Global-SystemAdmin']

    train_data = train_data.drop(drop_col_names, axis=1)
    test_data = test_data.drop(drop_col_names, axis=1)

    train_data = train_data.drop(train_data.columns[0], axis=1)
    test_data = test_data.drop(test_data.columns[0], axis=1)

    train_data = train_data[train_data["TIME_USED"] <= 1000]
    test_data = test_data[test_data["TIME_USED"] <= 1000]

    train_data['TIME_USED'] = train_data['TIME_USED'] / 60
    test_data['TIME_USED'] = test_data['TIME_USED'] / 60

    print(train_data.head())

    y_train = train_data['TIME_USED'].values.tolist()
    train_data_X = train_data.drop(['TIME_USED'], axis=1)   # drop the 'y' column
    X_train = train_data_X.values.tolist()

    # 选一个模型

    # regressor = SGDRegressor(l1_ratio=0.1)
    # regressor = Ridge()
    # regressor = SVR()
    # regressor = RandomForestRegressor(n_estimators=100, n_jobs=-1)
    # regressor = AdaBoostRegressor()
    regressor = GradientBoostingRegressor(n_estimators=400)
    # regressor = BaggingRegressor()
    # regressor = ExtraTreesRegressor()

    # 用训练集做交叉验证
    scores = cross_val_score(regressor, X_train, y_train, cv=4, scoring='r2', n_jobs=-1)

    print('交叉验证R方值:', scores)
    print('交叉验证R方均值:', np.mean([scores]))

    # 用训练集训练模型
    regressor.fit(X_train, y_train)

    print('feature_importances\n')
    print(regressor.feature_importances_)  # Only tree based model has this attribute

    # print(sorted(regressor.feature_importances_))

    feature_names = list(train_data_X.columns.values)
    feature_importances_names = DataFrame()
    feature_importances_names['name'] = feature_names
    feature_importances_names['importance'] = regressor.feature_importances_


    feature_importances_names.sort_values(by='importance', ascending=False, inplace=True)


    print(feature_importances_names)

    objects = feature_importances_names[:11]['name'].tolist()
    y_pos = np.arange(len(objects))
    performance = feature_importances_names[:11]['importance'].tolist()

    fig = plt.figure()

    plt.bar(y_pos, performance, align='center', alpha=0.5)
    plt.xticks(y_pos, objects)
    plt.ylabel('Feature Importance')
    plt.title('Feature Importance by GBDT')


    fig.autofmt_xdate()

    plt.show()




if __name__ == '__main__':
    do()
