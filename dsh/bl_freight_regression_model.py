import os
import time

mingw_path = r"C:\Anaconda3_4.4.0\Library\mingw-w64\bin"

os.environ['PATH'] = mingw_path + ';' + os.environ['PATH']

import numpy as np

# 交叉验证器
from sklearn.model_selection import cross_val_score

# 模型
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import BaggingRegressor

from xgboost import XGBRegressor
from lightgbm.sklearn import LGBMRegressor

# 评估器
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

from pandas import DataFrame
import pandas as pd



def do():
    train_data = pd.read_csv('D:/testFiles/for_excute_folder/activity_blFreight_2017_5_train_input.csv')
    test_data = pd.read_csv('D:/testFiles/for_excute_folder/activity_blFreight_2017_5_test_input.csv')

    # test_data = pd.read_csv('D:/testFiles/for_excute_folder/activity_blFreight_2017_5_train_input.csv')
    # train_data = pd.read_csv('D:/testFiles/for_excute_folder/activity_blFreight_2017_5_test_input.csv')

    # drop_col_names = ['Global-SystemAdmin']

    train_data = train_data.drop(train_data.columns[0], axis=1)
    test_data = test_data.drop(test_data.columns[0], axis=1)

    train_data = train_data[train_data["TIME_USED"] <= 1000]
    test_data = test_data[test_data["TIME_USED"] <= 1000]

    # train_data = train_data[train_data["ASSIGN_COUNT"] <= 1]
    # test_data = test_data[test_data["ASSIGN_COUNT"] <= 1]

    # train_data = train_data.drop(drop_col_names, axis=1)
    # test_data = test_data.drop(drop_col_names, axis=1)

    train_data['TIME_USED'] = train_data['TIME_USED'] / 60
    test_data['TIME_USED'] = test_data['TIME_USED'] / 60

    train_data['TIME_USERD_MEDIAN_S2'] = train_data['TIME_USERD_MEDIAN'] ** 2
    test_data['TIME_USERD_MEDIAN_S2'] = test_data['TIME_USERD_MEDIAN'] ** 2

    #bkgOffice_median_by_task_type

    train_data['TIME_USERD_MEDIAN_S3'] = train_data['TIME_USERD_MEDIAN'] * train_data['bkgOffice_median_by_task_type']
    test_data['TIME_USERD_MEDIAN_S3'] = test_data['TIME_USERD_MEDIAN'] * test_data['bkgOffice_median_by_task_type']

    train_data['TIME_USERD_MEDIAN_S4'] = train_data['bkgOffice_mean_by_task_type'] * train_data['bkgOffice_median_by_task_type']
    test_data['TIME_USERD_MEDIAN_S4'] = test_data['bkgOffice_mean_by_task_type'] * test_data['bkgOffice_median_by_task_type']

    # train_data = train_data[
    #     ['TIME_USED', 'TIME_USERD_MEDIAN', 'Freight_Type=Semi Auto', 'Freight_Type=No rate', 'TIME_USERD_COUNT', 'TIME_USERD_VAR', 'AWAY_COUNT',
    #      'AWAY_MEAN', 'TIME_USED_BY_REGION' ,'COUNT_BY_REGION', 'TIME_USED_VAR_BY_REGION']]
    # test_data = test_data[
    #     ['TIME_USED', 'TIME_USERD_MEDIAN', 'Freight_Type=Semi Auto', 'Freight_Type=No rate', 'TIME_USERD_COUNT', 'TIME_USERD_VAR', 'AWAY_COUNT',
    #      'AWAY_MEAN', 'TIME_USED_BY_REGION' ,'COUNT_BY_REGION', 'TIME_USED_VAR_BY_REGION']]

    print(test_data.head())

    # print(train_data.describe())

    y_train = train_data['TIME_USED'].values.tolist()
    X_train = train_data.drop(['TIME_USED'], axis=1).values.tolist()

    y_test = test_data['TIME_USED'].values.tolist()
    X_test = test_data.drop(['TIME_USED'], axis=1).values.tolist()

    # 选一个模型

    # regressor = SGDRegressor(l1_ratio=0.1)
    # regressor = Ridge()
    # regressor = Lasso()
    # regressor = SVR()
    # regressor = RandomForestRegressor(n_estimators=400, n_jobs=-1, max_features='sqrt')
    # regressor = AdaBoostRegressor()
    # regressor = GradientBoostingRegressor(n_estimators=400)
    # regressor = BaggingRegressor()
    regressor = XGBRegressor(n_estimators=400, learning_rate=0.02, colsample_bytree=0.1, seed=2017)
    # regressor = LGBMRegressor(n_estimators=400, learning_rate=0.02, seed=2017, colsample_bytree=1)

    # 用训练集做交叉验证
    # scores = cross_val_score(regressor, X_train, y_train, cv=4, scoring='neg_mean_absolute_error', n_jobs=-1)
    #
    # print('交叉验证R方值:', scores)
    # print('交叉验证R方均值:', np.mean([scores]))

    # 用训练集训练模型
    regressor.fit(X_train, y_train)
    # 用模型预测测试集, 打分方法也是r2
    print('测试集R方值:', regressor.score(X_test, y_test))

    # 对比预测数据与真实数据
    y_predict = regressor.predict(X_test)
    df = DataFrame()
    df['predict'] = y_predict
    df['real'] = y_test
    df['diff'] = y_predict - y_test
    df['diff_abs'] = abs(df['diff'])

    df.sort_values(by='diff_abs', ascending=False, inplace=True)

    print(df.head(20))

    print(df['diff_abs'].describe(percentiles=np.arange(0.1, 1, 0.1)))

    print('MAE =  ', mean_absolute_error(y_test, y_predict))
    print('MSE =  ', mean_squared_error(y_test, y_predict))
    print('R2 =  ', r2_score(y_test, y_predict))

    print('feature_importances\n')
    # print(regressor.feature_importances_)  # Only tree based model has this attribute


if __name__ == '__main__':
    start_time = time.time()
    do()
    stop_time = time.time()
    print('用时: ', (stop_time - start_time), ' (s)')
