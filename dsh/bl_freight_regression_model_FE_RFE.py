import os

mingw_path = r"C:\Anaconda3_4.4.0\Library\mingw-w64\bin"

os.environ['PATH'] = mingw_path + ';' + os.environ['PATH']

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

from xgboost import XGBRegressor
from lightgbm.sklearn import LGBMRegressor

# 评估器
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

from pandas import DataFrame
import pandas as pd

import matplotlib.pyplot as plt
from sklearn.feature_selection import RFECV


def do():
    train_data = pd.read_csv('D:/testFiles/for_excute_folder/activity_blFreight_2017_5_train_input.csv')
    test_data = pd.read_csv('D:/testFiles/for_excute_folder/activity_blFreight_2017_5_test_input.csv')

    # Filter the Timeused <= 1000s
    train_data = train_data[train_data["TIME_USED"] <= 1000]
    test_data = test_data[test_data["TIME_USED"] <= 1000]

    # convert second to minute
    train_data['TIME_USED'] = train_data['TIME_USED'] / 60
    test_data['TIME_USED'] = test_data['TIME_USED'] / 60

    train_data['TIME_USERD_MEDIAN_S2'] = train_data['TIME_USERD_MEDIAN'] ** 2
    test_data['TIME_USERD_MEDIAN_S2'] = test_data['TIME_USERD_MEDIAN'] ** 2

    # bkgOffice_median_by_task_type

    train_data['TIME_USERD_MEDIAN_S3'] = train_data['TIME_USERD_MEDIAN'] * train_data['bkgOffice_median_by_task_type']
    test_data['TIME_USERD_MEDIAN_S3'] = test_data['TIME_USERD_MEDIAN'] * test_data['bkgOffice_median_by_task_type']

    print(train_data.head())

    y_train = train_data['TIME_USED'].values.tolist()
    X_train = train_data.drop(['TIME_USED'], axis=1).values.tolist()

    # 选一个模型

    # regressor = SGDRegressor(l1_ratio=0.1)
    # regressor = Ridge()
    # regressor = SVR()
    # regressor = RandomForestRegressor(n_estimators=100)
    # regressor = AdaBoostRegressor()
    # regressor = GradientBoostingRegressor()
    # regressor = BaggingRegressor()
    # regressor = XGBRegressor(n_estimators=400)    # NOT WORK!
    regressor = LGBMRegressor(n_estimators=400, learning_rate=0.02, seed=2017, colsample_bytree=1)

    rfecv = RFECV(estimator=regressor, step=1, cv=5,
                  scoring='r2', n_jobs=-1)
    rfecv.fit(X_train, y_train)

    print("Optimal number of features : %d" % rfecv.n_features_)

    # Plot number of features VS. cross-validation scores
    plt.figure()
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score (nb of correct classifications)")
    plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
    plt.show()

    print(rfecv.support_)
    print(rfecv.ranking_)


if __name__ == '__main__':
    do()
