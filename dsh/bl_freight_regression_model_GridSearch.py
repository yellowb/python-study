import numpy as np

# 交叉验证器
from sklearn.model_selection import cross_val_score

# 超参数搜索
from sklearn.model_selection import GridSearchCV

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
import time


def do():
    train_data = pd.read_csv('D:/testFiles/for_excute_folder/activity_blFreight_2017_5_train_input.csv')
    test_data = pd.read_csv('D:/testFiles/for_excute_folder/activity_blFreight_2017_5_test_input.csv')

    drop_col_names = ['Global-SystemAdmin']

    train_data = train_data.drop(drop_col_names, axis=1)
    test_data = test_data.drop(drop_col_names, axis=1)

    # Drop the 1st no-meaning column
    train_data = train_data.drop(train_data.columns[0], axis=1)
    test_data = test_data.drop(test_data.columns[0], axis=1)

    # limit the TIME_USED of data
    train_data = train_data[train_data["TIME_USED"] <= 1000]
    test_data = test_data[test_data["TIME_USED"] <= 1000]

    # change second -> minute
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
    regressor = RandomForestRegressor(n_jobs=-1)
    # regressor = AdaBoostRegressor()
    # regressor = GradientBoostingRegressor(n_estimators=400, max_depth=4, loss='huber')
    # regressor = BaggingRegressor()

    # 超参数搜索空间
    param_grid = {
        'criterion': ['mse', 'mae'],
        'n_estimators': [10, 20, 50, 100, 200, 300, 400, 500, 750, 1000],
        'max_features': ['auto', 'sqrt', 'log2']
    }

    grid_search = GridSearchCV(estimator=regressor, param_grid=param_grid, n_jobs=-1, verbose=1,
                               scoring='neg_mean_absolute_error', cv=4)

    # 搜索超参数组合
    grid_search.fit(X_train, y_train)
    print('最佳效果：%0.3f' % grid_search.best_score_)
    print('最优参数组合：')
    best_parameters = grid_search.best_estimator_.get_params()

    # 输出最佳超参数
    for param_name in sorted(param_grid.keys()):
        print('\t%s: %r' % (param_name, best_parameters[param_name]))

    # 用最佳超参数预测测试集
    y_predict = grid_search.predict(X_test)
    print('MAE =  ', mean_absolute_error(y_test, y_predict))
    print('MSE =  ', mean_squared_error(y_test, y_predict))
    print('R2 =  ', r2_score(y_test, y_predict))


if __name__ == '__main__':
    start_time = time.time()
    do()
    stop_time = time.time()
    print('用时: ', (stop_time - start_time), ' (s)')
