import numpy as np
# 数据集
from sklearn.datasets import load_boston

from sklearn.preprocessing import StandardScaler

# 交叉验证器
from sklearn.model_selection import train_test_split

# 模型
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.externals import joblib




def do():
    data = load_boston()
    # 数据集3/4作为训练集, 其余作为测试集
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.25, random_state=3)

    # 标准化数值feature
    X_scaler = StandardScaler().fit(X_train)
    X_train = X_scaler.transform(X_train)

    # 选一个模型
    regressor = GradientBoostingRegressor()
    regressor.fit(X_train, y_train)

    # Store the model to disk
    joblib.dump(regressor, 'GBDT_regressor.pkl')
    joblib.dump(X_scaler, 'Scaler.pkl')

if __name__ == '__main__':
    do()
