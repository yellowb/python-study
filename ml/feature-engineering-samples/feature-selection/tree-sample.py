import numpy as np
from pandas import DataFrame
from sklearn.datasets import load_iris
from sklearn import metrics

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import RidgeClassifier

from sklearn.svm import NuSVC

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

from sklearn.preprocessing import Normalizer

# from sklearn.cross_validation import train_test_split

tree = RandomForestClassifier(n_estimators=10, n_jobs=-1)
iris = load_iris()
normalizer = Normalizer()


X = DataFrame(iris.data)
y = iris.target
X = normalizer.fit_transform(X)

# print(panda)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

tree.fit(X_train, y_train)
print(tree.feature_importances_)
print(tree.score(X_test, y_test))

scores = cross_val_score(tree, X, y, cv=5)

print('交叉验证R方值:', scores)
print('交叉验证R方均值:', np.mean(scores))