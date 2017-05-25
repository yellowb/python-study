from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
import numpy as np

X = np.array([
    [0., 0., 5., 13., 9., 1.],
    [0., 0., 13., 15., 10., 15.],
    [0., 3., 15., 2., 0., 11.]
])

print(preprocessing.scale(X))

X_scaled = StandardScaler().fit_transform(X)
print(X_scaled)