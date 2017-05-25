from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler
import  numpy as np

X = np.array([
    [0., 0., 5., 13., 9., 1.],
    [0., 0., 13., 15., 10., 15.],
    [0., 3., 15., 2., 0., 11.]
])

X_normalized = Normalizer().fit_transform(X)
print(X_normalized)

X_scaled = StandardScaler().fit_transform(X)
print(X_scaled)

X_normalized_scaled = StandardScaler().fit_transform(X_normalized)
print(X_normalized_scaled)

X_scaled_normalized = Normalizer().fit_transform(X_scaled)
print(X_scaled_normalized)