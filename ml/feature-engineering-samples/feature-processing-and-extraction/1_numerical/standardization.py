from sklearn.preprocessing import StandardScaler
import numpy as np

X = np.array([[1, 4, -8],
              [-2, -1, 8],
              [3.3, 1, 0]])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(X)
print('\n')
print(X_scaled)