from pandas import Series
import numpy as np

data = np.arange(0, 1000)
_1_d_array = Series(data)
print(_1_d_array.describe())

map1 = Series([1, 2, 3], index=['a', 'b', 'c'])
print(map1['a'])
