from pandas import Series
import numpy as np

data = np.arange(0, 1000)
_1_d_array = Series(data)
print(_1_d_array.index)

map1 = Series([1, 2, 3])
print(map1.index)


print(_1_d_array.var())

