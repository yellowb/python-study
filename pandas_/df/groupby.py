from pandas import DataFrame
import pandas as pd

matrix = [
    ['Tom', 'EN', 90],
    ['Mary', 'CH', 80],
    ['Ken', 'EN', 70],
    ['Ken', 'CH', 40],
    ['Tom', 'CH', 70]
]

df = DataFrame(matrix, columns=['Name', 'Subject', 'Score'])

print(df.groupby(by='Subject').mean())

_2_level_grouped = df.groupby(by=['Subject', 'Name'])
for i,j in _2_level_grouped:
    print(i)
    print('------------')
    print(j)
