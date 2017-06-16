import pandas as pd
from pandas import DataFrame

df = DataFrame(
    [
        [1, 'a'],
        [2, 'b'],
        [3, 'c'],
    ],
    columns=['age', 'name']
)

print(df.describe(include='all'))

print(df[df['age'] > 1])

df['age'] = df['age'] * 2

print(df.as_matrix())

s = pd.Series(df['age'])

print(s.var())
