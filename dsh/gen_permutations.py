from itertools import permutations
from pandas import DataFrame
import pandas as pd
import numpy as np

pd.options.mode.chained_assignment = None  # default='warn'

list = ['P1', 'P2', 'P3']
gen = permutations(list, len(list))

for line in gen:
    print(line)

performance_matrix = np.array([
    [2, 5, 3],
    [2, 1, 4],
    [4, 3, 1],
    [3, 2, 1],
    [4, 5, 8],
    [7, 4, 4]
])

performance_matrix_df = DataFrame(performance_matrix,
                                  columns=list,
                                  index=['T1', 'T2', 'T3', 'T4', 'T5', 'T6'])

sub_df = performance_matrix_df.iloc[3:6]

# print(performance_matrix_df.iloc[0:3])

sub_df['P1'] += 2


print(performance_matrix_df[:6])

print(performance_matrix_df.index[0])   # Get one row's index name

print([0] * 3)


