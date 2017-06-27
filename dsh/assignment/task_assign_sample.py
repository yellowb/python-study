import time

import pandas as pd
from pandas import DataFrame, Series
import numpy as np
from itertools import permutations

pd.options.mode.chained_assignment = None  # default='warn'


def build_empty_assignment_detail(users):
    """build the empty assignment detail obj from a list of users"""
    result = {user: {'total_time_used': 0, 'own_tasks': []} for user in users}
    return result


def assign_sub_matrix(sub_performance_matrix, offsets, assignment_detail):
    """Calculate the VDC users with sub-tasks"""
    # TODO

    users = sub_performance_matrix.columns.tolist()
    for idx, user in enumerate(users):
        sub_performance_matrix[user] += offsets[idx]

    print(sub_performance_matrix)
    return offsets


def assign_tasks(performance_matrix, assignment_detail):
    users = performance_matrix.columns.tolist()  # the columns of the DataFrame is userIds
    user_count = len(users)
    task_count = len(performance_matrix.index)  # The number of tasks
    offsets = [0] * user_count  # Offsets means the init value of each user when starting a round of calculation

    start_row = 0  # the start index of task in a round of calculation
    stop_row = 0
    remain_task_count = task_count  # How many tasks left to be calculated

    # Loop of each round of calculation
    while remain_task_count > 0:
        # Split the sub matrix from performance_matrix
        stop_row += remain_task_count if (remain_task_count < user_count) else user_count
        sub_performance_matrix = performance_matrix.iloc[start_row:stop_row]

        # Do calculation!
        offsets = assign_sub_matrix(sub_performance_matrix, offsets, assignment_detail)

        # Update the remain count & start index after one round
        remain_task_count -= (stop_row - start_row)
        start_row = stop_row
    return


def main():
    # Sample data
    users = ['P1', 'P2', 'P3']
    performance_2darray = np.array([
        [2, 5, 3],
        [2, 1, 4],
        [4, 3, 1],
        [3, 2, 1],
        [4, 5, 8],
        [7, 4, 4]
    ])
    performance_matrix = DataFrame(performance_2darray,
                                   columns=users,
                                   index=['T1', 'T2', 'T3', 'T4', 'T5', 'T6'])

    assignment_detail = build_empty_assignment_detail(users)

    assign_tasks(performance_matrix, assignment_detail)
    return


if __name__ == '__main__':
    start_time = time.time()
    main()
    stop_time = time.time()
    print('用时: ', (stop_time - start_time), ' (s)')
