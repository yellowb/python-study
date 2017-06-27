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


def assign_sub_matrix(sub_performance_matrix, assignment_detail):
    """Calculate the VDC users with sub-tasks"""
    users = sub_performance_matrix.columns.tolist()
    task_count = len(sub_performance_matrix.index)  # How many tasks in this sub-matrix, maybe less then users

    # Add offset to each users' baseline
    for idx, user in enumerate(users):
        sub_performance_matrix[user] += assignment_detail[user]['total_time_used']

    user_permutations = permutations(users, task_count)  # The permutations of all users on tasks = A(users, task_count)

    minimum_permutation_time_used = -1
    minimum_permutation = None

    # Loop each permutation to find the smallest
    for idx_p, permutation in enumerate(user_permutations):
        # print(permutation)

        maximum_time_used_in_permutation = -1  # the time_used of smallest permutation
        for idx_u, user in enumerate(permutation):
            time_used_for_user_on_task = sub_performance_matrix.iloc[idx_u][user]

            # the max time used in one permutation
            maximum_time_used_in_permutation = time_used_for_user_on_task if (
                time_used_for_user_on_task > maximum_time_used_in_permutation) else maximum_time_used_in_permutation

        # Find the smallest max time used permutation
        if (maximum_time_used_in_permutation < minimum_permutation_time_used) or (minimum_permutation_time_used == -1):
            minimum_permutation_time_used = maximum_time_used_in_permutation
            minimum_permutation = permutation

    print('Smallest permutation: ', minimum_permutation)

    for idx, user in enumerate(minimum_permutation):
        time_used_for_user_on_task = sub_performance_matrix.iloc[idx][user]
        task_name = sub_performance_matrix.index[idx]
        assignment_detail[user]['total_time_used'] = time_used_for_user_on_task
        assignment_detail[user]['own_tasks'].append(task_name)

    print(sub_performance_matrix)
    return


def assign_tasks(performance_matrix, assignment_detail):
    users = performance_matrix.columns.tolist()  # the columns of the DataFrame is userIds
    user_count = len(users)
    task_count = len(performance_matrix.index)  # The number of tasks
    # offsets = [0] * user_count  # Offsets means the init value of each user when starting a round of calculation

    start_row = 0  # the start index of task in a round of calculation
    stop_row = 0
    remain_task_count = task_count  # How many tasks left to be calculated

    # Loop of each round of calculation
    while remain_task_count > 0:
        # Split the sub matrix from performance_matrix
        stop_row += remain_task_count if (remain_task_count < user_count) else user_count
        sub_performance_matrix = performance_matrix.iloc[start_row:stop_row]

        # Do calculation!
        assign_sub_matrix(sub_performance_matrix, assignment_detail)

        # Update the remain count & start index after one round
        remain_task_count -= (stop_row - start_row)
        start_row = stop_row

        print(assignment_detail)

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
        [7, 4, 4],

        [6, 3, 3],
        [3, 5, 2]
    ])
    performance_matrix = DataFrame(performance_2darray,
                                   columns=users,
                                   index=['T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8'])

    assignment_detail = build_empty_assignment_detail(users)

    assign_tasks(performance_matrix, assignment_detail)
    return


if __name__ == '__main__':
    start_time = time.time()
    main()
    stop_time = time.time()
    print('用时: ', (stop_time - start_time), ' (s)')
