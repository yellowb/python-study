import pandas as pd
import numpy as np

task_data = pd.read_csv('../../data/activity_main_mongodb.csv')

# print(task_data.describe());


percents = np.arange(0.9, 1, 0.01).tolist()

grouped_by_task_type_data = task_data.groupby('ACTIVITY_TYPE_NAME')
print(grouped_by_task_type_data.count())