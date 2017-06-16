import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer

task_train_data = pd.read_csv('D:\\testFiles\\activity_blFreight_2017_5_train.csv')

# print(task_train_data.describe());

# print(task_train_data.info());
#
# print(task_train_data['SKILL_GROUPS'].describe())

# print(task_train_data['SKILL_GROUPS'][0])

# task_train_data['TIME_USED'] = task_train_data['TIME_USED']/60.0

# print(task_train_data[task_train_data['Region'] == 'ANGC'].groupby('OWNER_ID').agg([np.max, np.min , np.median, np.std]))

# print(task_train_data.groupby('OWNER_ID').count())

skill_groups_array = task_train_data['SKILL_GROUPS'].tolist()

# split combined string into skill group array
for idx, val in enumerate(skill_groups_array):
    skill_groups_array[idx] = skill_groups_array[idx][1:-1].split(', ')

for idx, val in enumerate(skill_groups_array):
    print(idx, val)

mlb = MultiLabelBinarizer()
skill_groups_array_encoded = mlb.fit_transform(skill_groups_array)

print(skill_groups_array_encoded)

print('rows: ', len(skill_groups_array_encoded))
print('columns: ', len(skill_groups_array_encoded[0]))

print(mlb.get_params())
