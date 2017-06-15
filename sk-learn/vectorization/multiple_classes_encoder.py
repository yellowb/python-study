from sklearn.preprocessing import MultiLabelBinarizer
from pandas import DataFrame

mlb = MultiLabelBinarizer()

X_raw = [
    ['skill-group-A', 'skill-group-B'],
    ['skill-group-C'],
    ['skill-group-C', 'skill-group-D', 'skill-group-E']
]

X_encoded = mlb.fit_transform(X_raw)

print(X_encoded)

df = DataFrame(X_encoded, columns=['skill-group-A', 'skill-group-B', 'skill-group-C', 'skill-group-D', 'skill-group-E'])

print(df)
