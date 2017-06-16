from sklearn.preprocessing import MultiLabelBinarizer
from pandas import DataFrame

mlb = MultiLabelBinarizer()

X_raw = [
    ['skill-group-E', 'skill-group-B'],
    ['skill-group-C', 'skill-group-D'],
    ['skill-group-C', 'skill-group-A', 'skill-group-E']
]

X_encoded = mlb.fit_transform(X_raw)

print(X_encoded)

df = DataFrame(X_encoded, columns=mlb.classes_)

print(df)
