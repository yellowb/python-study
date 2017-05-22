from sklearn.feature_extraction import DictVectorizer

onehot_encoder = DictVectorizer()
instances = [{'a': 1, 'city': 'New York'}, {'a': 2, 'city': 'San Francisco'}, {'city': 'Chapel Hill'}]

print(onehot_encoder.fit_transform(instances).toarray())
