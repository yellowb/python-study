from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import euclidean_distances

corpus = [
    'UNC played Duke in basketball',
    'Duke lost the basketball game',
    'I ate a sandwich'
]
vectorizer = CountVectorizer(stop_words='english')
vectors = vectorizer.fit_transform(corpus).todense()
print(vectors)
print(vectorizer.vocabulary_)

# distance of 2 vectors
dist = euclidean_distances(vectors[1], vectors[0])
print(dist)