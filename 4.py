from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np

# 5 sample documents (same as before)
documents = [
    "Words have meaning based on their use and context. Synonymy and antonymy show relations.",
    "Vector semantics represents word meaning in space. Similar words appear in similar contexts.",
    "TF-IDF gives importance to unique words by reducing scores of common terms like 'the'.",
    "Word2Vec creates dense vector embeddings using neural networks and context windows.",
    "Cosine similarity helps compare vectors and find similar words or documents."
]

# Preprocess and tokenize
tokenized_docs = [doc.lower().split() for doc in documents]

# Train Word2Vec model
model = Word2Vec(sentences=tokenized_docs, vector_size=100, window=5, min_count=1, sg=1)

# Create document vectors by averaging word vectors
def get_doc_vector(doc):
    vectors = [model.wv[word] for word in doc if word in model.wv]
    return np.mean(vectors, axis=0)

X = np.array([get_doc_vector(doc) for doc in tokenized_docs])
y = np.array([0, 0, 0, 1, 1])  # fake labels for demo (e.g. class 0 = meaning, class 1 = models)

# Train logistic regression
clf = LogisticRegression()
clf.fit(X, y)

# Predict and evaluate
y_pred = clf.predict(X)
accuracy = accuracy_score(y, y_pred)
print(f"Accuracy: {accuracy}")
