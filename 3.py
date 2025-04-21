from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# same 5 documents
documents = [
    "Words have meaning based on their use and context. Synonymy and antonymy show relations.",
    "Vector semantics represents word meaning in space. Similar words appear in similar contexts.",
    "TF-IDF gives importance to unique words by reducing scores of common terms like 'the'.",
    "Word2Vec creates dense vector embeddings using neural networks and context windows.",
    "Cosine similarity helps compare vectors and find similar words or documents."
]

# TF-IDF vectorization
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(documents)

# Compute cosine similarity
cos_sim = cosine_similarity(tfidf_matrix)

# Print matrix
print("Cosine Similarity Matrix:")
print(np.round(cos_sim, 2))

max_val = -1 # to be safe incase no pair
pair = None

for i in range(len(documents)):
    for j in range(i + 1, len(documents)):   # Avoid duplicate 
        if cos_sim[i][j] > max_val:
            max_val = cos_sim[i][j]
            pair = (i, j)

if pair:
    print(f"\nMost similar documents: Document {pair[0]} and Document {pair[1]} (cosine similarity = {round(max_val, 2)})")