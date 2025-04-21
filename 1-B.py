from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

# Same 5 documents
documents = [
    "Words have meaning based on their use and context. Synonymy and antonymy show relations.",
    "Vector semantics represents word meaning in space. Similar words appear in similar contexts.",
    "TF-IDF gives importance to unique words by reducing scores of common terms like 'the'.",
    "Word2Vec creates dense vector embeddings using neural networks and context windows.",
    "Cosine similarity helps compare vectors and find similar words or documents."
]

# Create term-document matrix using TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(documents)
df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())

# Print matrix
print("TF-IDF Term-Document Matrix:")
print(df)
