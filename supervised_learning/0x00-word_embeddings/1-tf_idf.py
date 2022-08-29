#!/usr/bin/env python3
"""module containing function tf_idf"""
from sklearn.feature_extraction.text import TfidfVectorizer


def tf_idf(sentences, vocab=None):
    """creates a TF-IDF embedding"""
    tfidfVectorizer = TfidfVectorizer(vocabulary=vocab)
    newArray = tfidfVectorizer.fit_transform(sentences)
    embeddings = newArray.toarray()
    features = tfidfVectorizer.get_feature_names()
    return embeddings, features
