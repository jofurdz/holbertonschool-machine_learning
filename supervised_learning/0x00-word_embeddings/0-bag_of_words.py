#!/usr/bin/env python3
"""module containing function bag_of_words"""
from sklearn.feature_extraction.text import CountVectorizer


def bag_of_words(sentences, vocab=None):
    """creates a bag of words embedding matrix"""
    vectorizer = CountVectorizer(vocabulary=vocab)
    newArray = vectorizer.fit_transform(sentences)
    embeddings = newArray.toarray()
    features = vectorizer.get_feature_names()

    return embeddings, features