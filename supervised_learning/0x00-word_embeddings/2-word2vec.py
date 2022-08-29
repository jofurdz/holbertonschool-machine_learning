#!/usr/bin/env python3
"""module containing function word2vec_model"""
from gensim.models import Word2Vec
import gensim.models


def word2vec_model(sentences, size=100, min_count=5, window=5,
                   negative=5, cbow=True, iterations=5,
                   seed=0, workers=1):
    """creates and trains a gensim word2vec model"""
    model = Word2Vec(sentences=sentences, size=size,
                     min_count=min_count, window=window,
                     negative=negative, iter=iterations,
                     seed=seed, workers=workers)
    model.train(sentences, total_examples=model.corpus_count,
                epochs=iterations, total_words=model.corpus_total_words)

    return model
