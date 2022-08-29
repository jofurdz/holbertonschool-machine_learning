#!/usr/bin/env python3
"""module containing function fasttext_model"""
from gensim.models import FastText
import gensim.models


def fasttext_model(sentences, size=100, min_count=5, negative=5, window=5,
                   cbow=True, iterations=5,
                   seed=0, workers=1):
    """creates and trains a fastText model"""
    model = FastText(sentences=sentences, size=size,
                     min_count=min_count,
                     negative=negative, window=window,
                     iter=iterations,
                     seed=seed, workers=workers)
    model.train(sentences, total_examples=model.corpus_count,
                epochs=iterations,
                total_words=model.corpus_total_words)
    return model
